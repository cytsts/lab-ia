"""Testes da biblioteca de arquiteturas (spec B7 — A.1, A.2 e A.3).

Cobrem três coisas distintas:
  * A.1 — cada componente moderno faz o que a matemática diz (rotação preserva norma,
    o produto interno depende só da distância entre posições, SwiGLU não infla o custo);
  * A.2 — as peças se integram ao GPT e o caminho antigo (GPT-2) não mudou;
  * A.3 — a contagem de parâmetros acompanha a arquitetura e config impossível é recusada.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from labia.bancada import presets
from labia.models.componentes import (
    RMSNorm,
    RedeSwiGLU,
    aplicar_rope,
    intermediario_swiglu,
    rope_frequencias,
)
from labia.models.gpt import AtencaoMultiCabeca, ConfigGPT, GPT

RAIZ = Path(__file__).resolve().parents[2]


def cfg_pequena(**extra) -> ConfigGPT:
    base = dict(vocab=128, dim=64, camadas=2, cabecas=4, janela_ctx=32, abandono=0.0)
    base.update(extra)
    return ConfigGPT(**base)


# ---------------------------------------------------------------- A.1

def test_rmsnorm_bate_com_a_formula_e_nao_tem_vies():
    norm = RMSNorm(8)
    assert not hasattr(norm, "bias"), "RMSNorm não tem viés"
    assert torch.allclose(norm.weight, torch.ones(8)), "peso inicial é 1"
    x = torch.randn(2, 5, 8)
    esperado = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * norm.weight
    assert torch.allclose(norm(x), esperado, atol=1e-6)
    # sem subtração de média: somar uma constante muda a saída (LayerNorm não mudaria)
    assert not torch.allclose(norm(x + 3.0), norm(x), atol=1e-3)


def test_rope_preserva_a_norma():
    cos, sen = rope_frequencias(16, 8)
    x = torch.randn(2, 3, 16, 8)
    girado = aplicar_rope(x, cos, sen)
    assert torch.allclose(girado.norm(dim=-1), x.norm(dim=-1), atol=1e-6), "rotação não muda tamanho"
    assert not torch.allclose(girado, x), "mas muda a direção"


def test_rope_depende_so_da_distancia_entre_posicoes():
    """A propriedade que faz o RoPE funcionar: q·k depende de (n−m), não de m e n."""
    dim_cabeca, tokens = 8, 10
    cos, sen = rope_frequencias(tokens, dim_cabeca)
    # A MESMA direção em todas as posições — é o que isola o efeito da posição. Gerar um
    # vetor diferente por posição (primeira versão deste teste) mede outra coisa e a
    # propriedade não tem por que valer. float64 porque a identidade é exata.
    direcao_q = torch.randn(1, 1, 1, dim_cabeca, dtype=torch.float64)
    direcao_k = torch.randn(1, 1, 1, dim_cabeca, dtype=torch.float64)
    q = aplicar_rope(direcao_q.expand(1, 1, tokens, dim_cabeca), cos, sen)
    k = aplicar_rope(direcao_k.expand(1, 1, tokens, dim_cabeca), cos, sen)

    def produto(m: int, n: int) -> float:
        return float((q[0, 0, m] * k[0, 0, n]).sum())

    for m, n in ((0, 0), (1, 3), (2, 5)):
        for deslocamento in (1, 2):
            if n + deslocamento >= tokens:
                continue
            # 1e-6: as tabelas cos/sen são float32 (é o que o modelo usa), então a
            # identidade vale até a precisão delas — medido 5e-8 de diferença
            assert produto(m, n) == pytest.approx(produto(m + deslocamento, n + deslocamento), abs=1e-6)


def test_rope_recusa_dimensao_impar():
    with pytest.raises(ValueError, match="par"):
        rope_frequencias(8, 7)


@pytest.mark.parametrize("dim", [64, 192, 256, 384])
def test_swiglu_fica_proximo_do_custo_da_ffn_densa(dim):
    """8/3·dim existe justamente para o SwiGLU não ficar 50% maior que a FFN densa."""
    densa = 8 * dim * dim + 5 * dim
    swiglu = 3 * dim * intermediario_swiglu(dim)
    assert 0.95 <= swiglu / densa <= 1.05, f"dim {dim}: razão {swiglu / densa:.3f}"


def test_swiglu_nao_tem_vies_e_tem_tres_matrizes():
    rede = RedeSwiGLU(cfg_pequena(dim=64))
    assert not any(p.dim() == 1 for p in rede.parameters()), "nenhum viés"
    assert sum(p.numel() for p in rede.parameters()) == 3 * 64 * rede.intermediario
    x = torch.randn(2, 5, 64)
    y, aux = rede(x)
    assert y.shape == x.shape and float(aux) == 0.0


def test_gqa_corta_pela_metade_os_parametros_de_kv():
    mha = AtencaoMultiCabeca(cfg_pequena(cabecas=4))
    gqa = AtencaoMultiCabeca(cfg_pequena(cabecas=4, n_cabecas_kv=2))
    assert mha.kv_dim == 64 and gqa.kv_dim == 32
    assert gqa.cabecas_kv == 2
    assert gqa.qkv.weight.numel() < mha.qkv.weight.numel()
    # a diferença é exatamente a metade de K/V: 2*dim*(dim - kv_dim)
    assert mha.qkv.weight.numel() - gqa.qkv.weight.numel() == 2 * 64 * (64 - 32)
    x = torch.randn(2, 5, 64)
    assert gqa(x).shape == mha(x).shape


def test_gqa_recusa_grupo_que_nao_fecha():
    # dim 96 é divisível por 3, então o erro que sobra é o do agrupamento de cabeças
    with pytest.raises(ValueError, match="múltiplo"):
        AtencaoMultiCabeca(cfg_pequena(dim=96, cabecas=4, n_cabecas_kv=3))
    # dim não divisível: a mensagem fala de dim, não de cabeças
    with pytest.raises(ValueError, match="dim"):
        AtencaoMultiCabeca(cfg_pequena(dim=64, cabecas=4, n_cabecas_kv=3))


# ---------------------------------------------------------------- A.2

@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("pos", ["aprendido", "rope"])
def test_gpt_roda_e_aprende_em_todas_as_combinacoes(norm, mlp, pos):
    torch.manual_seed(0)
    modelo = GPT(cfg_pequena(norm=norm, mlp=mlp, pos=pos))
    modelo.init_pesos(0)
    idx = torch.randint(0, 128, (2, 12))
    logits, perda = modelo(idx, alvos=idx.clone())
    assert logits.shape == (2, 12, 128)
    perda.backward()
    aprenderam = [p for p in modelo.parameters() if p.requires_grad and p.grad is not None]
    assert aprenderam and any(float(p.grad.abs().sum()) > 0 for p in aprenderam)


def test_rope_dispensa_o_embedding_de_posicao():
    antigo = GPT(cfg_pequena())
    moderno = GPT(cfg_pequena(pos="rope"))
    assert antigo.wpe is not None and moderno.wpe is None
    assert antigo.contar_parametros() - moderno.contar_parametros() == 32 * 64  # janela x dim
    assert "wpe.weight" in antigo.state_dict() and "wpe.weight" not in moderno.state_dict()
    # tabela de RoPE é derivada da config: fica fora do checkpoint (buffer não persistente)
    assert any("_rope_cos" in n for n, _ in moderno.named_buffers())
    assert not any("_rope" in n for n in moderno.state_dict())


def test_padrao_mantem_o_state_dict_do_gpt2():
    """Retrocompatibilidade estrutural: o caminho antigo não ganhou nem perdeu chave."""
    modelo = GPT(cfg_pequena())
    chaves = set(modelo.state_dict())
    esperadas = {"wte.weight", "wpe.weight"}
    for i in range(2):
        esperadas |= {
            f"blocos.{i}.ln1.weight", f"blocos.{i}.ln1.bias",
            f"blocos.{i}.ln2.weight", f"blocos.{i}.ln2.bias",
            f"blocos.{i}.atencao.qkv.weight", f"blocos.{i}.atencao.proj.weight",
            f"blocos.{i}.mlp.fc1.weight", f"blocos.{i}.mlp.fc1.bias",
            f"blocos.{i}.mlp.fc2.weight", f"blocos.{i}.mlp.fc2.bias",
        }
    esperadas |= {"ln_f.weight", "ln_f.bias", "cabeca.weight"}  # cabeça atada ao embedding
    assert chaves == esperadas


def test_checkpoint_real_do_laboratorio_ainda_carrega():
    """Um checkpoint de verdade, gravado antes desta mudança, tem de carregar estrito."""
    ckpts = sorted((RAIZ / "runs" / "g1-treino-zero" / "ckpt").glob("passo-*.pt"))
    if not ckpts:
        pytest.skip("run g1-treino-zero não está neste workspace")
    dados = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    cfg = ConfigGPT.de_dict(dados["config_modelo"])
    modelo = GPT(cfg)
    faltando = modelo.load_state_dict(dados["modelo"], strict=True)
    assert not faltando.missing_keys and not faltando.unexpected_keys
    with torch.no_grad():
        logits, _ = modelo(torch.tensor([[1, 2, 3]]))
    assert logits.shape[-1] == cfg.vocab


# ---------------------------------------------------------------- A.3

@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("mlp", ["gelu", "swiglu"])
@pytest.mark.parametrize("pos,kv", [("aprendido", 0), ("rope", 2)])
def test_contagem_de_parametros_bate_com_o_modelo(norm, mlp, pos, kv):
    cfg = cfg_pequena(norm=norm, mlp=mlp, pos=pos, n_cabecas_kv=kv)
    real = GPT(cfg).contar_parametros()
    estimado = presets.contar_parametros(
        128, 64, 2, 32, 0, 1,
        norm=norm, pos=pos, mlp=mlp, cabecas=4, n_cabecas_kv=kv,
    )
    assert estimado["total"] == real


def test_contagem_no_moe_tambem_bate():
    cfg = cfg_pequena(n_especialistas=4, top_k=2, norm="rmsnorm", mlp="swiglu")
    modelo = GPT(cfg)
    estimado = presets.contar_parametros(
        128, 64, 2, 32, 4, 2, norm="rmsnorm", mlp="swiglu", cabecas=4
    )
    assert estimado["total"] == modelo.contar_parametros()
    assert estimado["ativos"] == modelo.contar_parametros_ativos()


def test_preset_moderna_e_menor_que_o_gpt2_e_gera_config_aceita():
    gpt2 = presets.montar_config("a", "c.txt", preset="equilibrado")
    moderna = presets.montar_config("b", "c.txt", preset="moderna")
    p_gpt2 = presets.contar_parametros(4096, 256, 6, 256, cabecas=8)
    p_moderna = presets.contar_parametros(
        4096, 256, 6, 256, cabecas=8, norm="rmsnorm", pos="rope", mlp="swiglu", n_cabecas_kv=2
    )
    assert p_moderna["total"] < p_gpt2["total"], "RMSNorm + RoPE + GQA tiram parâmetros"
    assert moderna["modelo"]["norm"] == "rmsnorm"
    # a config gerada é aceita pelo modelo de verdade
    modelo = GPT(ConfigGPT.de_dict({"vocab": 4096, **moderna["modelo"]}))
    assert modelo.contar_parametros() == p_moderna["total"]
    assert gpt2["modelo"].get("norm") is None  # o preset antigo não ganhou campos novos


@pytest.mark.parametrize(
    "modelo,erro",
    [
        ({"dim": 64, "cabecas": 4, "janela_ctx": 32, "pos": "rope", "dim_cabeca_impar": True}, "dim_cabeca par"),
        ({"dim": 64, "cabecas": 4, "janela_ctx": 32, "n_cabecas_kv": 3}, "múltiplo"),
        ({"dim": 64, "cabecas": 4, "janela_ctx": 32, "norm": "batchnorm"}, "norm desconhecido"),
        ({"dim": 64, "cabecas": 4, "janela_ctx": 32, "pos": "senoidal"}, "pos desconhecido"),
        ({"dim": 64, "cabecas": 4, "janela_ctx": 32, "mlp": "relu"}, "mlp desconhecido"),
    ],
)
def test_validacao_recusa_arquitetura_impossivel(modelo, erro):
    """A mensagem tem de dizer o conserto, não só que está errado."""
    modelo = dict(modelo)
    modelo.pop("dim_cabeca_impar", None)
    if erro == "dim_cabeca par":
        modelo.update(dim=96, cabecas=4)  # 96/4 = 24 é par; força ímpar com dim 100/4 = 25
        modelo.update(dim=100, cabecas=4)
    cfg = {
        "nome": "x", "modelo": modelo, "passos": 10, "lote": 2, "vocab_bpe": 256,
        "warmup": 1, "lr": 1e-3, "minimo_lr": 1e-4, "stride": 0,
    }
    with pytest.raises(ValueError, match=erro):
        presets._validar(cfg)
