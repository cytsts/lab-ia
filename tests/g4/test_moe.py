"""Testes MoE (spec G4: CA1–CA6, CPU)."""
from __future__ import annotations

import pytest
import torch

from common import config_micro, criar_dir_run
from labia.experiments.runner import ler_metricas, ultimo_checkpoint
from labia.models.gpt import CamadaMoE, ConfigGPT, GPT
from labia.trainer.gerar import gerar_de_checkpoint
from labia.trainer.treino import executar_treino


def _cfg_moe(**over):
    base = dict(vocab=64, dim=32, camadas=2, cabecas=2, janela_ctx=16, abandono=0.0, n_especialistas=4, top_k=1)
    base.update(over)
    return ConfigGPT(**base)


def test_dense_continua_igual():
    torch.manual_seed(0)
    m = GPT(ConfigGPT(vocab=64, dim=32, camadas=2, cabecas=2, janela_ctx=16, abandono=0.0))
    assert m.mapa_uso_especialistas() is None and m.ultimo_aux is None
    idx = torch.randint(0, 64, (2, 8))
    assert m.contar_parametros_ativos() == m.contar_parametros()
    logits, perda = m(idx, alvos=idx.clone())
    assert logits.shape == (2, 8, 64) and perda.dim() == 0


def test_top1_rota_exatamente_um_especialista():
    torch.manual_seed(0)
    m = GPT(_cfg_moe())
    camada = m.blocos[0].mlp
    assert isinstance(camada, CamadaMoE) and len(camada.especialistas) == 4
    x = torch.randn(3, 5, 32)
    with torch.no_grad():
        g, aux = camada.roteador(x.reshape(-1, 32))
    assert torch.equal(g.gt(0).float().sum(-1), torch.ones(15))
    assert torch.allclose(g.sum(-1), torch.ones(15))
    assert 0.2 <= float(aux) <= 4.0 + 1e-6  # n·Σ f·P̄ com n=4: limitado por n, mínimo prático ~1


def test_gradientes_roteador_e_especialistas():
    torch.manual_seed(0)
    m = GPT(_cfg_moe(top_k=2))
    idx = torch.randint(0, 64, (2, 8))
    _, perda = m(idx, alvos=idx.clone())
    perda.backward()
    assert m.blocos[0].mlp.roteador.peso.weight.grad is not None
    com_grad = [p for e in m.blocos[0].mlp.especialistas for p in e.parameters() if p.grad is not None]
    assert com_grad, "top-2 deveria dar gradiente a especialistas"


def test_parametros_ativos_correspondem_a_topk():
    torch.manual_seed(0)
    m = GPT(_cfg_moe(n_especialistas=4, top_k=1))
    total = m.contar_parametros()
    ativos = m.contar_parametros_ativos()
    por_esp = sum(p.numel() for p in m.blocos[0].mlp.especialistas[0].parameters())
    assert ativos < total
    assert total - ativos == 3 * por_esp * 2  # (n−k)=3 × 2 camadas MoE


def _config_moe(corpus, run, **over):
    cfg = config_micro(corpus, run, passos=120, avaliar_a_cada=60, salvar_a_cada=60, lr=3e-3)
    cfg["modelo"] = dict(cfg["modelo"], n_especialistas=2, top_k=1, coef_auxiliar=0.02)
    cfg.update(over)
    return cfg


def test_treino_moe_registra_roteamento(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "moe")
    executar_treino(_config_moe(corpus_arquivo, run), run, raiz=tmp_path)
    registros = ler_metricas(run)
    assert [r["passo"] for r in registros] == [0, 60, 120]
    for r in registros[1:]:
        assert r["aux_router"] is not None
        uso = r["uso_especialistas"]
        assert uso is not None and len(uso) == 2 and abs(sum(uso) - 1.0) < 0.01
    assert max(registros[-1]["uso_especialistas"]) < 0.6, "CA2/CA3: especialista com >60% do tráfego"
    assert registros[-1]["aux_router"] <= registros[1]["aux_router"] + 1e-9, "CA3: aux divergiu"
    assert registros[-1]["aux_router"] <= 1.15, "CA3: fora do piso de balanceamento"


def test_retomada_moe_identica(tmp_path, corpus_arquivo):
    cfg_a = _config_moe(corpus_arquivo, None, arquivo_eventos=str(tmp_path / "ea.jsonl"))
    dir_a = criar_dir_run(tmp_path / "a")
    executar_treino(cfg_a, dir_a, raiz=tmp_path)

    cfg_b = _config_moe(corpus_arquivo, None, arquivo_eventos=str(tmp_path / "eb.jsonl"))
    dir_b = criar_dir_run(tmp_path / "b")
    executar_treino(cfg_b, dir_b, limite=60, raiz=tmp_path)
    executar_treino(cfg_b, dir_b, retomar=True, raiz=tmp_path)

    reg_a = [r for r in ler_metricas(dir_a) if r["passo"] > 0]
    reg_b = [r for r in ler_metricas(dir_b) if r["passo"] > 0]
    for ra, rb in zip(reg_a, reg_b):
        assert abs(ra["loss_val"] - rb["loss_val"]) < 1e-6


def test_gerar_no_moe(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "gera")
    executar_treino(_config_moe(corpus_arquivo, run, passos=300, avaliar_a_cada=150, salvar_a_cada=150), run, raiz=tmp_path)
    texto = gerar_de_checkpoint(run, "A noite", passos_max=12, guloso=True)
    assert isinstance(texto, str) and any(c.isalpha() for c in texto)
