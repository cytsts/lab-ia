"""Testes da bancada B1 — presets, estimativas e YAML gerado (specs/B1.md)."""
from __future__ import annotations

import math

import pytest
import torch
import yaml

from labia.bancada import presets as bp
from labia.models.gpt import ConfigGPT, GPT


# --- contagem de parâmetros bate com o modelo de verdade --------------------

@pytest.mark.parametrize(
    "vocab,dim,camadas,cabecas,janela,especialistas,top_k",
    [
        (4096, 256, 6, 8, 256, 0, 1),
        (2048, 128, 4, 4, 128, 0, 1),
        (8192, 384, 3, 6, 320, 0, 1),
        (4096, 256, 2, 8, 256, 4, 1),
        (4096, 256, 2, 8, 256, 4, 2),
    ],
)
def test_contar_parametros_bate_com_gpt(vocab, dim, camadas, cabecas, janela, especialistas, top_k):
    cfg = ConfigGPT(
        vocab=vocab, dim=dim, camadas=camadas, cabecas=cabecas, janela_ctx=janela,
        n_especialistas=especialistas, top_k=top_k,
    )
    modelo = GPT(cfg)
    calculado = bp.contar_parametros(vocab, dim, camadas, janela, especialistas, top_k)
    assert calculado["total"] == modelo.contar_parametros()
    assert calculado["ativos"] == modelo.contar_parametros_ativos()


def test_moe_tem_ativos_menores_que_total():
    calculado = bp.contar_parametros(4096, 256, 6, 256, 4, 1)
    assert calculado["ativos"] < calculado["total"]


# --- validações do montador de config --------------------------------------

def test_montar_config_preset_equilibrado_reproduz_a_g1():
    cfg = bp.montar_config("meu-run", "data/x/trem.txt")
    assert cfg["modelo"] == {"dim": 256, "camadas": 6, "cabecas": 8, "janela_ctx": 256, "abandono": 0.1}
    assert cfg["passos"] == 2500 and cfg["lote"] == 32


def test_montar_config_aplica_sobrescritas():
    cfg = bp.montar_config("x", "c.txt", sobrescritas={"dim": 128, "lote": 8, "especialistas": 8, "top_k": 2})
    assert cfg["modelo"]["dim"] == 128
    assert cfg["lote"] == 8
    assert cfg["modelo"]["n_especialistas"] == 8
    assert cfg["modelo"]["top_k"] == 2


def test_montar_config_preset_desconhecido_falha():
    with pytest.raises(ValueError, match="preset desconhecido"):
        bp.montar_config("x", "c.txt", preset="turbo")


def test_montar_config_sobrescrita_desconhecida_falha():
    with pytest.raises(ValueError, match="sobrescrita desconhecida"):
        bp.montar_config("x", "c.txt", sobrescritas={"turbo": 1})


@pytest.mark.parametrize(
    "sobrescritas,erro",
    [
        ({"dim": 100, "cabecas": 8}, "divisível"),
        ({"warmup": 5000}, "warmup"),
        ({"stride": 9999}, "stride"),
        ({"especialistas": 2, "top_k": 5}, "top_k"),
        ({"passos": 0}, "passos"),
    ],
)
def test_montar_config_recusa_combinacao_invalida(sobrescritas, erro):
    with pytest.raises(ValueError, match=erro):
        bp.montar_config("x", "c.txt", sobrescritas=sobrescritas)


# --- YAML gerado ------------------------------------------------------------

def test_yaml_faz_round_trip_e_preserva_tipos():
    """Armadilha real: '3e-05' não é float em YAML 1.1 — o treino quebrava no passo 1000."""
    cfg = bp.montar_config("round-trip", "data/x/trem.txt")
    texto = bp.texto_yaml(cfg, ["cabecalho"])
    lido = yaml.safe_load(texto)
    assert lido == cfg
    assert isinstance(lido["minimo_lr"], float)
    assert isinstance(lido["lr"], float)
    assert isinstance(lido["lote"], int)
    assert isinstance(lido["modelo"]["abandono"], float)
    assert lido["modelo"]["dim"] == cfg["modelo"]["dim"]


def test_yaml_float_nunca_sai_sem_ponto():
    assert bp._yaml_float(3e-5) == "3.0e-05"
    assert bp._yaml_float(0.0003) == "0.0003"
    assert bp._yaml_float(1.0) == "1.0"
    assert isinstance(yaml.safe_load("v: " + bp._yaml_float(3e-5))["v"], float)


def test_yaml_escapa_caminho_com_barra_invertida():
    cfg = bp.montar_config("x", r"data\meu\corpus.txt")
    texto = bp.texto_yaml(cfg, [])
    lido = yaml.safe_load(texto)
    assert lido["corpus"] == r"data\meu\corpus.txt"
    assert "\t" not in lido["corpus"]


def test_yaml_tem_comentarios_explicativos():
    cfg = bp.montar_config("x", "c.txt")
    texto = bp.texto_yaml(cfg, ["linha de cabecalho"])
    assert "# linha de cabecalho" in texto
    assert texto.count("#") > 10


# --- estimativas ------------------------------------------------------------

def test_modelo_de_custo_generaliza_para_medicao_fora_da_amostra():
    """Validação cruzada deixando-um-de-fora: erro de previsão de tempo por treino."""
    validacao = bp.validar_modelo_de_custo()
    assert validacao["erro_maximo"] is not None
    assert validacao["erro_maximo"] < 0.35, validacao
    assert set(validacao["erros_relativos"]) == {m["nome"] for m in bp.MEDICOES}


def test_custo_fixo_domina_modelos_pequenos():
    """Achado da bancada: com poucos parâmetros o gargalo é o passo, não a GPU."""
    pequeno = bp.estimar_desempenho(bp.contar_parametros(2048, 128, 3, 256), 3, 128, 256, 32, 100)
    grande = bp.estimar_desempenho(bp.contar_parametros(8192, 512, 8, 512), 8, 512, 512, 32, 100)
    assert pequeno["custo_fixo_s"] / pequeno["segundos_por_passo"] > 0.5
    assert grande["custo_fixo_s"] / grande["segundos_por_passo"] < 0.2


def test_estimativa_com_taxa_informada_ignora_o_modelo():
    estimativa = bp.estimar_desempenho(
        bp.contar_parametros(4096, 256, 6, 256), 6, 256, 256, 32, 1000, tokens_por_s=100_000
    )
    assert estimativa["segundos_por_passo"] == pytest.approx(32 * 256 / 100_000)
    assert "informada pelo usuário" in estimativa["origem_taxa"]


def test_estimativa_vram_cresce_com_lote():
    pequeno = bp.estimar_vram(bp.contar_parametros(4096, 256, 6, 256), 6, 256, 256, 8, 8, 4096)
    grande = bp.estimar_vram(bp.contar_parametros(4096, 256, 6, 256), 6, 256, 256, 64, 8, 4096)
    assert grande["mb_total"] > pequeno["mb_total"]
    assert grande["mb_ativacoes"] > pequeno["mb_ativacoes"]


def test_estimativa_vram_tem_ordem_de_grandeza_do_run_real():
    """Rode real: dim=192, 4 camadas, janela=192, lote=24 → pico medido bem abaixo de 1 GB."""
    vram = bp.estimar_vram(bp.contar_parametros(4096, 192, 4, 192), 4, 192, 192, 24, 6, 4096)
    assert 50 < vram["mb_total"] < 700  # ordem de grandeza, não precisão
