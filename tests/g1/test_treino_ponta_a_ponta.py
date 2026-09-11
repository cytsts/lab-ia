"""Teste de ponta a ponta do treino (spec G1: RF1–RF6, CA1–CA5, CPU, sem GPU)."""
from __future__ import annotations

import torch

from labia.experiments.runner import carregar_estado, ler_metricas, ultimo_checkpoint
from labia.trainer.gerar import gerar_de_checkpoint
from labia.trainer.treino import executar_treino
from common import config_micro, criar_dir_run


def test_treino_converge_e_registra_metricas(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "direto")
    cfg = config_micro(corpus_arquivo, run, lr=3e-3)
    executar_treino(cfg, run, raiz=tmp_path)

    registros = ler_metricas(run)
    assert [r["passo"] for r in registros] == [0, 10, 20]
    assert registros[0]["entropia_unigram"] > 0, "linha de base deve registrar entropia unigram"
    assert registros[2]["loss_val"] < registros[0]["loss_val"] * 0.95, "perda de validação não caiu sobre a linha de base"
    for r in registros[1:]:
        assert {"loss_trem", "lr", "tokens_por_s", "tempo_s", "dispositivo"} <= set(r)

    estado = carregar_estado(run)
    assert estado["concluido"] is True and estado["passo"] == 20
    assert ultimo_checkpoint(run).name == "passo-20.pt"
    assert (run / "tokens" / "tokenizer.json").exists()
    assert (run / "eventos.jsonl").exists()


def test_retomada_exatamente_equivalente(tmp_path, corpus_arquivo):
    """CA2 (forte): 20 passos diretos == 10 passos + kill + retomar 10, passo a passo."""
    cfg = config_micro(corpus_arquivo, None, lr=3e-3)

    dir_a = criar_dir_run(tmp_path / "a")
    cfg_a = dict(cfg, arquivo_eventos=str(tmp_path / "a-eventos.jsonl"))
    executar_treino(cfg_a, dir_a, raiz=tmp_path)

    dir_b = criar_dir_run(tmp_path / "b")
    cfg_b = dict(cfg, arquivo_eventos=str(tmp_path / "b-eventos.jsonl"))
    executar_treino(cfg_b, dir_b, limite=10, raiz=tmp_path)  # simula queda no passo 10
    estado_parcial = carregar_estado(dir_b)
    assert estado_parcial["concluido"] is False and estado_parcial["passo"] == 10

    executar_treino(cfg_b, dir_b, retomar=True, raiz=tmp_path)  # retoma até 20

    reg_a = ler_metricas(dir_a)
    reg_b = ler_metricas(dir_b)
    assert [r["passo"] for r in reg_b] == [0, 10, 20], "retomada não deve repetir métricas antigas"
    apenas_trem = [r for r in reg_a if r["passo"] > 0], [r for r in reg_b if r["passo"] > 0]
    for ra, rb in zip(*apenas_trem):
        assert abs(ra["loss_trem"] - rb["loss_trem"]) < 1e-6
        assert abs(ra["loss_val"] - rb["loss_val"]) < 1e-6
    assert torch.equal(
        torch.load(ultimo_checkpoint(dir_a), weights_only=True)["modelo"]["ln_f.weight"],
        torch.load(ultimo_checkpoint(dir_b), weights_only=True)["modelo"]["ln_f.weight"],
    )


def test_gerar_devolve_texto(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "gera")
    cfg = config_micro(corpus_arquivo, run, passos=300, avaliar_a_cada=150, salvar_a_cada=150, lr=3e-3)
    executar_treino(cfg, run, raiz=tmp_path)
    registros = ler_metricas(run)
    assert registros[-1]["loss_val"] < registros[0]["entropia_unigram"], "300 passos devem superar o preditor unigram"
    texto = gerar_de_checkpoint(run, "A noite", passos_max=12, guloso=True)
    assert isinstance(texto, str) and len(texto.strip()) > 0
    assert any(c.isalpha() for c in texto), f"modelo sub-treinado só cospe espaço: {texto!r}"
