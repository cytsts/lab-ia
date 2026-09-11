"""Fixtures G2: base micro treinada (sessão) + corpus-tarefa de outro domínio."""
from __future__ import annotations

from pathlib import Path

import pytest

from common import FRASES_TAREFA, config_micro, criar_dir_run, gerar_corpus, gerar_tarefa
from labia.trainer.treino import executar_treino

__all__ = ["FRASES_TAREFA", "base_micro", "tarefa_arquivo", "base_g1_real", "ajuste_g2_real", "tarefa_real"]

RAIZ_LAB = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def base_micro(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("baseg2")
    corpus = raiz / "corpus.txt"
    corpus.write_text(gerar_corpus(), encoding="utf-8")
    run = criar_dir_run(raiz / "base")
    cfg = config_micro(corpus, run, passos=600, avaliar_a_cada=300, salvar_a_cada=300, lr=3e-3)
    executar_treino(cfg, run, raiz=raiz)
    return run


@pytest.fixture()
def tarefa_arquivo(tmp_path):
    destino = tmp_path / "tarefa_ciencia.txt"
    destino.write_text(gerar_tarefa(), encoding="utf-8")
    return destino


@pytest.fixture(scope="session")
def base_g1_real():
    d = RAIZ_LAB / "runs" / "g1-treino-zero"
    if not d.exists() or not list((d / "ckpt").glob("passo-*.pt")):
        pytest.skip("run real da G1 ausente (rode o treino primeiro)")
    return d


@pytest.fixture(scope="session")
def ajuste_g2_real():
    d = RAIZ_LAB / "runs" / "g2-ajuste-ciencia"
    if not (d / "adaptador" / "meta.json").exists():
        pytest.skip("ajuste real da G2 ausente (rode lab-ia ajustar primeiro)")
    return d


@pytest.fixture(scope="session")
def tarefa_real():
    d = RAIZ_LAB / "data" / "tarefa_ciencia.txt"
    if not d.exists():
        pytest.skip("corpus-tarefa real ausente (rode scripts/prepara_tarefa_ciencia.py)")
    return d
