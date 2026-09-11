"""Fixtures G3: micro-base treinada + corpus de avaliação (tarefa)."""
from __future__ import annotations

import pytest

from common import config_micro, criar_dir_run, gerar_corpus, gerar_tarefa
from labia.trainer.treino import executar_treino


@pytest.fixture(scope="session")
def base_g3(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("baseg3")
    corpus = raiz / "corpus.txt"
    corpus.write_text(gerar_corpus(), encoding="utf-8")
    run = criar_dir_run(raiz / "base")
    executar_treino(config_micro(corpus, run, passos=600, avaliar_a_cada=300, salvar_a_cada=300, lr=3e-3), run, raiz=raiz)
    tarefa = raiz / "tarefa.txt"
    tarefa.write_text(gerar_tarefa(), encoding="utf-8")
    return run, tarefa
