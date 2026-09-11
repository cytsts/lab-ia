"""Fixtures G5: micro-base + ajuste treinado no formato CoT (aritmética soma 1..9)."""
from __future__ import annotations

import json
import random

import pytest

from common import criar_dir_run, gerar_corpus
from labia.trainer.ajuste import executar_ajuste
from labia.trainer.treino import executar_treino

COT = "Vamos pensar passo a passo."


def itens_soma(semente: int = 99):
    rnd = random.Random(semente)
    pares = [(a, b) for a in range(1, 10) for b in range(1, 10)]
    rnd.shuffle(pares)
    itens = []
    for i, (a, b) in enumerate(pares):
        itens.append(
            {
                "id": f"soma-{a}-{b}",
                "familia": "soma",
                "split": "teste" if i < 10 else "treino",
                "enunciado": f"Quanto e {a} + {b}?",
                "cot": f"{COT}\nPasso 1: Some {b} ao valor inicial {a}.\nPasso 2: O total e {a + b}.\nResposta: {a + b}",
                "resposta": a + b,
            }
        )
    return itens


@pytest.fixture(scope="session")
def ajuste_cot_micro(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("g5")
    corpus = raiz / "corpus.txt"
    corpus.write_text(gerar_corpus(), encoding="utf-8")
    base = criar_dir_run(raiz / "base")
    executar_treino(
        {
            "nome": "micro96", "corpus": str(corpus), "vocab_bpe": 384,
            "modelo": {"dim": 96, "camadas": 4, "cabecas": 4, "janela_ctx": 96, "abandono": 0.0},
            "passos": 900, "lote": 4, "avaliar_a_cada": 450, "salvar_a_cada": 450, "iters_avaliacao": 5,
            "lr": 3e-3, "minimo_lr": 3e-4, "warmup": 40, "peso_decay": 0.0, "grad_clip": 1.0,
            "semente": 42, "dispositivo": "cpu", "arquivo_eventos": str(raiz / "evb.jsonl"),
        },
        base,
        raiz=raiz,
    )

    itens = itens_soma()
    bench = raiz / "benchmark_micro.jsonl"
    bench.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in itens) + "\n", encoding="utf-8")
    corpo = raiz / "cot.txt"
    corpo.write_text("\n\n".join(f"{i['enunciado']}\n{i['cot']}" for i in itens if i["split"] == "treino") + "\n", encoding="utf-8")

    run = criar_dir_run(raiz / "ajuste")
    executar_ajuste(
        {
            "nome": "cot-micro",
            "base": str(base),
            "corpus_tarefa": str(corpo),
            "r": 16,
            "alpha": 32,
            "tipo": "lora",
            "bits": 8,
            "passos": 6000,
            "lote": 6,
            "stride": 8,
            "avaliar_a_cada": 2000,
            "salvar_a_cada": 2000,
            "iters_avaliacao": 5,
            "lr": 1.2e-2,
            "minimo_lr": 1.2e-3,
            "warmup": 100,
            "grad_clip": 1.0,
            "semente": 42,
            "dispositivo": "cpu",
            "arquivo_eventos": str(raiz / "ev.jsonl"),
        },
        run,
        raiz=raiz,
    )
    return {"run": run, "base": base, "benchmark": bench}
