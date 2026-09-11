"""Gera o benchmark interno de aritmética pt-BR (spec G5, RF1).

Saídas determinísticas (semente fixa):
- data/benchmark_matematica.jsonl  → itens com split treino/teste e resposta-ouro
- data/corpus_cot_matematica.txt   → exemplos formatados em CoT p/ ajuste LoRA

Uso: python scripts/prepara_benchmark.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data"
SEMENTE = 2026

COT = "Vamos pensar passo a passo."


def passos_soma(a: int, b: int) -> tuple[list[str], int]:
    return [f"Passo 1: Some {b} ao valor inicial {a}.", f"Passo 2: O total e {a + b}."], a + b


def passos_subtracao(a: int, b: int) -> tuple[list[str], int]:
    return [f"Passo 1: Parta de {a} e remova {b}.", f"Passo 2: Resta {a - b}."], a - b


def passos_multiplicacao(a: int, b: int) -> tuple[list[str], int]:
    return [f"Passo 1: Some {a} {b} vezes.", f"Passo 2: Isso da {a * b}."], a * b


def montar_itens() -> tuple[list[dict], list[dict]]:
    rnd = random.Random(SEMENTE)
    itens: list[dict] = []

    def familia(nome, pares, fn_passos, simbolo):
        pares = sorted(pares)
        rnd.shuffle(pares)
        n_teste = min(60, max(1, len(pares) // 2))  # ≥60 por família no teste
        for split, grupo in (("treino", pares[n_teste:]), ("teste", pares[:n_teste])):
            for a, b in grupo:
                passos, resposta = fn_passos(a, b)
                itens.append(
                    {
                        "id": f"{nome}-{a}-{b}",
                        "familia": nome,
                        "split": split,
                        "enunciado": f"Quanto e {a} {simbolo} {b}?",
                        "cot": "\n".join([COT, *passos, f"Resposta: {resposta}"]),
                        "resposta": resposta,
                    }
                )

    pares_soma = {(a, b) for a in range(1, 21) for b in range(1, 21)}
    pares_sub = {(a, b) for a in range(2, 21) for b in range(1, a)}
    pares_mult = {(a, b) for a in range(2, 10) for b in range(2, 10)}
    familia("soma", pares_soma, passos_soma, "+")
    familia("subtracao", pares_sub, passos_subtracao, "-")
    familia("multiplicacao", pares_mult, passos_multiplicacao, "*")

    teste = [i for i in itens if i["split"] == "teste"]
    return itens, teste


def main() -> int:
    DADOS.mkdir(exist_ok=True)
    itens, teste = montar_itens()
    (DADOS / "benchmark_matematica.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in itens) + "\n", encoding="utf-8"
    )
    corpo_treino = "\n\n".join(f"{i['enunciado']}\n{i['cot']}" for i in itens if i["split"] == "treino")
    (DADOS / "corpus_cot_matematica.txt").write_text(corpo_treino + "\n", encoding="utf-8")
    por_fam = {}
    for i in teste:
        por_fam[i["familia"]] = por_fam.get(i["familia"], 0) + 1
    print(f"[ok] {len(itens)} itens (teste={len(teste)} {por_fam}) -> data/benchmark_matematica.jsonl")
    print(f"[ok] corpus CoT treino: {(DADOS / 'corpus_cot_matematica.txt').stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
