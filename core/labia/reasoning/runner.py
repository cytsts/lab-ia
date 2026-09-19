"""Executor de benchmark de raciocínio + relatório comparativo (spec G5, RF3/RF5)."""
from __future__ import annotations

import json
from pathlib import Path

from ..trainer.gerar import carregar_para_geracao
from .estrategias import carregar_benchmark, responder_cot, responder_direta, responder_tot

RESPONDER = {"direta": responder_direta, "cot": responder_cot, "tot": responder_tot}


def executar_benchmark(
    run_dir: Path | str,
    estrategia: str,
    benchmark: Path | str = "data/benchmark_matematica.jsonl",
    split: str = "teste",
    semente: int = 1234,
    limite: int | None = None,
    raiz: Path | str = ".",
    dispositivo: str = "auto",
) -> dict:
    if estrategia not in RESPONDER:
        raise ValueError(f"estratégia {estrategia!r} inválida (use {sorted(RESPONDER)})")
    run_dir = Path(run_dir)
    bench_caminho = Path(benchmark) if Path(benchmark).is_absolute() else Path(raiz) / benchmark
    itens = carregar_benchmark(bench_caminho, split=split, limite=limite)
    modelo, tok, disp = carregar_para_geracao(run_dir, dispositivo=dispositivo)
    responde = RESPONDER[estrategia]

    acertos = {"__global__": 0}
    totais = {"__global__": 0}
    amostra_erros: list[dict] = []
    validas = 0
    for pos, item in enumerate(itens):
        semente_item = semente + pos
        resposta, texto = responde(modelo, tok, disp, item["enunciado"], semente=semente_item)
        acerto = resposta == item["resposta"]
        validas += int(resposta is not None)
        totais["__global__"] += 1
        totais[item["familia"]] = totais.get(item["familia"], 0) + 1
        if acerto:
            acertos["__global__"] += 1
            acertos[item["familia"]] = acertos.get(item["familia"], 0) + 1
        elif len(amostra_erros) < 10:
            amostra_erros.append({"id": item["id"], "esperado": item["resposta"], "obtido": resposta, "texto": texto[:160]})

        parcial = acertos["__global__"] / totais["__global__"]
        print(
            f"[benchmark] item {pos + 1} de {len(itens)} — acurácia parcial: {parcial * 100:.1f}% ({acertos['__global__']}/{totais['__global__']})",
            flush=True,
        )
        try:
            (run_dir / "benchmark_progresso.json").write_text(
                json.dumps(
                    {
                        "item_atual": pos + 1,
                        "total_itens": len(itens),
                        "acertos": acertos["__global__"],
                        "acuracia_parcial": round(parcial, 4),
                        "concluido": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    relatorio = {
        "run": run_dir.name,
        "estrategia": estrategia,
        "split": split,
        "semente": semente,
        "itens": len(itens),
        "acuracia_global": round(acertos["__global__"] / max(1, len(itens)), 4),
        "acuracia_por_familia": {
            f: round(acertos.get(f, 0) / totais[f], 4) for f in sorted(totais) if f != "__global__"
        },
        "taxa_resposta_valida": round(validas / max(1, len(itens)), 4),
        "amostra_erros": amostra_erros,
    }
    destino = run_dir / f"benchmark-{estrategia}-{split}.json"
    destino.write_text(json.dumps(relatorio, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    try:
        (run_dir / "benchmark_progresso.json").write_text(
            json.dumps(
                {
                    "item_atual": len(itens),
                    "total_itens": len(itens),
                    "acertos": acertos["__global__"],
                    "acuracia_parcial": relatorio["acuracia_global"],
                    "concluido": True,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    with open(run_dir / "metricas.jsonl", "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "passo": None,
                    "tipo": "benchmark",
                    "estrategia": estrategia,
                    "split": split,
                    "semente": semente,
                    "acuracia_global": relatorio["acuracia_global"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return relatorio


def comparar_estrategias(run_dir: Path | str, estrategias: tuple[str, ...] = ("direta", "cot", "tot"), **kw) -> dict:
    rels = {}
    for e in estrategias:
        rels[e] = executar_benchmark(run_dir, e, **kw)
    run_dir = Path(run_dir)
    comparativo = {
        "run": run_dir.name,
        "estrategias": rels,
        "cot_menos_direta_pp": round((rels["cot"]["acuracia_global"] - rels["direta"]["acuracia_global"]) * 100, 2)
        if "cot" in rels and "direta" in rels
        else None,
        "tot_menos_cot_pp": round((rels["tot"]["acuracia_global"] - rels["cot"]["acuracia_global"]) * 100, 2)
        if "tot" in rels and "cot" in rels
        else None,
    }
    (run_dir / "comparativo.json").write_text(
        json.dumps(comparativo, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )
    return comparativo
