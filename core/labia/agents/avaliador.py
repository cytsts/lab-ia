"""Agente Avaliador: mede, compara e ranqueia runs (spec G9)."""
from __future__ import annotations

import json
from pathlib import Path

from ..experiments.runner import ler_metricas, ultimo_checkpoint
from ..reasoning.runner import executar_benchmark
from ..trainer.gerar import gerar_de_checkpoint
from .base import AgenteBase, Skill


def _resumo_val(run_dir: Path) -> dict | None:
    registros = [r for r in ler_metricas(run_dir) if r.get("passo") and r.get("loss_val") is not None]
    if not registros:
        return None
    vals = [r["loss_val"] for r in registros]
    return {"min": min(vals), "final": vals[-1], "passos": registros[-1]["passo"], "curvas": len(registros)}


class AgenteAvaliador(AgenteBase):
    nome = "avaliador"
    papel = "avalia modelos, roda benchmarks e compara corridas"
    escopo = "somente leitura de runs; nunca treina nem apaga"

    def definir_skills(self) -> list[Skill]:
        return [
            Skill(
                "evaluate_model",
                "perda de validação + amostra gerada de um run",
                "evaluate_model(run: str, prompt: str = 'Uma noite', passos_max: int = 24) -> dict",
                self.evaluate_model,
            ),
            Skill(
                "benchmark",
                "roda o benchmark de raciocínio numa estratégia",
                "benchmark(run: str, estrategia: str = 'cot', limite: int | None = None) -> dict",
                self.benchmark,
            ),
            Skill(
                "compare_runs",
                "compara perdas de validação entre dois runs e declara vencedor",
                "compare_runs(run_a: str, run_b: str) -> dict",
                self.compare_runs,
            ),
        ]

    def evaluate_model(self, run: str, prompt: str = "Uma noite", passos_max: int = 24) -> dict:
        run_dir = self.raiz / "runs" / run
        if ultimo_checkpoint(run_dir) is None:
            raise FileNotFoundError(f"run {run!r} sem checkpoint")
        resumo = _resumo_val(run_dir) or {}
        amostra = gerar_de_checkpoint(run_dir, prompt, passos_max=passos_max, guloso=True)
        return {"run": run, **resumo, "amostra": amostra[:240]}

    def benchmark(self, run: str, estrategia: str = "cot", limite: int | None = None) -> dict:
        rel = executar_benchmark(
            self.raiz / "runs" / run, estrategia, limite=limite, raiz=self.raiz
        )
        return {
            "run": run, "estrategia": estrategia,
            "acuracia_global": rel["acuracia_global"],
            "taxa_resposta_valida": rel["taxa_resposta_valida"],
            "itens": rel["itens"],
        }

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        a = _resumo_val(self.raiz / "runs" / run_a)
        b = _resumo_val(self.raiz / "runs" / run_b)
        if not a or not b:
            raise ValueError("ambos os runs precisam de métricas com passo>0 e loss_val")
        vencedor = run_a if a["min"] <= b["min"] else run_b
        return {
            run_a: a, run_b: b, "vencedor": vencedor,
            "delta_min": round(abs(a["min"] - b["min"]), 4),
            "criterio": "menor loss_val mínimo",
        }
