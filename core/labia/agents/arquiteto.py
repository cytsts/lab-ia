"""Agente Arquiteto: sugere arquiteturas a partir de orçamento e dados (spec G9)."""
from __future__ import annotations

from pathlib import Path

from ..experiments.runner import ler_metricas
from ..models.gpt import ConfigGPT
from .base import AgenteBase, Skill


def _contar(cfg: ConfigGPT) -> int:
    """Contagem analítica de parâmetros (sem materializar tensores)."""
    d, v, L = cfg.dim, cfg.vocab, cfg.camadas
    atencao = 3 * d * d + d * d + 2 * d
    mlp_denso = d * 4 * d + 4 * d * d + 2 * d
    lns = 4 * d
    if cfg.n_especialistas > 1:
        um_esp = d * 4 * d + 4 * d * d + 2 * d
        roteador = d * cfg.n_especialistas
        mlp = cfg.n_especialistas * um_esp + roteador
    else:
        mlp = mlp_denso
    emb = v * d + cfg.janela_ctx * d
    return emb + (atencao + mlp + lns) * L + 2 * d


class AgenteArquiteto(AgenteBase):
    nome = "arquiteto"
    papel = "propõe arquiteturas viáveis para o hardware e o corpus disponíveis"
    escopo = "analisa configs e métricas; não executa treinos"

    def definir_skills(self) -> list[Skill]:
        return [
            Skill(
                "suggest_architecture",
                "sugere config de treino que caiba na VRAM para um corpus",
                "suggest_architecture(corpus_bytes: int, vram_gb: float = 8.0, alvo_epocas: float = 2.0) -> dict",
                self.suggest_architecture,
            ),
            Skill(
                "optimize_moe",
                "diagnostica uso de especialistas de um run MoE e sugere ajustes",
                "optimize_moe(run: str) -> dict",
                self.optimize_moe,
            ),
        ]

    def suggest_architecture(
        self, corpus_bytes: int, vram_gb: float = 8.0, alvo_epocas: float = 2.0
    ) -> dict:
        orcamento_params = int(vram_gb * 0.6 * 2**30 / 9)  # fp32 pesos + Adam ≈ 9 B/param
        candidatos = []
        for dim in (128, 192, 256, 320, 384):
            for camadas in (4, 6, 8):
                cfg = ConfigGPT(vocab=4096, dim=dim, camadas=camadas, cabecas=max(1, dim // 64), janela_ctx=256)
                n = _contar(cfg)
                if n <= orcamento_params:
                    candidatos.append((n, cfg))
        if not candidatos:
            raise ValueError(f"nenhuma arquitetura cabe no orçamento de {vram_gb} GB")
        n, cfg = max(candidatos, key=lambda t: t[0])

        tokens_estimados = int(corpus_bytes / 2.2)  # BPE pt-BR ≈ 2,2 bytes/token
        janelas = max(1, tokens_estimados // cfg.janela_ctx)
        lote = 32 if vram_gb >= 8 else 8
        passos_por_epoch = max(1, janelas // lote)
        passos = max(500, int(alvo_epocas * passos_por_epoch) // 50 * 50)
        return {
            "modelo": {
                "dim": cfg.dim, "camadas": cfg.camadas, "cabecas": cfg.cabecas,
                "janela_ctx": cfg.janela_ctx, "abandono": 0.1,
            },
            "vocab_bpe": cfg.vocab,
            "passos": passos,
            "lote": lote,
            "parametros_estimados": n,
            "orcamento_params": orcamento_params,
            "justificativa": (
                f"maior modelo que cabe em {vram_gb:.0f} GB com folga de otimizador; "
                f"{tokens_estimados:,} tokens estimados p/ corpus de {corpus_bytes:,} B; "
                f"{passos} passos ≈ {alvo_epocas} épocas"
            ),
        }

    def optimize_moe(self, run: str) -> dict:
        registros = [r for r in ler_metricas(self.raiz / "runs" / run) if r.get("uso_especialistas")]
        if not registros:
            raise ValueError(f"run {run!r} não tem métricas de roteamento (não é MoE?)")
        ultimo = registros[-1]["uso_especialistas"]
        auxes = [r["aux_router"] for r in registros if r.get("aux_router") is not None]
        sugestoes: list[str] = []
        max_uso = max(ultimo)
        if max_uso > 0.6:
            sugestoes.append("roteador colapsando: aumente coef_auxiliar (ex.: 0.01 → 0.03) e re-treine")
        elif len(ultimo) > 1 and max_uso < 0.45 and min(ultimo) > 0.1:
            sugestoes.append("carga saudável; considere reduzir n_especialistas se a perda não bater o denso")
        if auxes and auxes[-1] > 1.5:
            sugestoes.append(f"aux ainda alto ({auxes[-1]:.2f} vs piso ~1.0): mais passos ou coef maior")
        if not sugestoes:
            sugestoes.append("diagnóstico neutro: manter config atual")
        return {
            "run": run,
            "uso_final": ultimo,
            "aux_final": auxes[-1] if auxes else None,
            "n_especialistas": len(ultimo),
            "sugestoes": sugestoes,
        }
