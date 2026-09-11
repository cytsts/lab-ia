"""Comparativo externo (RF5 da G3): erro de reconstrução int8 — nativo vs bitsandbytes.

Uso: .venv\\Scripts\\python scripts\\comparar_bitsandbytes.py
Registra o resultado em .lab-ia/eventos.jsonl (evidência WORM do lab).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import bitsandbytes as bnb  # noqa: E402
import torch  # noqa: E402

from labia.experiments.runner import ultimo_checkpoint  # noqa: E402
from labia.models.quant import dequant_int8, quant_int8  # noqa: E402
from labia.utils.estado import LogEventos  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
MODULO = "blocos.0.mlp.fc1"


def main() -> int:
    ck = torch.load(ultimo_checkpoint(RAIZ / "runs" / "g1-treino-zero"), map_location="cpu", weights_only=True)
    w = ck["modelo"][f"{MODULO}.weight"].float()

    nativa = dequant_int8(quant_int8(w))
    erro_nativa = float(((w - nativa).norm() / w.norm()).item())

    q, estado = bnb.functional.quantize_blockwise(w)
    d = bnb.functional.dequantize_blockwise(q, quant_state=estado)
    erro_bnb = float(((w - d.float()).norm() / w.norm()).item())

    print(f"módulo {MODULO}: erro relativo nativo={erro_nativa:.5f}  bitsandbytes={erro_bnb:.5f}")
    eventos = LogEventos(RAIZ / ".lab-ia" / "eventos.jsonl")
    eventos.registrar(
        "comparacao_bitsandbytes",
        modulo=MODULO,
        erro_relativo_native_int8=round(erro_nativa, 5),
        erro_relativo_bnb_blockwise8=round(erro_bnb, 5),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
