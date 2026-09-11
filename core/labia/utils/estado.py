"""Utilitários de estado durável: escrita atômica em JSON e log de eventos append-only.

Projeto para SSD: rename atômico (os.replace) e um único append por evento.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def salvar_json_atomico(caminho: Path, obj: object) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, caminho)


class LogEventos:
    """Log append-only para auditoria e replay (G8)."""

    def __init__(self, caminho: Path):
        self.caminho = Path(caminho)

    def registrar(self, tipo: str, **dados) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        linha = {"tempo": agora_iso(), "tipo": tipo}
        linha.update(dados)
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    def ler(self) -> list[dict]:
        if not self.caminho.exists():
            return []
        return [json.loads(l) for l in self.caminho.read_text(encoding="utf-8").splitlines() if l.strip()]
