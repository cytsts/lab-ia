"""API interna (FastAPI) para a camada visual TS: runs, métricas, eventos."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException

_ID_OK = re.compile(r"[A-Za-z0-9._-]+")


def _validar_run_id(run_id: str) -> str:
    if not _ID_OK.fullmatch(run_id) or run_id in (".", ".."):
        raise HTTPException(status_code=400, detail="run-id inválido")
    return run_id


def criar_app(raiz: Path | str = ".") -> FastAPI:
    raiz = Path(raiz)
    app = FastAPI(title="Lab-IA API", version="0.1.0")

    @app.get("/corre")
    def listar_runs() -> list[dict]:
        pasta = raiz / "runs"
        if not pasta.exists():
            return []
        saida = []
        for d in sorted(pasta.iterdir()):
            estado = d / "estado.json"
            if d.is_dir() and estado.exists():
                registro = json.loads(estado.read_text(encoding="utf-8"))
                registro["run_id"] = d.name
                saida.append(registro)
        return saida

    @app.get("/corre/{run_id}/metricas")
    def metricas(run_id: str) -> list[dict]:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "metricas.jsonl"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run ou métricas não encontrados")
        return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]

    @app.get("/corre/{run_id}/estado")
    def estado(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "estado.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run não encontrado")
        return json.loads(caminho.read_text(encoding="utf-8"))

    @app.get("/eventos")
    def eventos(desde: int = 0) -> list[dict]:
        caminho = raiz / ".lab-ia" / "eventos.jsonl"
        if not caminho.exists():
            return []
        linhas = caminho.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in linhas[desde:] if l.strip()]

    return app
