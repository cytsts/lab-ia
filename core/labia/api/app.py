"""API interna (FastAPI) para a camada visual TS: runs, métricas, eventos, execuções."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ID_OK = re.compile(r"[A-Za-z0-9._-]+")
ACOES = {
    "train": lambda b, raiz: ["train", "--config", _config_segura(b.config, raiz), "--raiz", str(raiz)],
    "ajustar": lambda b, raiz: ["ajustar", "--config", _config_segura(b.config, raiz), "--raiz", str(raiz)],
}


def _validar_run_id(run_id: str) -> str:
    if not _ID_OK.fullmatch(run_id) or run_id in (".", ".."):
        raise HTTPException(status_code=400, detail="run-id inválido")
    return run_id


def _config_segura(nome: str, raiz: Path) -> str:
    if not nome or "/" in nome or "\\" in nome or ".." in nome:
        raise HTTPException(status_code=400, detail="config deve ser um nome de arquivo em configs/")
    caminho = (raiz / "configs" / nome).resolve()
    if not str(caminho).startswith(str((raiz / "configs").resolve())) or not caminho.suffix == ".yaml":
        raise HTTPException(status_code=400, detail="config fora de configs/ ou extensão inválida")
    if not caminho.exists():
        raise HTTPException(status_code=404, detail="config não encontrada")
    return str(caminho)


class ExecucaoCorpo(BaseModel):
    acao: str
    config: str


def criar_app(raiz: Path | str = ".") -> FastAPI:
    raiz = Path(raiz)
    app = FastAPI(title="Lab-IA API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    execucoes: dict[str, subprocess.Popen] = {}

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

    @app.get("/configs")
    def listar_configs() -> list[str]:
        pasta = raiz / "configs"
        if not pasta.exists():
            return []
        return sorted(p.name for p in pasta.glob("*.yaml"))

    @app.get("/eventos")
    def eventos(desde: int = 0) -> list[dict]:
        caminho = raiz / ".lab-ia" / "eventos.jsonl"
        if not caminho.exists():
            return []
        linhas = caminho.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in linhas[desde:] if l.strip()]

    @app.get("/corre/{run_id}/tamanhos")
    def tamanhos(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "tamanhos.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run sem relatório de tamanhos")
        return json.loads(caminho.read_text(encoding="utf-8"))

    @app.get("/corre/{run_id}/comparativo")
    def comparativo(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "comparativo.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run sem comparativo de estratégias")
        return json.loads(caminho.read_text(encoding="utf-8"))

    @app.post("/execucao")
    def iniciar_execucao(corpo: ExecucaoCorpo) -> dict:
        if corpo.acao not in ACOES:
            raise HTTPException(status_code=400, detail=f"ação inválida (use {sorted(ACOES)})")
        args = [sys.executable, "-m", "labia.cli"] + ACOES[corpo.acao](corpo, raiz)
        log_dir = raiz / ".lab-ia" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chave = f"{corpo.acao}-{corpo.config}"
        log = open(log_dir / f"{chave}.log", "a", encoding="utf-8")
        processo = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(raiz))
        execucoes[chave] = processo
        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "execucao_iniciada", acao=corpo.acao, config=corpo.config, pid=processo.pid
        )
        return {"chave": chave, "pid": processo.pid, "comando": args}

    @app.get("/execucoes")
    def listar_execucoes() -> list[dict]:
        saida = []
        for chave, proc in execucoes.items():
            status = proc.poll()
            saida.append({"chave": chave, "pid": proc.pid, "vivo": status is None, "codigo_saida": status})
        return saida

    return app
