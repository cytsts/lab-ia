"""Registro de experimentos: layout de runs/, checkpoints atômicos e métricas append-only."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

from ..utils.estado import agora_iso, salvar_json_atomico

_NOME_CKPT = re.compile(r"^passo-(\d+)\.pt$")


def dir_run(raiz: Path | str, run_id: str) -> Path:
    if not run_id or run_id in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(f"run-id inválido: {run_id!r} (use letras, dígitos, ponto, hífen, _)")
    d = Path(raiz) / "runs" / run_id
    for sub in ("ckpt", "tokens"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def salvar_estado(run_dir: Path, **campos) -> dict:
    caminho = Path(run_dir) / "estado.json"
    estado = {}
    if caminho.exists():
        estado = json.loads(caminho.read_text(encoding="utf-8"))
    estado.update(campos)
    estado["atualizado_em"] = agora_iso()
    salvar_json_atomico(caminho, estado)
    return estado


def carregar_estado(run_dir: Path) -> dict | None:
    caminho = Path(run_dir) / "estado.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_checkpoint(run_dir: Path, passo: int, dados: dict) -> Path:
    caminho = Path(run_dir) / "ckpt" / f"passo-{passo}.pt"
    tmp = caminho.with_name(caminho.name + ".tmp")
    torch.save(dados, tmp)
    os.replace(tmp, caminho)
    salvar_json_atomico(Path(run_dir) / "ckpt" / "ultimo.json", {"passo": passo, "arquivo": caminho.name})
    return caminho


def ultimo_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = Path(run_dir) / "ckpt"
    passos = [int(m.group(1)) for p in ckpt_dir.glob("passo-*.pt") if (m := _NOME_CKPT.match(p.name))]
    if not passos:
        return None
    return ckpt_dir / f"passo-{max(passos)}.pt"


def registrar_metrica(run_dir: Path, registro: dict) -> None:
    registro.setdefault("tempo", agora_iso())
    with open(Path(run_dir) / "metricas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def ler_metricas(run_dir: Path) -> list[dict]:
    caminho = Path(run_dir) / "metricas.jsonl"
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]
