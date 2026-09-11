"""Teste do ponto de entrada CLI (spec G1: train/gerar headless)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from conftest import config_micro, criar_dir_run

RAIZ_REPO = Path(__file__).resolve().parents[2]


def test_cli_train_e_gerar(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "cli-run")
    cfg = config_micro(corpus_arquivo, run, passos=300, avaliar_a_cada=150, salvar_a_cada=150, lr=3e-3)
    config_path = tmp_path / "micro.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    feito = subprocess.run(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(config_path),
         "--run-id", "cli-run", "--raiz", str(tmp_path)],
        cwd=RAIZ_REPO, capture_output=True, text=True, timeout=600,
    )
    assert feito.returncode == 0, feito.stdout + feito.stderr
    assert (tmp_path / "runs" / "cli-run" / "metricas.jsonl").exists()

    feito = subprocess.run(
        [sys.executable, "-m", "labia.cli", "gerar", "--run", "cli-run",
         "--prompt", "A noite", "--passos-max", "12", "--guloso", "--raiz", str(tmp_path)],
        cwd=RAIZ_REPO, capture_output=True, text=True, timeout=300,
    )
    assert feito.returncode == 0, feito.stdout + feito.stderr
    assert feito.stdout.strip()
