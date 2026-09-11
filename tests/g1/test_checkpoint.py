"""Testes de checkpoint atômico e estado (spec G1, RF2/RF3; base da G8)."""
from __future__ import annotations

import json

import torch

from labia.experiments.runner import (
    carregar_estado,
    dir_run,
    ler_metricas,
    registrar_metrica,
    salvar_checkpoint,
    salvar_estado,
    ultimo_checkpoint,
)


def test_run_id_invalido_rejeitado(tmp_path):
    for ruim in ("", "../x", "a b", "a/b", "."):
        try:
            dir_run(tmp_path, ruim)
        except ValueError:
            continue
        raise AssertionError(f"deveria rejeitar run-id {ruim!r}")


def test_checkpoint_atomico_e_ultimo(tmp_path):
    run = tmp_path / "r1"
    (run / "ckpt").mkdir(parents=True)
    p5 = salvar_checkpoint(run, 5, {"passo": 5, "peso": torch.ones(2)})
    p10 = salvar_checkpoint(run, 10, {"passo": 10, "peso": torch.ones(2)})
    assert p5.exists() and p10.exists()
    assert not list((run / "ckpt").glob("*.tmp")), "sobraram arquivos .tmp"
    assert ultimo_checkpoint(run) == p10
    ck = torch.load(p10, weights_only=True)
    assert int(ck["passo"]) == 10


def test_estado_merge_e_jsonl_metricas(tmp_path):
    run = tmp_path / "r2"
    run.mkdir()
    salvar_estado(run, passo=10, concluido=False)
    estado = salvar_estado(run, passo=20, concluido=True)
    assert estado["passo"] == 20 and estado["concluido"] is True
    assert carregar_estado(run) == json.loads((run / "estado.json").read_text(encoding="utf-8"))
    registrar_metrica(run, {"passo": 10, "loss_val": 3.2})
    registrar_metrica(run, {"passo": 20, "loss_val": 2.1})
    registros = ler_metricas(run)
    assert [r["passo"] for r in registros] == [10, 20]  # append-only, ordem preservada
