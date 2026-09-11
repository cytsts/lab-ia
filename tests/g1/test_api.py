"""Testes da API interna (spec G1, RF7)."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from labia.api.app import criar_app
from labia.experiments.runner import registrar_metrica, salvar_estado


def _run_falso(raiz, run_id, registros):
    run = raiz / "runs" / run_id
    (run / "ckpt").mkdir(parents=True)
    for r in registros:
        registrar_metrica(run, r)
    salvar_estado(run, passo=20, passos_totais=20, concluido=True)
    return run


def test_listagem_e_metricas(tmp_path):
    _run_falso(tmp_path, "alpha", [{"passo": 10, "loss_val": 3.0}, {"passo": 20, "loss_val": 2.0}])
    cliente = TestClient(criar_app(tmp_path))

    lista = cliente.get("/corre").json()
    assert [r["run_id"] for r in lista] == ["alpha"]

    serie = cliente.get("/corre/alpha/metricas").json()
    assert [r["passo"] for r in serie] == [10, 20]
    assert serie[1]["loss_val"] == 2.0

    estado = cliente.get("/corre/alpha/estado").json()
    assert estado["concluido"] is True


def test_404_e_400(tmp_path):
    _run_falso(tmp_path, "beta", [{"passo": 1}])
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.get("/corre/inexistente/metricas").status_code == 404
    assert cliente.get("/corre/a%20b/metricas").status_code == 400
    assert cliente.get("/corre/..%2Fx/estado").status_code in (400, 404)


def test_eventos_append_only(tmp_path):
    pasta = tmp_path / ".lab-ia"
    pasta.mkdir()
    (pasta / "eventos.jsonl").write_text(
        json.dumps({"tipo": "a"}) + "\n" + json.dumps({"tipo": "b"}) + "\n", encoding="utf-8"
    )
    cliente = TestClient(criar_app(tmp_path))
    assert [e["tipo"] for e in cliente.get("/eventos").json()] == ["a", "b"]
    assert [e["tipo"] for e in cliente.get("/eventos?desde=1").json()] == ["b"]
