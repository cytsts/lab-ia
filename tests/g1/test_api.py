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


# ---------- spec G6 RF5: /execucao, /execucoes, /tamanhos, /comparativo ----------


def test_execucao_validacoes(tmp_path):
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.post("/execucao", json={"acao": "nanar", "config": "x.yaml"}).status_code == 400
    assert cliente.post("/execucao", json={"acao": "train", "config": "../passwd"}).status_code == 400
    assert cliente.post("/execucao", json={"acao": "train", "config": "naoexiste.yaml"}).status_code == 404


def test_execucao_inicia_subprocesso(monkeypatch, tmp_path):
    from labia.api import app as mod

    chamado: dict = {}

    class FakePopen:
        def __init__(self, args, **kw):
            chamado["args"] = args
            self.pid = 999

        def poll(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "micro.yaml").write_text("nome: x\n", encoding="utf-8")
    cliente = TestClient(mod.criar_app(tmp_path))

    r = cliente.post("/execucao", json={"acao": "train", "config": "micro.yaml"})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["pid"] == 999 and "labia.cli" in chamado["args"]
    assert any("micro.yaml" in a for a in chamado["args"])
    lista = cliente.get("/execucoes").json()
    assert lista[0]["vivo"] is True
    eventos = cliente.get("/eventos").json()
    assert any(e["tipo"] == "execucao_iniciada" for e in eventos)


def test_tamanhos_e_comparativo(tmp_path):
    run = tmp_path / "runs" / "q9"
    run.mkdir(parents=True)
    (run / "tamanhos.json").write_text('{"fator_alvos": 3.9}', encoding="utf-8")
    (run / "comparativo.json").write_text('{"cot_menos_direta_pp": 12.0}', encoding="utf-8")
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.get("/corre/q9/tamanhos").json()["fator_alvos"] == 3.9
    assert cliente.get("/corre/q9/comparativo").json()["cot_menos_direta_pp"] == 12.0
    assert cliente.get("/corre/q9/tamanhos").status_code == 200
    (run / "tamanhos.json").unlink()
    assert cliente.get("/corre/q9/tamanhos").status_code == 404
