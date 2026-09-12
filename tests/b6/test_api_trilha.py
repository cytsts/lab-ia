"""Testes da trilha na API (spec B6): as lições chegam à janela do laboratório."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from labia.api.app import criar_app

RAIZ = Path(__file__).resolve().parents[2]


def test_indice_das_licoes(tmp_path):
    cliente = TestClient(criar_app(RAIZ))
    dados = cliente.get("/trilha").json()
    assert dados["total"] == 10
    assert [l["numero"] for l in dados["licoes"]] == list(range(1, 11))
    assert dados["citacoes_quebradas"] == [], "lições citam código que não existe mais"
    assert all(l["titulo"] and l["resumo"] for l in dados["licoes"])
    assert all(l["tem_experimento"] for l in dados["licoes"])


def test_licao_em_markdown(tmp_path):
    cliente = TestClient(criar_app(RAIZ))
    resposta = cliente.get("/trilha/3")
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/markdown")
    assert resposta.text.startswith("# Lição 3")
    assert "No núcleo" in resposta.text and "Exercícios" in resposta.text


def test_licao_inexistente(tmp_path):
    cliente = TestClient(criar_app(RAIZ))
    assert cliente.get("/trilha/99").status_code == 404
    assert cliente.get("/trilha/0").status_code == 404


def test_rodar_experimento_da_licao(monkeypatch, tmp_path):
    from labia.api import app as mod

    chamado: dict = {}

    class FakePopen:
        def __init__(self, args, **kw):
            chamado["args"] = args
            self.pid = 777

        def poll(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    cliente = TestClient(criar_app(RAIZ))
    resposta = cliente.post("/trilha/1/rodar")
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["chave"] == "trilha-01"
    assert corpo["experimento"] == "e01_autograd.py"
    assert corpo["pid"] == 777
    assert chamado["args"][1].endswith("e01_autograd.py")
    # a execução aparece na listagem que a janela consulta
    assert "trilha-01" in [e["chave"] for e in cliente.get("/execucoes").json()]


def test_rodar_licao_inexistente(tmp_path):
    cliente = TestClient(criar_app(RAIZ))
    assert cliente.post("/trilha/99/rodar").status_code == 404
