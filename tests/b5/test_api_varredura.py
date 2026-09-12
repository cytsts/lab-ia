"""Testes da varredura na API (spec B5): conferir a grade, disparar e ler o resultado."""
from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from labia.api.app import criar_app


@pytest.fixture()
def base_micro(tmp_path, corpus_arquivo):
    pasta = tmp_path / "configs"
    pasta.mkdir(exist_ok=True)
    cfg = {
        "nome": "base",
        "corpus": str(corpus_arquivo),
        "vocab_bpe": 256,
        "modelo": {"dim": 64, "camadas": 2, "cabecas": 2, "janela_ctx": 64, "abandono": 0.0},
        "passos": 8,
        "lote": 2,
        "stride": 0,
        "avaliar_a_cada": 4,
        "salvar_a_cada": 4,
        "iters_avaliacao": 2,
        "lr": 1e-3,
        "minimo_lr": 1e-4,
        "warmup": 2,
        "peso_decay": 0.0,
        "grad_clip": 1.0,
        "semente": 42,
        "dispositivo": "cpu",
        "arquivo_eventos": str(tmp_path / "eventos.jsonl"),
    }
    (pasta / "base.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return "base.yaml"


def test_seco_lista_combinacoes_sem_treinar(tmp_path, base_micro):
    cliente = TestClient(criar_app(tmp_path))
    resposta = cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["lr=0.0005,0.002"], "passos": 6})
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["prefixo"] == "varrer-base"
    assert len(corpo["variantes"]) == 2
    assert corpo["passos_por_variante"] == 6
    assert corpo["falhas"] == []
    assert "varredura : varrer-base" in corpo["tabela"]
    assert not (tmp_path / "runs").exists()


def test_seco_aponta_variante_impossivel(tmp_path, base_micro):
    cliente = TestClient(criar_app(tmp_path))
    corpo = cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["dim=64,65"]}).json()
    assert corpo["falhas"] == ["varrer-base-02"]
    assert "divisível" in corpo["variantes"][1]["erro"]


def test_seco_valida_entrada(tmp_path, base_micro):
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.post("/varrer/seco", json={"base": "naoexiste.yaml", "grades": ["lr=1,2"]}).status_code == 404
    assert cliente.post("/varrer/seco", json={"base": base_micro, "grades": []}).status_code == 422
    assert cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["turbo=1,2"]}).status_code == 400
    assert cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["lr=1"]}).status_code == 400
    assert cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["lr;rm -rf /"]}).status_code == 400
    assert cliente.post("/varrer/seco", json={"base": base_micro, "grades": ["lr=1,2"], "prefixo": "../x"}).status_code == 400


def test_varrer_dispara_processo(monkeypatch, tmp_path, base_micro):
    from labia.api import app as mod

    chamado: dict = {}

    class FakePopen:
        def __init__(self, args, **kw):
            chamado["args"] = args
            self.pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    cliente = TestClient(criar_app(tmp_path))
    resposta = cliente.post("/varrer", json={"base": base_micro, "grades": ["lr=0.0005,0.002"], "passos": 6})
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["chave"] == "varrer-varrer-base"
    assert corpo["pid"] == 4321
    args = chamado["args"]
    assert args[1:4] == ["-m", "labia.cli", "varrer"]
    assert "--base" in args and "--grade" in args and "lr=0.0005,0.002" in args
    assert "--passos" in args and "6" in args
    # a execução aparece na listagem que a janela usa
    assert [e["chave"] for e in cliente.get("/execucoes").json()] == ["varrer-varrer-base"]


def test_listar_e_ler_varredura(tmp_path):
    pasta = tmp_path / "runs" / "_varredura" / "sw-x"
    pasta.mkdir(parents=True)
    relatorio = {
        "versao": 1, "prefixo": "sw-x", "base": "configs/base.yaml", "modo": "grade", "semente": 42,
        "passos_por_variante": 10, "criado_em": "2026-09-11T00:00:00+00:00",
        "grades": {"lr": ["0.001", "0.002"]}, "valores_base": {"lr": 0.001}, "seco": False,
        "variantes": [
            {"indice": 1, "run_id": "sw-x-01", "sobrecritas": {"lr": 0.001}, "melhor_val": 5.0, "passo_melhor_val": 10, "diagnostico": "estavel"},
            {"indice": 2, "run_id": "sw-x-02", "sobrecritas": {"lr": 0.002}, "melhor_val": 4.5, "passo_melhor_val": 10, "diagnostico": "estavel"},
        ],
        "ranking": ["sw-x-02", "sw-x-01"],
        "melhor": {"run_id": "sw-x-02", "melhor_val": 4.5, "sobrecritas": {"lr": 0.002}},
        "efeito_por_chave": {"lr": {"base": 0.001, "por_valor": {"0.001": {"n": 1, "media_melhor_val": 5.0, "melhor_val": 5.0}, "0.002": {"n": 1, "media_melhor_val": 4.5, "melhor_val": 4.5}}, "melhor_valor": "0.002"}},
        "falhas": [],
    }
    (pasta / "varredura.json").write_text(json.dumps(relatorio), encoding="utf-8")
    cliente = TestClient(criar_app(tmp_path))

    listados = cliente.get("/varreduras").json()
    assert listados[0]["prefixo"] == "sw-x"
    assert listados[0]["melhor"]["run_id"] == "sw-x-02"

    lido = cliente.get("/varreduras/sw-x").json()
    assert lido["ranking"][0] == "sw-x-02"
    assert "varredura : sw-x" in lido["tabela"]

    markdown = cliente.get("/varreduras/sw-x?markdown=true")
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert markdown.text.startswith("# Varredura sw-x")

    assert cliente.get("/varreduras/nao-existe").status_code == 404
    assert cliente.get("/varreduras/..%2Fx").status_code in (400, 404)
