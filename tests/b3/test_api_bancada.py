"""Testes da bancada na API (spec B3): dados, config, comparação, curvas e guia.

É o contrato que a janela Electron consome — se algum destes quebrar, o laboratório
portátil perde a capacidade de operar o ciclo sem terminal.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from labia.api.app import criar_app
from labia.experiments.runner import registrar_metrica, salvar_estado
from common import gerar_corpus


def _run_falso(raiz, run_id, valores, **estado):
    run = raiz / "runs" / run_id
    (run / "ckpt").mkdir(parents=True)
    for indice, valor in enumerate(valores):
        registrar_metrica(run, {"passo": indice * 100, "loss_val": valor, "loss_trem": valor - 0.1, "tokens_por_s": 250000})
    salvar_estado(run, passo=(len(valores) - 1) * 100, passos_totais=(len(valores) - 1) * 100, concluido=True, **estado)
    return run


@pytest.fixture()
def fonte(tmp_path):
    arquivo = tmp_path / "meu-corpus.txt"
    arquivo.write_text(gerar_corpus(120, semente=13), encoding="utf-8")
    return arquivo


def test_saude_responde_estado_do_nucleo(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "bancada.md").write_text("# guia", encoding="utf-8")
    dados = TestClient(criar_app(tmp_path)).get("/saude").json()
    assert dados["ok"] is True
    assert dados["raiz"].endswith(str(tmp_path.name))
    assert isinstance(dados["cuda"], bool)
    assert dados["guia"] == ["bancada.md"]
    assert {"runs", "datasets", "configs"} <= set(dados)


def test_presets_listados(tmp_path):
    presets = TestClient(criar_app(tmp_path)).get("/presets").json()
    assert {p["nome"] for p in presets} == {"micro", "rapido", "equilibrado", "longo", "moe"}
    assert all(p["descricao"] and p["modelo"] for p in presets)


def test_dados_prepara_e_lista(tmp_path, fonte):
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.get("/datasets").json() == []

    resposta = cliente.post("/dados", json={"fontes": [str(fonte)], "id": "meu"})
    assert resposta.status_code == 200, resposta.text
    manifesto = resposta.json()
    assert manifesto["id"] == "meu"
    assert manifesto["limpeza"]["paragrafos_finais"] > 0

    listados = cliente.get("/datasets").json()
    assert [d["id"] for d in listados] == ["meu"]
    assert listados[0]["tokens_aprox_trem"] > 0


def test_dados_valida_entrada(tmp_path, fonte):
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.post("/dados", json={"fontes": [], "id": "x"}).status_code == 422
    assert cliente.post("/dados", json={"fontes": [str(fonte)], "id": "../x"}).status_code == 400
    assert cliente.post("/dados", json={"fontes": [str(tmp_path / "nao.txt")], "id": "x"}).status_code == 404
    assert cliente.post("/dados", json={"fontes": [str(fonte)], "id": "x", "frac_val": 0.9}).status_code == 400


def test_novo_gera_config_consumivel_pelo_treino(tmp_path, fonte):
    cliente = TestClient(criar_app(tmp_path))
    cliente.post("/dados", json={"fontes": [str(fonte)], "id": "meu"})

    resposta = cliente.post("/novo", json={"nome": "run-api", "dados": "meu", "preset": "micro"})
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["arquivo_config"] == "run-api.yaml"
    assert "próximo : lab-ia train" in corpo["resumo"]
    assert corpo["parametros"]["total"] > 0

    # a config gravada é aceita pelo núcleo sem ajuste manual
    from labia.trainer.treino import ConfigTreino

    lida = ConfigTreino.de_arquivo(tmp_path / "configs" / "run-api.yaml")
    assert lida.corpus_val and lida.corpus_val.endswith("val.txt")

    # e aparece na lista que a tela de Executar usa
    assert "run-api.yaml" in cliente.get("/configs").json()


def test_novo_erros(tmp_path, fonte):
    cliente = TestClient(criar_app(tmp_path))
    cliente.post("/dados", json={"fontes": [str(fonte)], "id": "meu"})
    assert cliente.post("/novo", json={"nome": "x", "dados": "fantasma"}).status_code == 404
    assert cliente.post("/novo", json={"nome": "x", "dados": "meu", "preset": "turbo"}).status_code == 400
    assert (
        cliente.post("/novo", json={"nome": "x", "dados": "meu", "sobrescritas": {"dim": 100, "cabecas": 8}}).status_code
        == 400
    )
    cliente.post("/novo", json={"nome": "x", "dados": "meu"})
    assert cliente.post("/novo", json={"nome": "x", "dados": "meu"}).status_code == 409


def test_comparar_usa_todos_os_runs_por_padrao(tmp_path):
    _run_falso(tmp_path, "bom", [8.0, 4.7, 4.75])
    _run_falso(tmp_path, "ruim", [8.0, 5.5, 5.6])
    cliente = TestClient(criar_app(tmp_path))
    dados = cliente.get("/comparar").json()
    assert dados["melhor"] == "bom"
    assert [r["run_id"] for r in dados["runs"]] == ["bom", "ruim"]
    assert "bom generaliza melhor" in dados["veredito"]


def test_comparar_aceita_lista_e_valida(tmp_path):
    _run_falso(tmp_path, "bom", [8.0, 4.7])
    _run_falso(tmp_path, "ruim", [8.0, 5.5])
    cliente = TestClient(criar_app(tmp_path))
    assert [r["run_id"] for r in cliente.get("/comparar?runs=ruim").json()["runs"]] == ["ruim"]
    assert cliente.get("/comparar?runs=,").status_code == 400
    assert TestClient(criar_app(tmp_path)).get("/comparar?runs=fantasma").json()["runs"][0]["diagnostico"] == "ausente"


def test_comparar_sem_runs_devolve_404(tmp_path):
    assert TestClient(criar_app(tmp_path)).get("/comparar").status_code == 404


def test_relatorio_em_markdown(tmp_path):
    _run_falso(tmp_path, "bom", [8.0, 4.7, 4.75])
    resposta = TestClient(criar_app(tmp_path)).get("/comparar/relatorio?runs=bom")
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/markdown")
    assert resposta.text.startswith("# Comparação de runs")


def test_curvas_svg_sem_dependencia_grafica(tmp_path):
    _run_falso(tmp_path, "a", [8.0, 5.0, 4.8])
    _run_falso(tmp_path, "b", [8.0, 5.6, 5.4])
    resposta = TestClient(criar_app(tmp_path)).get("/comparar/curvas.svg?runs=a,b&metricas=loss_val")
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("image/svg+xml")
    raiz = ET.fromstring(resposta.text)  # XML válido, não string solta
    assert len(raiz.findall(".//{http://www.w3.org/2000/svg}polyline")) == 2
    assert "a" in resposta.text and "b" in resposta.text


def test_curvas_svg_erro_quando_nao_ha_metrica(tmp_path):
    _run_falso(tmp_path, "a", [8.0, 5.0])
    resposta = TestClient(criar_app(tmp_path)).get("/comparar/curvas.svg?runs=a&metricas=inexistente")
    assert resposta.status_code == 404


def test_guia_serve_o_material_do_lab(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "bancada.md").write_text("# Bancada\n\ntexto do guia", encoding="utf-8")
    cliente = TestClient(criar_app(tmp_path))
    resposta = cliente.get("/guia/bancada")
    assert resposta.status_code == 200
    assert "texto do guia" in resposta.text
    assert cliente.get("/guia/bancada.md").status_code == 200
    assert cliente.get("/guia/inexistente").status_code == 404
    assert cliente.get("/guia/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)


def test_log_da_execucao_para_a_janela(tmp_path):
    cliente = TestClient(criar_app(tmp_path))
    assert cliente.get("/execucoes/train-x.yaml/log").json()["existe"] is False
    pasta = tmp_path / ".lab-ia" / "logs"
    pasta.mkdir(parents=True)
    (pasta / "train-x.yaml.log").write_text("linha 1\nlinha 2\nlinha 3\n", encoding="utf-8")
    dados = cliente.get("/execucoes/train-x.yaml/log?linhas=2").json()
    assert dados["existe"] is True and dados["total"] == 3
    assert dados["linhas"] == ["linha 2", "linha 3"]
    # a rota pode recusar antes do handler (404) ou o handler recusar (400); o que
    # importa é nunca servir arquivo fora de .lab-ia/logs
    assert cliente.get("/execucoes/..%2Fsegredo/log").status_code in (400, 404)
