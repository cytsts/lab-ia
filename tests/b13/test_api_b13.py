"""Testes da API e dos contratos da especificação B13."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from labia.api.app import criar_app

RAIZ_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cliente():
    app = criar_app(raiz=RAIZ_REPO)
    return TestClient(app)


def test_obter_dataset_logica_pq(cliente):
    """CA1: Leitura de manifesto e metadados de logica-pq."""
    resp = cliente.get("/datasets/logica-pq")
    assert resp.status_code == 200
    dados = resp.json()
    assert dados["id"] == "logica-pq"
    assert "avaliacao" in dados["familias"]
    assert "treino" in dados["splits"]
    assert dados["tem_benchmark"] is True


def test_obter_dataset_inexistente_retorna_404(cliente):
    resp = cliente.get("/datasets/fantasma-nao-existe")
    assert resp.status_code == 404


def test_listar_itens_logica_pq(cliente):
    """CA1: Paginação e filtros por família e split no navegador de itens."""
    # Itens do split teste
    resp = cliente.get("/datasets/logica-pq/itens?split=teste&limite=5")
    assert resp.status_code == 200
    dados = resp.json()
    assert dados["total"] > 0
    assert dados["filtrados"] == 300  # split teste tem 300 itens
    assert len(dados["itens"]) == 5
    item = dados["itens"][0]
    assert "enunciado" in item
    assert "cot" in item
    assert "resposta" in item

    # Filtro por família
    resp_fam = cliente.get("/datasets/logica-pq/itens?familia=avaliacao&limite=10")
    assert resp_fam.status_code == 200
    dados_fam = resp_fam.json()
    assert all(i["familia"] == "avaliacao" for i in dados_fam["itens"])

    # Busca textual
    resp_busca = cliente.get("/datasets/logica-pq/itens?busca=satisfativel&limite=5")
    assert resp_busca.status_code == 200


def test_progresso_run_existente(cliente):
    """RF3.3 e RF3.4: Leitura leve de progresso e sparkline de run existente."""
    resp = cliente.get("/corre/logica-1/progresso")
    assert resp.status_code == 200
    dados = resp.json()
    assert dados["run_id"] == "logica-1"
    assert "passo" in dados
    assert "loss_val" in dados
    assert isinstance(dados["sparkline"], list)


def test_benchmarks_do_run(cliente):
    """RF6: Listagem dos relatórios salvos de benchmark."""
    resp = cliente.get("/corre/logica-1/benchmarks")
    assert resp.status_code == 200
    lista = resp.json()
    assert isinstance(lista, list)


def test_catalogo_modelos_e_gguf(tmp_path):
    """RF2.2, RF2.3 e CA13: Catálogo consolida runs e arquivos GGUF com honestidade."""
    # Cria pasta modelos com um gguf falso de teste
    app_tmp = criar_app(raiz=tmp_path)
    cliente_tmp = TestClient(app_tmp)
    pasta_modelos = tmp_path / "modelos"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    (pasta_modelos / "qwen2.5-1.5b-q4_k_m.gguf").write_bytes(b"GGUF_SIMULADO" * 1024)

    resp = cliente_tmp.get("/modelos")
    assert resp.status_code == 200
    dados = resp.json()
    assert "runs" in dados
    assert "gguf" in dados
    assert len(dados["gguf"]) == 1
    gguf = dados["gguf"][0]
    assert gguf["arquivo"] == "qwen2.5-1.5b-q4_k_m.gguf"
    assert gguf["quantizacao"] == "Q4_K_M"
    assert gguf["reasoning_declarado"] is True
    assert "declarado" in gguf["rotulo_honestidade"]


def test_testar_modelo_com_guloso_determinismo(cliente):
    """RF5 e CA6: Inferência gulosa no run logica-1 com resposta determinística."""
    corpo = {
        "run": "logica-1",
        "enunciado": "p = 1, q = 0. Calcule: (p E q).",
        "estrategia": "cot",
        "guloso": True,
        "temperatura": 0.0,
        "max_tokens": 128,
        "semente": 42,
    }
    resp1 = cliente.post("/testar", json=corpo)
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert "texto_gerado" in d1
    assert d1["resposta_extraida"] is not None

    resp2 = cliente.post("/testar", json=corpo)
    d2 = resp2.json()
    assert d1["texto_gerado"] == d2["texto_gerado"]
    assert d1["resposta_extraida"] == d2["resposta_extraida"]


def test_testar_com_prefixo_exploracao_cot(tmp_path):
    """RF7, CA15: Exploração do CoT via teacher forcing e isolamento em exploracao.jsonl."""
    app_tmp = criar_app(raiz=tmp_path)
    # Copiar run logica-1 para tmp_path/runs
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(RAIZ_REPO / "runs" / "logica-1", tmp_path / "runs" / "logica-1")

    cliente_tmp = TestClient(app_tmp)
    corpo = {
        "run": "logica-1",
        "enunciado": "p = 1, q = 0. Calcule: (p E q). Use 1 para verdadeiro e 0 para falso.",
        "estrategia": "cot",
        "guloso": True,
        "prefixo": "Passo 1: (p E q) = 0\n",
        "max_tokens": 60,
        "semente": 42,
    }
    resp = cliente_tmp.post("/testar", json=corpo)
    assert resp.status_code == 200
    dados = resp.json()
    assert dados["exploracao"] is True
    assert dados["texto_gerado"].startswith("Passo 1: (p E q) = 0\n")
    assert "Resposta: 0" in dados["texto_gerado"]
    assert dados["resposta_extraida"] == 0

    # Confere que gravou em .lab-ia/exploracao.jsonl
    exploracao_file = tmp_path / ".lab-ia" / "exploracao.jsonl"
    assert exploracao_file.exists()
    linhas = exploracao_file.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 1
    reg = json.loads(linhas[0])
    assert reg["prefixo"] == corpo["prefixo"]


def test_log_execucao_cauda_rapida(tmp_path):
    """CA12: Leitura rápida da cauda do log."""
    app_tmp = criar_app(raiz=tmp_path)
    cliente_tmp = TestClient(app_tmp)
    log_dir = tmp_path / ".lab-ia" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "teste-log.log"
    # Escreve 1000 linhas
    log_file.write_text("\n".join(f"linha {i}" for i in range(1000)), encoding="utf-8")

    resp = cliente_tmp.get("/execucoes/teste-log/log?linhas=50")
    assert resp.status_code == 200
    dados = resp.json()
    assert len(dados["linhas"]) == 50
    assert dados["linhas"][-1] == "linha 999"
    assert dados["linhas"][0] == "linha 950"
