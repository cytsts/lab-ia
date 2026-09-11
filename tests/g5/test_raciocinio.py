"""Testes de raciocínio (spec G5 v2: CA4/CA5 + mecanismo das 3 estratégias, CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from labia.reasoning.estrategias import parse_resposta, responder_cot, responder_direta, responder_tot
from labia.reasoning.runner import comparar_estrategias, executar_benchmark
from labia.trainer.gerar import carregar_para_geracao


def test_parse_resposta():
    assert parse_resposta("blá blá\nResposta: 42") == 42
    assert parse_resposta("Resposta: -7") == -7
    assert parse_resposta("não tem número aqui") is None


def test_tres_estrategias_funcionam(ajuste_cot_micro):
    run = ajuste_cot_micro["run"]
    modelo, tok, disp = carregar_para_geracao(run, dispositivo="cpu")
    for fn, nome in ((responder_direta, "direta"), (responder_cot, "cot"), (responder_tot, "tot")):
        resposta, texto = fn(modelo, tok, disp, "Quanto e 3 + 4?", semente=7)
        assert resposta is None or isinstance(resposta, int), nome
        assert isinstance(texto, str), nome


def test_cot_aprende_formato(ajuste_cot_micro):
    """CA3(micro): o formato 'Resposta: N' deve aparecer em ≥80% das respostas CoT."""
    rel = executar_benchmark(
        ajuste_cot_micro["run"],
        "cot",
        benchmark=ajuste_cot_micro["benchmark"],
        limite=10,
        raiz=ajuste_cot_micro["run"].parents[1],
        dispositivo="cpu",
    )
    assert rel["taxa_resposta_valida"] >= 0.8, rel
    assert rel["itens"] == 10


def test_determinismo_byte_a_byte(ajuste_cot_micro):
    """CA4: mesma semente → relatório idêntico."""
    raiz = ajuste_cot_micro["run"].parents[1]
    executar_benchmark(
        ajuste_cot_micro["run"], "cot",
        benchmark=ajuste_cot_micro["benchmark"], limite=6, semente=1234, raiz=raiz, dispositivo="cpu",
    )
    arquivo = ajuste_cot_micro["run"] / "benchmark-cot-teste.json"
    b1 = arquivo.read_bytes()
    r2 = executar_benchmark(
        ajuste_cot_micro["run"], "cot",
        benchmark=ajuste_cot_micro["benchmark"], limite=6, semente=1234, raiz=raiz, dispositivo="cpu",
    )
    assert r2["itens"] == 6
    assert b1 == arquivo.read_bytes()


def test_comparativo_gera_arquivos(ajuste_cot_micro):
    raiz = ajuste_cot_micro["run"].parents[1]
    comp = comparar_estrategias(
        ajuste_cot_micro["run"],
        benchmark=ajuste_cot_micro["benchmark"],
        limite=4,
        raiz=raiz,
        dispositivo="cpu",
    )
    assert set(comp["estrategias"]) == {"direta", "cot", "tot"}
    assert (ajuste_cot_micro["run"] / "comparativo.json").exists()
    metricas = (ajuste_cot_micro["run"] / "metricas.jsonl").read_text(encoding="utf-8").splitlines()
    tipos = [json.loads(l).get("tipo") for l in metricas]
    assert tipos.count("benchmark") >= 3


def test_comparativo_real_valida_ca1_ca2_ca3():
    """Evidência viva: revalida CA1/CA2/CA3 v2 nos números reais gravados."""
    raiz = Path(__file__).resolve().parents[2]
    comp_caminho = raiz / "runs" / "g5-ajuste-cot" / "comparativo.json"
    if not comp_caminho.exists():
        pytest.skip("comparativo real ausente (rode lab-ia raciocinio --comparar)")

    comp = json.loads(comp_caminho.read_text(encoding="utf-8"))
    est = comp["estrategias"]
    assert comp["cot_menos_direta_pp"] >= 5.0, "CA1: CoT não superou direta em 5 p.p."
    assert comp["tot_menos_cot_pp"] >= -1.0, "CA2: ToT piorou demais vs CoT"
    assert est["cot"]["taxa_resposta_valida"] >= 0.90, "CA3: formato não internalizado"
    assert "soma" in est["cot"]["acuracia_por_familia"]
