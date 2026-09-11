"""Testes dos agentes (spec G9): skills expostas, execução, log de ações."""
from __future__ import annotations

import json

import pytest
import yaml

from common import config_micro, criar_dir_run
from labia.agents import REGISTRO, criar_agentes
from labia.experiments.runner import registrar_metrica


def test_tres_agentes_com_skills_e_assinaturas():
    agentes = criar_agentes(raiz=".")
    assert len(agentes) >= 3
    for nome, ag in agentes.items():
        skills = ag.expor_skills()
        assert len(skills) >= 2, nome
        for s in skills:
            assert s["assinatura"].startswith(s["nome"]), s
            assert ag.papel and ag.escopo


def test_treinador_treina_micro(tmp_path, corpus_arquivo):
    run = criar_dir_run(tmp_path / "runs" / "ag-micro")
    cfg = config_micro(corpus_arquivo, run, passos=60, avaliar_a_cada=30, salvar_a_cada=30)
    config_path = tmp_path / "micro.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    ag = REGISTRO["treinador"](raiz=tmp_path)
    resultado = ag.executar("train_model", config=str(config_path), run_id="ag-micro")
    assert resultado["run_id"] == "ag-micro"
    assert (tmp_path / "runs" / "ag-micro" / "metricas.jsonl").exists()
    eventos = (tmp_path / ".lab-ia" / "eventos.jsonl").read_text(encoding="utf-8")
    assert '"agente_acao"' in eventos and '"ok": true' in eventos


def test_skill_desconhecida_loga_falha(tmp_path):
    ag = REGISTRO["avaliador"](raiz=tmp_path)
    with pytest.raises(ValueError, match="não conhece"):
        ag.executar("nao_existe")
    eventos = (tmp_path / ".lab-ia" / "eventos.jsonl").read_text(encoding="utf-8")
    assert '"ok": false' in eventos


def _run_fake_metricas(tmp_path, nome, vals):
    run = tmp_path / "runs" / nome
    (run / "ckpt").mkdir(parents=True)
    for passo, v in vals:
        registrar_metrica(run, {"passo": passo, "loss_val": v})
    return run


def test_avaliador_compara_runs(tmp_path):
    _run_fake_metricas(tmp_path, "a", [(10, 4.0), (20, 3.5)])
    _run_fake_metricas(tmp_path, "b", [(10, 4.2), (20, 3.9)])
    ag = REGISTRO["avaliador"](raiz=tmp_path)
    r = ag.executar("compare_runs", run_a="a", run_b="b")
    assert r["vencedor"] == "a" and r["delta_min"] == pytest.approx(0.4)


def test_arquiteto_sugere_dentro_do_orcamento():
    ag = REGISTRO["arquiteto"](raiz=".")
    pequeno = ag.executar("suggest_architecture", corpus_bytes=900_000, vram_gb=4.0)
    grande = ag.executar("suggest_architecture", corpus_bytes=900_000, vram_gb=8.0)
    assert pequeno["parametros_estimados"] <= pequeno["orcamento_params"]
    assert grande["parametros_estimados"] <= grande["orcamento_params"]
    assert grande["parametros_estimados"] >= pequeno["parametros_estimados"]
    m = grande["modelo"]
    assert m["dim"] % m["cabecas"] == 0
    assert grande["passos"] >= 500


def test_arquiteto_diagnostica_moe(tmp_path):
    run = _run_fake_metricas(tmp_path, "m", [(10, 5.0), (20, 4.5)])
    registrar_metrica(run, {"passo": 20, "loss_val": 4.5, "aux_router": 1.9, "uso_especialistas": [0.85, 0.15]})
    ag = REGISTRO["arquiteto"](raiz=tmp_path)
    r = ag.executar("optimize_moe", run="m")
    assert any("coef_auxiliar" in s for s in r["sugestoes"]), r
    assert r["uso_final"] == [0.85, 0.15]
