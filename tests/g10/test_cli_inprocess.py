"""Testes in-process da CLI (cobertura G10 dos despachantes de `lab-ia`)."""
from __future__ import annotations

import json

import pytest

from labia.cli import main


def test_agentes_imprime_json_de_skills(capsys, tmp_path):
    assert main(["agentes", "--raiz", str(tmp_path)]) == 0
    saida = json.loads(capsys.readouterr().out)
    assert {"treinador", "avaliador", "arquiteto"} <= set(saida)
    assert saida["treinador"]["skills"][0]["assinatura"].startswith("train_model")


def test_agente_executa_skill_pela_cli(capsys, tmp_path):
    args = json.dumps({"corpus_bytes": 900_000, "vram_gb": 4.0})
    assert main(["agente", "arquiteto", "suggest_architecture", "--json", args, "--raiz", str(tmp_path)]) == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["parametros_estimados"] <= saida["orcamento_params"]


def test_agente_desconhecido_sai_com_erro():
    with pytest.raises(SystemExit) as excinfo:
        main(["agente", "fantasma", "qualquer", "--raiz", "."])
    assert excinfo.value.code != 0


def test_quantizar_pela_cli_sem_run_base_falha_limpo(tmp_path, capsys):
    destino = tmp_path / "runs" / "q"
    with pytest.raises(Exception):
        main(["quantizar", "--run", "inexistente", "--saida", destino.name, "--modo", "int8", "--raiz", str(tmp_path)])
