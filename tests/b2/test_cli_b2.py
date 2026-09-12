"""Testes da bancada B2 — CLI 'comparar' e 'curva' in-process (specs/B2.md)."""
from __future__ import annotations

import json

import pytest

from labia.cli import main


def escrever_run(tmp_path, run_id, valores, **estado):
    pasta = tmp_path / "runs" / run_id
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = [
        {"passo": i * 100, "loss_val": v, "loss_trem": v - 0.1, "lr": 3e-4, "tokens_por_s": 250_000.0}
        for i, v in enumerate(valores)
    ]
    (pasta / "metricas.jsonl").write_text(
        "\n".join(json.dumps(l) for l in linhas) + "\n", encoding="utf-8"
    )
    (pasta / "estado.json").write_text(
        json.dumps({"run_id": run_id, "passo": (len(valores) - 1) * 100,
                    "passos_totais": (len(valores) - 1) * 100, "concluido": True, **estado}),
        encoding="utf-8",
    )
    return pasta


def test_comparar_sem_runs_sai_com_erro(tmp_path, capsys):
    assert main(["comparar", "--raiz", str(tmp_path)]) == 2
    assert "nenhum run" in capsys.readouterr().err


def test_comparar_imprime_tabela_e_veredito(tmp_path, capsys):
    escrever_run(tmp_path, "bom", [8.0, 4.7, 4.75])
    escrever_run(tmp_path, "ruim", [8.0, 5.5, 5.6])
    assert main(["comparar", "--raiz", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "bom" in saida and "ruim" in saida
    assert "veredito:" in saida


def test_comparar_json(tmp_path, capsys):
    escrever_run(tmp_path, "a", [8.0, 4.7, 4.75])
    assert main(["comparar", "a", "--json", "--raiz", str(tmp_path)]) == 0
    dados = json.loads(capsys.readouterr().out)
    assert dados["ranking"] == ["a"]
    assert dados["runs"][0]["melhor_val"] == 4.7


def test_comparar_grava_relatorio(tmp_path, capsys):
    escrever_run(tmp_path, "a", [8.0, 4.7, 4.75])
    destino = tmp_path / "rel" / "comparacao.md"
    assert main(["comparar", "a", "--relatorio", str(destino), "--raiz", str(tmp_path)]) == 0
    assert "relatório" in capsys.readouterr().out
    assert destino.read_text(encoding="utf-8").startswith("# Comparação de runs")


def test_comparar_aceita_lista_explicita_de_runs(tmp_path, capsys):
    escrever_run(tmp_path, "a", [8.0, 4.7])
    escrever_run(tmp_path, "b", [8.0, 4.9])
    assert main(["comparar", "b", "--raiz", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["comparar", "b", "--json", "--raiz", str(tmp_path)]) == 0
    dados = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in dados["runs"]] == ["b"]


def test_curva_sem_runs_sai_com_erro(tmp_path, capsys):
    assert main(["curva", "--raiz", str(tmp_path)]) == 2
    assert "nenhum run" in capsys.readouterr().err


def test_curva_gera_arquivo_sem_matplotlib(tmp_path, capsys):
    escrever_run(tmp_path, "a", [8.0, 5.0, 4.8])
    saida = tmp_path / "figuras" / "curvas.svg"
    assert main(["curva", "a", "--saida", str(saida), "--raiz", str(tmp_path)]) == 0
    assert saida.exists()
    assert "curvas de a em" in capsys.readouterr().out


def test_curva_com_metrica_invalida_sai_com_erro(tmp_path, capsys):
    escrever_run(tmp_path, "a", [8.0, 5.0])
    assert main(["curva", "a", "--metricas", "nada", "--raiz", str(tmp_path)]) == 2
    assert "nenhum painel válido" in capsys.readouterr().err
