"""Testes da bancada B1 — CLI 'dados' e 'novo' in-process (specs/B1.md)."""
from __future__ import annotations

import json

import pytest
import yaml

from labia.cli import main
from common import gerar_corpus


@pytest.fixture()
def fonte(tmp_path):
    arquivo = tmp_path / "meu-corpus.txt"
    arquivo.write_text(gerar_corpus(120, semente=11), encoding="utf-8")
    return arquivo


def test_dados_prepara_e_imprime_relatorio(fonte, tmp_path, capsys):
    assert main(["dados", "--de", str(fonte), "--id", "meu", "--raiz", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "dataset : meu" in saida
    assert "próximo : lab-ia novo" in saida
    assert (tmp_path / "data" / "meu" / "manifesto.json").exists()


def test_dados_json_imprime_manifesto(fonte, tmp_path, capsys):
    assert main(["dados", "--de", str(fonte), "--id", "meu", "--json", "--raiz", str(tmp_path)]) == 0
    manifesto = json.loads(capsys.readouterr().out)
    assert manifesto["id"] == "meu"
    assert manifesto["limpeza"]["paragrafos_finais"] > 0


def test_dados_listar_antes_e_depois(fonte, tmp_path, capsys):
    assert main(["dados", "--listar", "--raiz", str(tmp_path)]) == 0
    assert "nenhum dataset preparado" in capsys.readouterr().out
    main(["dados", "--de", str(fonte), "--id", "meu", "--raiz", str(tmp_path)])
    capsys.readouterr()
    assert main(["dados", "--listar", "--raiz", str(tmp_path)]) == 0
    assert "meu" in capsys.readouterr().out


def test_dados_sem_fonte_ou_id_aborta(tmp_path):
    with pytest.raises(SystemExit):
        main(["dados", "--id", "meu", "--raiz", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["dados", "--de", "x.txt", "--raiz", str(tmp_path)])


def test_dados_fonte_inexistente_sai_com_erro(tmp_path, capsys):
    assert main(["dados", "--de", str(tmp_path / "nada.txt"), "--id", "meu", "--raiz", str(tmp_path)]) == 2
    assert "não encontrada" in capsys.readouterr().err


def test_novo_listar_presets(capsys):
    assert main(["novo", "--listar-presets"]) == 0
    saida = capsys.readouterr().out
    for preset in ("micro", "rapido", "equilibrado", "longo", "moe"):
        assert preset in saida


def test_novo_gera_config_a_partir_do_dataset(fonte, tmp_path, capsys):
    main(["dados", "--de", str(fonte), "--id", "meu", "--raiz", str(tmp_path)])
    capsys.readouterr()
    assert main(["novo", "--nome", "run-a", "--dados", "meu", "--preset", "micro", "--raiz", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "próximo : lab-ia train --config configs" in saida
    arquivo = tmp_path / "configs" / "run-a.yaml"
    cfg = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    assert cfg["nome"] == "run-a"
    assert cfg["corpus_val"].endswith("val.txt")
    assert cfg["corpus"].endswith("trem.txt")
    # a config gerada tem de ser aceita pelo treino sem ajuste manual
    from labia.trainer.treino import ConfigTreino

    lida = ConfigTreino.de_arquivo(arquivo)
    assert lida.corpus_val == cfg["corpus_val"]
    assert isinstance(lida.minimo_lr, float)


def test_novo_recusa_sobrescrever_sem_forcar(fonte, tmp_path, capsys):
    main(["dados", "--de", str(fonte), "--id", "meu", "--raiz", str(tmp_path)])
    assert main(["novo", "--nome", "run-a", "--dados", "meu", "--raiz", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["novo", "--nome", "run-a", "--dados", "meu", "--raiz", str(tmp_path)]) == 2
    assert "já existe" in capsys.readouterr().err
    assert main(["novo", "--nome", "run-a", "--dados", "meu", "--forcar", "--raiz", str(tmp_path)]) == 0


def test_novo_dados_inexistentes_sai_com_erro(tmp_path, capsys):
    assert main(["novo", "--nome", "x", "--dados", "fantasma", "--raiz", str(tmp_path)]) == 2
    assert "não são um dataset preparado" in capsys.readouterr().err


def test_novo_sobrescrita_invalida_sai_com_erro(fonte, tmp_path, capsys):
    main(["dados", "--de", str(fonte), "--id", "meu", "--raiz", str(tmp_path)])
    assert main(["novo", "--nome", "x", "--dados", "meu", "--dim", "100", "--cabecas", "8", "--raiz", str(tmp_path)]) == 2
    assert "divisível" in capsys.readouterr().err
