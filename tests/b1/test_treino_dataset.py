"""Testes da bancada B1 — treino consumindo dataset preparado (corpus_val)."""
from __future__ import annotations

import json

from labia.bancada import dados as bd
from labia.trainer.treino import ConfigTreino, executar_treino
from common import config_micro, criar_dir_run, gerar_corpus


def test_treino_usa_validacao_do_dataset_preparado(tmp_path):
    fonte = tmp_path / "fonte.txt"
    fonte.write_text(gerar_corpus(120, semente=5), encoding="utf-8")
    bd.preparar(bd.ConfigPreparo(fontes=[str(fonte)], id="ds", destino="data", raiz=str(tmp_path)))

    run = criar_dir_run(tmp_path / "runs" / "com-val")
    cfg = ConfigTreino(
        **config_micro(
            tmp_path / "data" / "ds" / "trem.txt",
            run,
            corpus_val=str(tmp_path / "data" / "ds" / "val.txt"),
            passos=6,
            avaliar_a_cada=3,
            salvar_a_cada=3,
            iters_avaliacao=2,
            dispositivo="cpu",
        )
    )
    executar_treino(cfg, run, raiz=tmp_path)

    estado = json.loads((run / "estado.json").read_text(encoding="utf-8"))
    assert estado["concluido"] is True
    assert "arquivos separados" in estado["origem_dados"]
    registros = [json.loads(l) for l in (run / "metricas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert registros[0]["passo"] == 0
    assert registros[-1]["passo"] == 6
    assert all(r["loss_val"] is not None for r in registros)


def test_treino_sem_corpus_val_registra_o_split(tmp_path):
    fonte = tmp_path / "fonte.txt"
    fonte.write_text(gerar_corpus(80, semente=6), encoding="utf-8")
    run = criar_dir_run(tmp_path / "runs" / "sem-val")
    cfg = ConfigTreino(
        **config_micro(fonte, run, passos=4, avaliar_a_cada=2, salvar_a_cada=2, iters_avaliacao=2, dispositivo="cpu")
    )
    executar_treino(cfg, run, raiz=tmp_path)
    estado = json.loads((run / "estado.json").read_text(encoding="utf-8"))
    assert "split" in estado["origem_dados"]


def test_config_com_expoente_em_texto_falha_na_leitura(tmp_path):
    """YAML 1.1 não reconhece '3e-05' como número: o erro tem de apontar o conserto."""
    arquivo = tmp_path / "quebrada.yaml"
    arquivo.write_text("nome: x\ncorpus: c.txt\nminimo_lr: 3e-05\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="3.0e-05"):
        ConfigTreino.de_arquivo(arquivo)
