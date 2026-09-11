"""Testes de quantização de runs (spec G3: CA1, CA2, CA3, CA4, CA5)."""
from __future__ import annotations

import json

import torch

from common import criar_dir_run
from labia.experiments.runner import ultimo_checkpoint
from labia.models.gpt import ConfigGPT, GPT
from labia.trainer.gerar import gerar_de_checkpoint
from labia.trainer.quantiza import _perda_em_janelas, _trocar_lineares, quantizar_run
from labia.trainer.dados import montar_dataset
from labia.trainer.tokenizacao import carregar_tokenizer


def test_int8_fator_perda_e_arquivos(tmp_path, base_g3):
    base, tarefa = base_g3
    destino = criar_dir_run(tmp_path / "q8")
    tam = quantizar_run(base, destino, "int8", tarefa, raiz=tmp_path, arquivo_eventos=str(tmp_path / "ev.jsonl"))
    assert tam["fator_alvos"] >= 1.9, tam
    assert tam["bits_efetivos_por_parametro"] <= 8.6, tam
    assert tam["perda_val_depois"] <= tam["perda_val_antes"] + 0.05, "CA1: int8 degradou a perda"
    assert (destino / "tamanhos.json").exists() and (destino / "quant_meta.json").exists()
    linhas = (destino / "metricas.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 2
    assert json.loads(linhas[0])["origem"] == "base-fp32"
    assert json.loads(linhas[1])["origem"] == "quantizado-int8"
    assert all(0.0 <= e < 0.05 for e in tam["erro_relativo_por_modulo"].values())  # CA3
    assert len(tam["erro_relativo_por_modulo"]) >= 8


def test_nf4_fator_perda(tmp_path, base_g3):
    base, tarefa = base_g3
    destino = criar_dir_run(tmp_path / "q4")
    tam = quantizar_run(base, destino, "nf4", tarefa, raiz=tmp_path, arquivo_eventos=str(tmp_path / "ev.jsonl"))
    assert tam["fator_alvos"] >= 3.2, tam
    assert tam["bits_efetivos_por_parametro"] <= 4.6, tam
    assert tam["perda_val_depois"] <= tam["perda_val_antes"] * 1.10, "CA2: nf4 degradou demais"


def test_mutacao_codigos_zerados_degrada_perda(tmp_path, base_g3):
    """CA4 (prova de que o forward lê o formato quantizado): zerar os códigos int8 degrada ≥0.5 nats.

    (Inflar só a escala ×4 move a perda apenas ~0.15 nats — o pré-LN absorve escala;
    zerar os códigos elimina o peso dos blocos e não tem como ser ignorado.)"""
    base, tarefa = base_g3
    destino = criar_dir_run(tmp_path / "qm")
    tam = quantizar_run(base, destino, "int8", tarefa, raiz=tmp_path, arquivo_eventos=str(tmp_path / "ev.jsonl"))

    ck = torch.load(ultimo_checkpoint(destino), map_location="cpu", weights_only=True)
    modelo = GPT(ConfigGPT.de_dict(ck["config_modelo"]))
    _trocar_lineares(modelo, "int8")
    estado = {k: (torch.zeros_like(t) if k.endswith("q_peso") else t) for k, t in ck["modelo"].items()}
    modelo.load_state_dict(estado, strict=True)

    tok = carregar_tokenizer(base / "tokens")  # run quantizado herdava o tokenizer da base
    x, y = montar_dataset(tok, tarefa.read_text(encoding="utf-8"), modelo.cfg.janela_ctx)
    x, y = x[:12], y[:12]
    perda_adulterada = _perda_em_janelas(modelo, x, y, torch.device("cpu"))
    assert perda_adulterada >= tam["perda_val_antes"] + 0.5, f"mutação passou despercebida: {perda_adulterada} vs {tam['perda_val_antes']}"
    # escala também é lida de verdade: ×4 muda os logits (mesmo que o pré-LN absorva parte)
    estado2 = {k: (t * 4.0 if k.endswith("q_escala") else t) for k, t in ck["modelo"].items()}
    modelo2 = GPT(ConfigGPT.de_dict(ck["config_modelo"]))
    _trocar_lineares(modelo2, "int8")
    modelo2.load_state_dict(estado2, strict=True)
    idx = x[0:1]
    with torch.no_grad():
        l1 = modelo(idx)[0]
        l2 = modelo2(idx)[0]
    assert not torch.equal(l1, l2)


def test_gerar_funciona_no_run_quantizado(tmp_path, base_g3):
    base, tarefa = base_g3
    destino = criar_dir_run(tmp_path / "qg")
    quantizar_run(base, destino, "int8", tarefa, raiz=tmp_path, arquivo_eventos=str(tmp_path / "ev.jsonl"))
    texto = gerar_de_checkpoint(destino, "A amostra", passos_max=12, guloso=True)
    assert isinstance(texto, str) and len(texto) > 0
