"""Testes de split de corpus e montagem determinística de dataset (spec G1, RF4/R3)."""
from __future__ import annotations

import torch

from labia.trainer.dados import dividir_corpus, lote_trem, montar_dataset
from labia.trainer.tokenizacao import treinar_tokenizer_ptbr
from conftest import gerar_corpus


def test_divisao_proporcao_e_determinismo():
    texto = gerar_corpus()
    t1, v1 = dividir_corpus(texto, semente=42)
    t2, v2 = dividir_corpus(texto, semente=42)
    assert (t1, v1) == (t2, v2)
    n_para = len([p for p in texto.split("\n\n") if p.strip()])
    esperados = max(1, int(n_para * 0.95))
    assert len([p for p in t1.split("\n\n") if p.strip()]) == esperados
    assert v1  # validação não vazia


def test_divisao_sementes_diferentes_divergem():
    texto = gerar_corpus()
    t_a, _ = dividir_corpus(texto, semente=1)
    t_b, _ = dividir_corpus(texto, semente=2)
    assert t_a != t_b


def test_montar_dataset_deslocamento_correto():
    class _TokFake:
        def encode(self, texto, add_special_tokens=False):
            ids = list(range(100))

            class _E:
                pass

            e = _E()
            e.ids = ids
            return e

    x, y = montar_dataset(_TokFake(), "x" * 10, janela=8)
    assert x.shape == (12, 8)  # (100-1)//8
    assert y.shape == x.shape
    # verificação direta: cada X[i, j+1] == Y[i, j] (alvo é o turno à esquerda)
    for i in range(3):
        assert torch.equal(x[i, 1:], y[i, :-1])
        assert int(y[i, 0]) == int(x[i, 0]) + 1


def test_stride_menor_dobra_aproveitamento():
    class _TokFake:
        def encode(self, texto, add_special_tokens=False):
            class _E:
                pass

            e = _E()
            e.ids = list(range(100))
            return e

    x_a, _ = montar_dataset(_TokFake(), "x" * 10, janela=8)          # stride = janela
    x_b, y_b = montar_dataset(_TokFake(), "x" * 10, janela=8, stride=4)  # deslizante
    assert x_b.shape[0] > x_a.shape[0]
    assert x_b.shape[1] == 8
    assert torch.equal(x_b[7, 1:], y_b[7, :-1])  # deslocamento interno preservado c/ stride


def test_lote_deterministico_por_passo_global():
    x = torch.arange(40).view(20, 2)
    y = torch.arange(40).view(20, 2) * 0
    a1 = lote_trem(x, y, passo=7, lote=4, semente=42, passos_por_epoch=5)
    a2 = lote_trem(x, y, passo=7, lote=4, semente=42, passos_por_epoch=5)
    assert torch.equal(a1[0], a2[0])
    b = lote_trem(x, y, passo=8, lote=4, semente=42, passos_por_epoch=5)
    assert not torch.equal(a1[0], b[0])


def test_tokenizer_treina_e_salva(tmp_path, corpus_arquivo):
    textos = [corpus_arquivo.read_text(encoding="utf-8")]
    tok = treinar_tokenizer_ptbr(textos, 200, tmp_path / "tokenizer.json")
    ids = tok.encode("casa de ferreiro, espeto de pau").ids
    assert len(ids) > 3
    texto_de_volta = tok.decode(ids)
    assert "casa" in texto_de_volta.replace("Ġ", " ").replace("Ċ", " ")
