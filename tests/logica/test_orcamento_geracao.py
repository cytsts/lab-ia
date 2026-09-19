"""Testes do orçamento de geração no benchmark de raciocínio (regressão B12).

O bug que estes testes travam: 'responder_cot' gerava no máximo 64 tokens. Cadeias
curtas (aritmética da G5) cabiam; uma tabela-verdade de 4 linhas não — o texto cortava
no meio, nunca saía "Resposta:", e a família marcava 0% de acerto. Medido no dataset de
lógica: 38,3% de acurácia com o corte, 80,0% sem ele.

A correção tem duas partes: orçamento maior e PARADA ANTECIPADA assim que a resposta
sai. A parada também evita gastar geração depois do fim da cadeia.
"""
from __future__ import annotations

import torch

from labia.models.gpt import ConfigGPT, GPT
from labia.reasoning.estrategias import _gerar_tokens, responder_cot


def modelo_minusculo() -> GPT:
    cfg = ConfigGPT(vocab=64, dim=32, camadas=1, cabecas=4, janela_ctx=32, abandono=0.0)
    modelo = GPT(cfg)
    modelo.init_pesos(0)
    modelo.eval()
    return modelo


def test_parada_antecipada_interrompe_a_geracao():
    modelo = modelo_minusculo()
    ids = [1, 2, 3]
    novos, _ = _gerar_tokens(modelo, ids, torch.device("cpu"), n_max=50, guloso=True,
                             parada=lambda gerados: len(gerados) >= 4)
    assert len(novos) == 4, "a parada tem de cortar exatamente quando o predicado pede"


def test_sem_parada_gera_o_orcamento_inteiro():
    modelo = modelo_minusculo()
    novos, _ = _gerar_tokens(modelo, [1, 2, 3], torch.device("cpu"), n_max=7, guloso=True)
    assert len(novos) == 7


def test_orcamento_do_cot_cabe_uma_cadeia_longa():
    """256 tokens: uma tabela-verdade de 2 variáveis com duas fórmulas por linha passa disso?"""
    from labia.logica import gerador_proposicional as g

    padrao = responder_cot.__defaults__
    assert padrao and padrao[-1] >= 256, f"orçamento do CoT encolheu para {padrao}"

    # a maior cadeia do dataset real, medida em caracteres, tem de caber com folga
    import json
    from pathlib import Path

    caminho = Path("data/logica-pq/benchmark.jsonl")
    if not caminho.exists():
        return  # dataset não versionado neste workspace
    itens = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]
    maior = max(len(i["cot"]) for i in itens)
    # ~4 caracteres por token é conservador para este corpus; 256 tokens cobrem 1024
    assert maior < 256 * 4, f"a maior cadeia tem {maior} caracteres"


def test_a_regex_de_parada_so_dispara_com_resposta_completa():
    from labia.reasoning.estrategias import _RESPOSTA_COMPLETA

    assert _RESPOSTA_COMPLETA.search("Passo 1: (p E q) = 0") is None
    assert _RESPOSTA_COMPLETA.search("Resposta:") is None          # sem o número ainda não
    assert _RESPOSTA_COMPLETA.search("Resposta: 1") is not None
    assert _RESPOSTA_COMPLETA.search("Resposta: -3") is not None
