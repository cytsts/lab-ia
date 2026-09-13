"""Testes do gerador de lógica proposicional (B12).

A pergunta que estes testes respondem: **o material ensina a verdade?** Cada item tem
uma resposta calculada e um texto de passos; se os dois discordarem, o modelo aprende
errado. Por isso a invariante principal (CoT x resposta) é verificada em todos os itens
gerados, e as classificações são conferidas contra tabela-verdade montada à mão.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from labia.logica import gerador_proposicional as g
from labia.reasoning.estrategias import carregar_benchmark, parse_resposta


def cfg(tmp_path, **extra) -> g.ConfigLogica:
    base = dict(id="teste-logica", n_treino=60, n_teste=15, n_dificil=10, operadores=2, operadores_dificeis=4,
                destino="data", raiz=str(tmp_path))
    base.update(extra)
    return g.ConfigLogica(**base)


# --------------------------------------------------- a semântica dos operadores

@pytest.mark.parametrize(
    "operador,esperado",
    [
        ("e", {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
        ("ou", {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
        ("imp", {(0, 0): 1, (0, 1): 1, (1, 0): 0, (1, 1): 1}),
        ("bi", {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
        ("xor", {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}),
    ],
)
def test_tabela_verdade_de_cada_operador(operador, esperado):
    """A tabela-verdade escrita à mão é a referência; o avaliador tem de bater com ela."""
    formula = (operador, ("var", "p"), ("var", "q"))
    for (a, b), valor in esperado.items():
        assert int(g.avaliar(formula, {"p": bool(a), "q": bool(b)})) == valor, f"{operador} com p={a}, q={b}"


def test_negacao_e_render():
    formula = ("nao", ("var", "p"))
    assert g.avaliar(formula, {"p": True}) is False
    assert g.render(formula) == "(NAO p)"
    assert g.render(("imp", ("var", "p"), ("var", "q"))) == "(p => q)"
    # parênteses em toda fórmula composta: ambiguidade é ruído para o modelo
    assert g.render(("e", ("ou", ("var", "p"), ("var", "q")), ("var", "p"))) == "((p OU q) E p)"


def test_leis_de_de_morgan():
    """(NAO (p E q)) equivale a ((NAO p) OU (NAO q)) — conferido, não assumido."""
    esquerda = ("nao", ("e", ("var", "p"), ("var", "q")))
    direita = ("ou", ("nao", ("var", "p")), ("nao", ("var", "q")))
    assert g.equivalentes(esquerda, direita, ["p", "q"]) == (True, None)
    # e a distributividade falha de propósito, para o teste não ser vazio
    assert g.equivalentes(("ou", ("var", "p"), ("var", "q")), ("e", ("var", "p"), ("var", "q")), ["p", "q"])[0] is False


def test_implicacao_so_vale_numa_direcao():
    """p E q obriga p; p NÃO obriga q. O contraexemplo tem de ser a linha p=1, q=0."""
    vale, _ = g.consequencia_logica([("e", ("var", "p"), ("var", "q"))], ("var", "p"), ["p", "q"])
    assert vale is True
    vale, contra = g.consequencia_logica([("var", "p")], ("var", "q"), ["p", "q"])
    assert vale is False and contra == {"p": True, "q": False}


def test_tautologia_e_insatisfativel():
    assert g.classificar(("ou", ("var", "p"), ("nao", ("var", "p"))), ["p"])["falsas"] == []
    assert g.classificar(("e", ("var", "p"), ("nao", ("var", "p"))), ["p"])["verdadeiras"] == []


# --------------------------------------------------- a invariante do material

def test_o_cot_nunca_discorda_da_resposta(tmp_path):
    """Se o texto ensina um número e a resposta é outro, o dataset está mentindo."""
    manifesto = g.gerar(cfg(tmp_path))
    itens = [json.loads(l) for l in (Path(manifesto["pasta"]) / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(itens) == 85
    mentiras = [i["id"] for i in itens if parse_resposta(i["cot"]) != i["resposta"]]
    assert mentiras == [], f"o CoT discorda da resposta em: {mentiras}"
    assert {i["resposta"] for i in itens} <= {0, 1}, "a resposta é sempre 0 ou 1 (o avaliador lê inteiro)"
    assert all(i["cot"].rstrip().endswith(f"Resposta: {i['resposta']}") for i in itens)


def test_todas_as_familias_aparecem_e_tem_split(tmp_path):
    manifesto = g.gerar(cfg(tmp_path))
    itens = [json.loads(l) for l in (Path(manifesto["pasta"]) / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {i["familia"] for i in itens} == set(g.FAMILIAS)
    assert {i["split"] for i in itens} == {"treino", "teste", "dificil"}


def test_split_dificil_tem_mais_operadores(tmp_path):
    """É o ponto do experimento: treinar pequeno, testar grande."""
    manifesto = g.gerar(cfg(tmp_path))
    itens = [json.loads(l) for l in (Path(manifesto["pasta"]) / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    media = lambda split: sum(i["operadores"] for i in itens if i["split"] == split) / len([i for i in itens if i["split"] == split])  # noqa: E731
    assert media("dificil") > media("treino")


def test_enunciado_nao_contem_a_palavra_resposta(tmp_path):
    """responder_direta monta 'enunciado + Resposta:'; um 'Resposta:' no meio quebraria o prompt."""
    manifesto = g.gerar(cfg(tmp_path))
    itens = [json.loads(l) for l in (Path(manifesto["pasta"]) / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("Resposta" not in i["enunciado"] for i in itens)
    assert all(g.LEGENDA in i["enunciado"] for i in itens)


def test_corpus_de_treino_casa_com_o_numero_de_itens(tmp_path):
    manifesto = g.gerar(cfg(tmp_path))
    trem = (Path(manifesto["pasta"]) / "trem.txt").read_text(encoding="utf-8")
    assert trem.count("Resposta:") == 60
    assert trem.count(g.COT) == 60


def test_determinismo_e_sensibilidade_a_semente(tmp_path):
    primeiro = g.gerar(cfg(tmp_path, id="a"))
    segundo = g.gerar(cfg(tmp_path, id="a"))
    outro = g.gerar(cfg(tmp_path, id="b", semente=7))
    assert primeiro["saidas"]["trem"]["sha256"] == segundo["saidas"]["trem"]["sha256"]
    assert primeiro["saidas"]["trem"]["sha256"] != outro["saidas"]["trem"]["sha256"]


def test_benchmark_carrega_com_o_leitor_da_g5(tmp_path):
    """Compatibilidade com o encanamento existente — sem isso, nada do resto serve."""
    manifesto = g.gerar(cfg(tmp_path))
    caminho = Path(manifesto["pasta"]) / "benchmark.jsonl"
    assert len(carregar_benchmark(caminho, split="teste")) == 15
    assert len(carregar_benchmark(caminho, split="dificil")) == 10
    item = carregar_benchmark(caminho, split="teste", limite=1)[0]
    assert set(item) >= {"id", "familia", "split", "enunciado", "cot", "resposta"}


# --------------------------------------------------- configuração

def test_config_recusa_entrada_invalida(tmp_path):
    with pytest.raises(ValueError, match="id inválido"):
        g.ConfigLogica(id="com espaço")
    with pytest.raises(ValueError, match="familia desconhecida"):
        g.ConfigLogica(id="x", familias=("sudoku",))
    with pytest.raises(ValueError, match="maior que operadores"):
        g.ConfigLogica(id="x", operadores=5, operadores_dificeis=5)
    with pytest.raises(ValueError, match="operadores precisa"):
        g.ConfigLogica(id="x", operadores=0)


def test_variaveis_derivadas_do_numero_pedido(tmp_path):
    assert g.ConfigLogica(id="x", n_variaveis=3).variaveis == ["p", "q", "r"]
    assert g.ConfigLogica(id="x", n_variaveis=1).variaveis == ["p", "q"]
    assert g.ConfigLogica(id="x", variaveis=["a", "b"]).variaveis == ["a", "b"]
