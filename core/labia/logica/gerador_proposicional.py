r"""Gerador de problemas de lógica proposicional, com resposta verificada por tabela-verdade.

Por que existe: o laboratório tinha um benchmark de aritmética (G5) e nenhuma forma de
gerar problemas de lógica. Aqui a tarefa é criada e a resposta é CALCULADA, nunca
escrita à mão: para toda fórmula existe a avaliação direta e, para as famílias de
classificação, a varredura de todas as atribuições. As duas contas são independentes.

Formato de saída: o mesmo do benchmark da G5, para o encanamento existente
(lab-ia raciocinio, LoRA, comparar direta x CoT x ToT) funcionar sem alteração:

    {"id", "familia", "split", "enunciado", "cot", "resposta"}

A resposta é sempre 1 (verdadeiro) ou 0 (falso). Não é capricho: o avaliador da G5
extrai um INTEIRO do texto, então 1/0 mantém compatibilidade total — e o CoT inteiro
usa 1/0, sem misturar V/F com número no mesmo texto.

Famílias: avaliacao, tautologia, satisfativel, equivalencia, implicacao.

A dificuldade é o número de operadores da fórmula — e é ela que permite o experimento
que interessa: treinar nos casos pequenos e medir se generaliza para os grandes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..bancada.dados import estimar_tokens, estatisticas

SIMBOLOS = {
    "e": "E",
    "ou": "OU",
    "xor": "OU-EX",
    "imp": "=>",
    "bi": "<=>",
}
FAMILIAS = ("avaliacao", "tautologia", "satisfativel", "equivalencia", "implicacao")
LEGENDA = "Use 1 para verdadeiro e 0 para falso."
COT = "Vamos pensar passo a passo."


def avaliar(no, ambiente: dict) -> bool:
    """Avalia a árvore sob uma atribuição. É a definição, não uma aproximação."""
    tipo = no[0]
    if tipo == "var":
        return bool(ambiente[no[1]])
    if tipo == "nao":
        return not avaliar(no[1], ambiente)
    esquerda, direita = avaliar(no[1], ambiente), avaliar(no[2], ambiente)
    if tipo == "e":
        return esquerda and direita
    if tipo == "ou":
        return esquerda or direita
    if tipo == "xor":
        return esquerda != direita
    if tipo == "imp":
        return (not esquerda) or direita
    if tipo == "bi":
        return esquerda == direita
    raise ValueError(f"operador desconhecido: {tipo!r}")


def render(no) -> str:
    """Texto da fórmula, sempre com parênteses — ambiguidade é ruído para o modelo."""
    if no[0] == "var":
        return no[1]
    if no[0] == "nao":
        return f"(NAO {render(no[1])})"
    return f"({render(no[1])} {SIMBOLOS[no[0]]} {render(no[2])})"


def contar_operadores(no) -> int:
    if no[0] == "var":
        return 0
    if no[0] == "nao":
        return 1 + contar_operadores(no[1])
    return 1 + contar_operadores(no[1]) + contar_operadores(no[2])


def variaveis_de(no, achadas=None) -> set:
    achadas = achadas if achadas is not None else set()
    if no[0] == "var":
        achadas.add(no[1])
    elif no[0] == "nao":
        variaveis_de(no[1], achadas)
    else:
        variaveis_de(no[1], achadas)
        variaveis_de(no[2], achadas)
    return achadas


def atribuicoes(variaveis) -> list:
    """Todas as linhas da tabela-verdade, em ordem determinística."""
    nomes = sorted(variaveis)
    return [dict(zip(nomes, valores)) for valores in itertools.product([True, False], repeat=len(nomes))]


def gerar_formula(rnd: random.Random, variaveis: list, operadores: int):
    """Fórmula aleatória com exatamente o número pedido de operadores binários."""
    if operadores <= 0:
        folha = ("var", rnd.choice(variaveis))
        return ("nao", folha) if rnd.random() < 0.15 else folha
    resto = operadores - 1
    a_esquerda = rnd.randint(0, resto)
    esquerda = gerar_formula(rnd, variaveis, a_esquerda)
    direita = gerar_formula(rnd, variaveis, resto - a_esquerda)
    return (rnd.choice(list(SIMBOLOS)), esquerda, direita)


def descrever_atribuicao(ambiente: dict) -> str:
    return ", ".join(f"{nome} = {int(valor)}" for nome, valor in sorted(ambiente.items()))


def classificar(no, variaveis) -> dict:
    """Varredura completa: é aqui que a resposta de classificação nasce."""
    linhas = [(ambiente, avaliar(no, ambiente)) for ambiente in atribuicoes(variaveis)]
    return {
        "linhas": linhas,
        "verdadeiras": [ambiente for ambiente, valor in linhas if valor],
        "falsas": [ambiente for ambiente, valor in linhas if not valor],
    }


def equivalentes(primeira, segunda, variaveis):
    for ambiente in atribuicoes(variaveis):
        if avaliar(primeira, ambiente) != avaliar(segunda, ambiente):
            return False, ambiente
    return True, None


def consequencia_logica(premissas, conclusao, variaveis):
    """Premissas obrigam a conclusão? Contraexemplo é a linha com premissas V e conclusão F."""
    for ambiente in atribuicoes(variaveis):
        if all(avaliar(p, ambiente) for p in premissas) and not avaliar(conclusao, ambiente):
            return False, ambiente
    return True, None


def avaliar_com_passos(no, ambiente: dict):
    """Avalia mostrando o caminho: cada sub-fórmula vira um passo numerado.

    É o que ensina o procedimento — o modelo aprende a avaliar de baixo para cima e,
    depois, reaproveita o resultado pelo número do passo em vez de repetir a conta.
    """
    passos = []
    rotulos = {}

    def rotulo(sub) -> str:
        if sub[0] == "var":
            return sub[1]
        return f"Passo {rotulos[id(sub)]}"

    def visita(sub) -> bool:
        if sub[0] == "var":
            return bool(ambiente[sub[1]])
        if sub[0] == "nao":
            valor = not visita(sub[1])
            passos.append(f"(NAO {rotulo(sub[1])}) = {int(valor)}")
        else:
            visita(sub[1])
            visita(sub[2])
            valor = avaliar(sub, ambiente)
            passos.append(f"({rotulo(sub[1])} {SIMBOLOS[sub[0]]} {rotulo(sub[2])}) = {int(valor)}")
        rotulos[id(sub)] = len(passos)
        return valor

    valor_final = visita(no)
    return valor_final, [f"Passo {i}: {linha}" for i, linha in enumerate(passos, start=1)]

# --------------------------------------------------------------- famílias

def _item_avaliacao(rnd, variaveis, operadores) -> dict:
    formula = gerar_formula(rnd, variaveis, operadores)
    ambiente = {nome: rnd.random() < 0.5 for nome in sorted(variaveis_de(formula))}
    valor, passos = avaliar_com_passos(formula, ambiente)
    enunciado = f"{descrever_atribuicao(ambiente)}. Calcule: {render(formula)}. {LEGENDA}"
    texto = "\n".join([COT, *passos, f"Resposta: {int(valor)}"])
    return {"enunciado": enunciado, "cot": texto, "resposta": int(valor)}


def _item_tautologia(rnd, variaveis, operadores) -> dict:
    formula = gerar_formula(rnd, variaveis, operadores)
    info = classificar(formula, variaveis)
    linhas = [
        f"{descrever_atribuicao(ambiente)}: {render(formula)} = {int(valor)}"
        for ambiente, valor in info["linhas"]
    ]
    if info["falsas"]:
        conclusao = f"A linha {descrever_atribuicao(info['falsas'][0])} torna a formula falsa."
        resposta = 0
    else:
        conclusao = "Todas as linhas sao verdadeiras."
        resposta = 1
    enunciado = f"A formula {render(formula)} e sempre verdadeira? {LEGENDA}"
    return {"enunciado": enunciado, "cot": "\n".join([COT, *linhas, conclusao, f"Resposta: {resposta}"]), "resposta": resposta}


def _item_satisfativel(rnd, variaveis, operadores) -> dict:
    formula = gerar_formula(rnd, variaveis, operadores)
    info = classificar(formula, variaveis)
    if info["verdadeiras"]:
        testemunha = info["verdadeiras"][0]
        linhas = [f"{descrever_atribuicao(testemunha)}: {render(formula)} = 1"]
        conclusao = f"A linha {descrever_atribuicao(testemunha)} torna a formula verdadeira."
        resposta = 1
    else:
        linhas = [f"{descrever_atribuicao(a)}: {render(formula)} = 0" for a, _ in info["linhas"]]
        conclusao = "Nenhuma linha torna a formula verdadeira."
        resposta = 0
    enunciado = f"Existe atribuicao que torne {render(formula)} verdadeira? {LEGENDA}"
    return {"enunciado": enunciado, "cot": "\n".join([COT, *linhas, conclusao, f"Resposta: {resposta}"]), "resposta": resposta}


def _item_equivalencia(rnd, variaveis, operadores) -> dict:
    primeira = gerar_formula(rnd, variaveis, max(1, operadores // 2))
    segunda = gerar_formula(rnd, variaveis, max(1, operadores - operadores // 2))
    todas = sorted(variaveis_de(primeira) | variaveis_de(segunda))
    iguais, contra = equivalentes(primeira, segunda, todas)
    linhas = []
    for ambiente in atribuicoes(todas):
        valor_a = int(avaliar(primeira, ambiente))
        valor_b = int(avaliar(segunda, ambiente))
        linhas.append(f"{descrever_atribuicao(ambiente)}: {render(primeira)} = {valor_a} e {render(segunda)} = {valor_b}")
    if iguais:
        conclusao = "Em todas as linhas os dois valores coincidem."
        resposta = 1
    else:
        valor_a = int(avaliar(primeira, contra))
        valor_b = int(avaliar(segunda, contra))
        conclusao = f"Na linha {descrever_atribuicao(contra)} os valores diferem: {valor_a} e {valor_b}."
        resposta = 0
    enunciado = f"As formulas {render(primeira)} e {render(segunda)} sao equivalentes? {LEGENDA}"
    return {"enunciado": enunciado, "cot": "\n".join([COT, *linhas, conclusao, f"Resposta: {resposta}"]), "resposta": resposta}


def _item_implicacao(rnd, variaveis, operadores) -> dict:
    premissas = [gerar_formula(rnd, variaveis, max(0, operadores - 1)) for _ in range(2)]
    conclusao_formula = gerar_formula(rnd, variaveis, max(0, operadores // 2))
    todas = set()
    for premissa in premissas:
        todas |= variaveis_de(premissa)
    todas |= variaveis_de(conclusao_formula)
    todas = sorted(todas)
    vale, contra = consequencia_logica(premissas, conclusao_formula, todas)
    linhas = []
    for ambiente in atribuicoes(todas):
        valores = ",".join(str(int(avaliar(p, ambiente))) for p in premissas)
        linhas.append(
            f"{descrever_atribuicao(ambiente)}: premissas = {valores} e conclusao = {int(avaliar(conclusao_formula, ambiente))}"
        )
    if vale:
        conclusao = "Em toda linha com as premissas verdadeiras a conclusao tambem e verdadeira."
        resposta = 1
    else:
        valores = ",".join(str(int(avaliar(p, contra))) for p in premissas)
        conclusao = (
            f"Na linha {descrever_atribuicao(contra)} as premissas valem {valores} "
            f"e a conclusao vale {int(avaliar(conclusao_formula, contra))}."
        )
        resposta = 0
    texto_premissas = " e ".join(render(p) for p in premissas)
    enunciado = f"Se {texto_premissas}, entao {render(conclusao_formula)}? {LEGENDA}"
    return {"enunciado": enunciado, "cot": "\n".join([COT, *linhas, conclusao, f"Resposta: {resposta}"]), "resposta": resposta}


GERADORES = {
    "avaliacao": _item_avaliacao,
    "tautologia": _item_tautologia,
    "satisfativel": _item_satisfativel,
    "equivalencia": _item_equivalencia,
    "implicacao": _item_implicacao,
}

def operadores_no_texto(enunciado: str) -> int:
    """Confere a dificuldade contando os operadores que aparecem no enunciado."""
    return sum(enunciado.count(s) for s in (" E ", " OU ", " OU-EX ", " => ", " <=> ", "NAO "))


@dataclass
class ConfigLogica:
    id: str
    familias: tuple = FAMILIAS
    n_variaveis: int = 2
    operadores: int = 3
    operadores_dificeis: int = 5
    n_treino: int = 3000
    n_teste: int = 300
    n_dificil: int = 200
    semente: int = 42
    destino: str = "data"
    raiz: str = "."
    variaveis: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not all(c.isalnum() or c in "-_." for c in self.id):
            raise ValueError(f"id inválido: {self.id!r} (use letras, digitos, hifen, ponto e _)")
        desconhecidas = set(self.familias) - set(FAMILIAS)
        if desconhecidas:
            raise ValueError(f"familia desconhecida: {sorted(desconhecidas)} (use {sorted(FAMILIAS)})")
        if not self.variaveis:
            self.variaveis = ["p", "q", "r", "s", "t"][: max(2, min(self.n_variaveis, 5))]
        if self.operadores < 1:
            raise ValueError("operadores precisa ser >= 1")
        if self.operadores_dificeis <= self.operadores:
            raise ValueError("operadores_dificeis precisa ser maior que operadores (e o salto de dificuldade)")

    def para_dict(self) -> dict:
        return {
            "id": self.id, "familias": list(self.familias), "n_variaveis": self.n_variaveis,
            "operadores": self.operadores, "operadores_dificeis": self.operadores_dificeis,
            "n_treino": self.n_treino, "n_teste": self.n_teste, "n_dificil": self.n_dificil,
            "semente": self.semente, "variaveis": list(self.variaveis),
        }


def gerar_split(rnd: random.Random, cfg: ConfigLogica, split: str, quantidade: int, operadores: int) -> list:
    itens = []
    for indice in range(quantidade):
        familia = cfg.familias[indice % len(cfg.familias)]
        item = GERADORES[familia](rnd, cfg.variaveis, operadores)
        item["id"] = f"{familia}-{split}-{indice:05d}"
        item["familia"] = familia
        item["split"] = split
        item["operadores_planejados"] = operadores
        item["operadores"] = operadores_no_texto(item["enunciado"])
        itens.append(item)
    return itens


def texto_do_item(item: dict) -> str:
    """O item como o treino vê: enunciado e, logo abaixo, a cadeia de passos."""
    return f"{item['enunciado']}\n{item['cot']}\n"


def gerar(cfg: ConfigLogica) -> dict:
    """Gera treino, teste e teste difícil; grava corpus, benchmark e manifesto."""
    raiz = Path(cfg.raiz)
    destino = Path(cfg.destino)
    if not destino.is_absolute():
        destino = raiz / destino
    pasta = destino / cfg.id
    pasta.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(cfg.semente)
    treino = gerar_split(rnd, cfg, "treino", cfg.n_treino, cfg.operadores)
    teste = gerar_split(rnd, cfg, "teste", cfg.n_teste, cfg.operadores)
    dificil = gerar_split(rnd, cfg, "dificil", cfg.n_dificil, cfg.operadores_dificeis)
    todos = [*treino, *teste, *dificil]

    ordem = list(range(len(treino)))
    random.Random(cfg.semente + 1).shuffle(ordem)
    trem_texto = "\n".join(texto_do_item(treino[i]) for i in ordem)
    val_texto = "\n".join(texto_do_item(treino[i]) for i in ordem[: max(1, len(ordem) // 20)])

    (pasta / "trem.txt").write_text(trem_texto, encoding="utf-8")
    (pasta / "val.txt").write_text(val_texto, encoding="utf-8")
    (pasta / "benchmark.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in todos) + "\n", encoding="utf-8"
    )

    por_familia = {}
    for item in todos:
        por_familia[item["familia"]] = por_familia.get(item["familia"], 0) + 1
    media = sum(item["operadores"] for item in todos) / max(1, len(todos))
    manifesto = {
        "versao": 1,
        "id": cfg.id,
        "gerado_em": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "parametros": cfg.para_dict(),
        "saidas": {
            # mesmo schema dos datasets da bancada (B1): chars, sha e contagem, para
            # 'lab-ia dados --listar' e 'lab-ia novo --dados <id>' sem caso especial
            "trem": {"arquivo": f"data/{cfg.id}/trem.txt", "bytes": len(trem_texto.encode("utf-8")),
                     "chars": len(trem_texto), "itens": len(treino),
                     "sha256": hashlib.sha256(trem_texto.encode("utf-8")).hexdigest()[:16]},
            "val": {"arquivo": f"data/{cfg.id}/val.txt", "bytes": len(val_texto.encode("utf-8")),
                    "chars": len(val_texto)},
            "benchmark": {"arquivo": f"data/{cfg.id}/benchmark.jsonl", "itens": len(todos)},
        },
        "estatisticas": {"trem": estatisticas(trem_texto), "val": estatisticas(val_texto)},
        "estimativas": {
            "tokens_aprox_trem": estimar_tokens(len(trem_texto)),
            "tokens_aprox_val": estimar_tokens(len(val_texto)),
            "nota_tokens": "estimativa por 3,6 chars/token; o numero real sai do tokenizer do run",
        },
        "distribuicao": {
            "por_split": {"treino": len(treino), "teste": len(teste), "dificil": len(dificil)},
            "por_familia": por_familia,
            "media_de_operadores": round(media, 2),
        },
        "nota_resposta": "1 = verdadeiro, 0 = falso; o avaliador da G5 le inteiro, entao 1/0 mantem o harness intacto",
        "nota_split_dificil": (
            f"o split 'dificil' usa ate {cfg.operadores_dificeis} operadores contra {cfg.operadores} do treino: "
            "e o teste de generalizacao de comprimento"
        ),
    }
    (pasta / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifesto["pasta"] = str(pasta)
    manifesto["amostras"] = todos[:3]
    return manifesto
