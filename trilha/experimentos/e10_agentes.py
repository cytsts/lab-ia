r"""Lição 10 — agentes: o modelo escolhendo o que executar (e o log provando).

Experimento: mede o que um agente deste laboratório é de fato — um conjunto de
funções com assinatura declarada, chamadas por nome, com cada tentativa registrada
em log (inclusive as que falham).

Rode:  .venv\Scripts\python trilha\experimentos\e10_agentes.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from labia.agents import REGISTRO, criar_agentes


def inventario() -> dict:
    """Quem existe, com que papel e quantas skills expostas."""
    with tempfile.TemporaryDirectory() as pasta:
        agentes = criar_agentes(pasta)
        return {
            "agentes": sorted(agentes),
            "por_agente": {
                nome: {
                    "papel": agente.papel,
                    "skills": [s["nome"] for s in agente.expor_skills()],
                    "assinaturas": [s["assinatura"] for s in agente.expor_skills()],
                }
                for nome, agente in agentes.items()
            },
            "total_skills": sum(len(a.expor_skills()) for a in agentes.values()),
            "registro": sorted(REGISTRO),
        }


def skill_bem_sucedida() -> dict:
    """Uma skill de verdade, executada e auditada no log."""
    with tempfile.TemporaryDirectory() as pasta:
        agente = criar_agentes(pasta)["arquiteto"]
        resultado = agente.executar("suggest_architecture", corpus_bytes=900_000, vram_gb=4.0)
        eventos = _ler_eventos(pasta)
    return {
        "resultado": {k: v for k, v in resultado.items() if isinstance(v, (int, float, str, bool))},
        "eventos": eventos,
    }


def skill_desconhecida_e_registrada_como_falha() -> dict:
    with tempfile.TemporaryDirectory() as pasta:
        agente = criar_agentes(pasta)["treinador"]
        try:
            agente.executar("skill_que_nao_existe")
            levantou = False
        except ValueError as e:
            levantou = True
            mensagem = str(e)
        eventos = _ler_eventos(pasta)
    return {"levantou_erro": levantou, "mensagem": mensagem, "eventos": eventos}


def _ler_eventos(pasta: str | Path) -> list[dict]:
    arquivo = Path(pasta) / ".lab-ia" / "eventos.jsonl"
    if not arquivo.exists():
        return []
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> dict:
    resultado = {
        "inventario": inventario(),
        "sucesso": skill_bem_sucedida(),
        "falha": skill_desconhecida_e_registrada_como_falha(),
    }
    inv = resultado["inventario"]
    print("1) inventário dos agentes")
    for nome in inv["agentes"]:
        dados = inv["por_agente"][nome]
        print(f"   {nome:<11} — {dados['papel']}")
        for nome_skill, assinatura in zip(dados["skills"], dados["assinaturas"]):
            print(f"      · {assinatura}")
    print(f"   total: {len(inv['agentes'])} agentes, {inv['total_skills']} skills")
    print()
    s = resultado["sucesso"]
    print("2) skill executada com sucesso (arquiteto.suggest_architecture)")
    print(f"   resultado: {s['resultado']}")
    print(f"   log: {json.dumps(s['eventos'][0], ensure_ascii=False) if s['eventos'] else 'vazio'}")
    print()
    f = resultado["falha"]
    print("3) skill desconhecida: falha alto E fica registrada")
    print(f"   levantou erro? {f['levantou_erro']} — {f['mensagem'][:90]}")
    print(f"   log: {json.dumps(f['eventos'][0], ensure_ascii=False) if f['eventos'] else 'vazio'}")
    print()
    print("Leia assim: o agente não é um modelo — é um conjunto de funções com nome,")
    print("descrição e assinatura, chamadas por um plano. O que o torna auditável é o log:")
    print("toda tentativa vira uma linha, com ok=true/false e a duração. Se um dia você")
    print("ligar um LLM para escolher a skill, o 'cérebro' muda e a auditoria continua igual.")
    return resultado


if __name__ == "__main__":
    main()
