r"""Lição 9 — raciocínio: pedir o passo a passo ajuda? Quando e por quanto?

Experimento: mede o que a estratégia muda de fato — quantos tokens ela gasta, o
quanto o formato da resposta é exigido, e o resultado real do benchmark deste
laboratório (lido do run da G5, medido nesta máquina).

Rode:  .venv\Scripts\python trilha\experimentos\e09_raciocinio.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from labia.models.gpt import GPT, ConfigGPT
from labia.reasoning.estrategias import (
    COT,
    carregar_benchmark,
    parse_resposta,
    responder_cot,
    responder_direta,
)
from labia.trainer.tokenizacao import carregar_tokenizer, treinar_tokenizer_ptbr

BENCHMARK = Path("data/benchmark_matematica.jsonl")
COMPARATIVO = Path("runs/g5-ajuste-cot/comparativo.json")
ENUNCIADO = "Quanto é 7 + 5?"


def o_formato_e_exigido() -> dict:
    """A métrica só aceita 'Resposta: N'. Texto certo no formato errado conta como erro."""
    exemplos = {
        "formato correto": "Resposta: 12",
        "com passo a passo": "7 + 5 = 12\\nResposta: 12",
        "texto livre": "doze",
        "sem a palavra-chave": "12",
    }
    return {chave: parse_resposta(texto) for chave, texto in exemplos.items()}


def tokens_por_estrategia() -> dict:
    """CoT gera muito mais tokens: a conta que o passo a passo cobra."""
    texto = "Quanto é 7 + 5?\\nResposta: 12\\n\\nQuanto é 3 + 4?\\nResposta: 7\\n\\n" * 20
    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta) / "tok"
        treinar_tokenizer_ptbr([texto], 256, destino / "tokenizer.json")
        tokenizer = carregar_tokenizer(destino)
        cfgm = ConfigGPT(vocab=tokenizer.get_vocab_size(), dim=64, camadas=2, cabecas=4, janela_ctx=48, abandono=0.0)
        torch.manual_seed(0)
        modelo = GPT(cfgm)
        modelo.init_pesos(0)
        modelo.eval()
        dispositivo = torch.device("cpu")
        ids_prompt_direta = tokenizer.encode(f"{ENUNCIADO} Resposta:", add_special_tokens=False).ids
        ids_prompt_cot = tokenizer.encode(f"{ENUNCIADO}\\n{COT}\\n", add_special_tokens=False).ids
        _, texto_direta = responder_direta(modelo, tokenizer, dispositivo, ENUNCIADO)
        _, texto_cot = responder_cot(modelo, tokenizer, dispositivo, ENUNCIADO, guloso=True)
    return {
        "tokens_prompt_direta": len(ids_prompt_direta),
        "tokens_prompt_cot": len(ids_prompt_cot),
        "tokens_gerados_direta": len(tokenizer.encode(texto_direta, add_special_tokens=False).ids),
        "tokens_gerados_cot": len(tokenizer.encode(texto_cot, add_special_tokens=False).ids),
        "modelo_treinado": False,
        "observacao": "modelo aleatório: mede o CUSTO da estratégia, não a acurácia",
    }


def resultado_medido_na_g5() -> dict | None:
    """Lê o comparativo do run real, se ele existir neste repositório."""
    if not COMPARATIVO.exists():
        return None
    dados = json.loads(COMPARATIVO.read_text(encoding="utf-8"))
    estrategias = dados.get("estrategias", {})
    return {
        "direta": estrategias.get("direta", {}).get("acuracia_global"),
        "cot": estrategias.get("cot", {}).get("acuracia_global"),
        "tot": estrategias.get("tot", {}).get("acuracia_global"),
        "cot_menos_direta_pp": dados.get("cot_menos_direta_pp"),
        "tot_menos_cot_pp": dados.get("tot_menos_cot_pp"),
        "itens": estrategias.get("direta", {}).get("total"),
    }


def benchmark_disponivel() -> dict:
    if not BENCHMARK.exists():
        return {"existe": False}
    itens = carregar_benchmark(BENCHMARK, split="teste")
    return {
        "existe": True,
        "itens_teste": len(itens),
        "exemplo": itens[0] if itens else None,
    }


def main() -> dict:
    resultado = {
        "formato": o_formato_e_exigido(),
        "custo": tokens_por_estrategia(),
        "benchmark": benchmark_disponivel(),
        "resultado_g5": resultado_medido_na_g5(),
        "frase_cot": COT,
    }
    print("1) o formato da resposta é parte da métrica")
    for chave, valor in resultado["formato"].items():
        print(f"   {chave:<20} → {valor}")
    print()
    c = resultado["custo"]
    print("2) quanto cada estratégia gasta (mesmo enunciado)")
    print(f"   prompt da resposta direta: {c['tokens_prompt_direta']} tokens")
    print(f"   prompt do CoT            : {c['tokens_prompt_cot']} tokens (a frase '{COT}')")
    print(f"   gerados pela direta      : {c['tokens_gerados_direta']} tokens")
    print(f"   gerados pelo CoT         : {c['tokens_gerados_cot']} tokens")
    print()
    b = resultado["benchmark"]
    if b.get("existe"):
        print(f"3) benchmark deste laboratório: {b['itens_teste']} itens de teste")
        if b.get("exemplo"):
            print(f"   exemplo: {str(b['exemplo'])[:110]}")
    print()
    r = resultado["resultado_g5"]
    if r:
        print("4) resultado real do run g5-ajuste-cot (medido nesta máquina, spec G5)")
        print(f"   direta {r['direta']:.3f} · CoT {r['cot']:.3f} · ToT {r['tot']:.3f}  ({r['itens']} itens)")
        print(f"   CoT − direta = {r['cot_menos_direta_pp']:+.1f} p.p. · ToT − CoT = {r['tot_menos_cot_pp']:+.1f} p.p.")
    else:
        print("4) run g5-ajuste-cot não encontrado — rode 'lab-ia raciocinio --run g5-ajuste-cot --comparar'")
    print()
    print("Leia assim: pedir o passo a passo troca tokens por acerto. No run real deste")
    print("laboratório, o CoT ganhou 5,3 p.p. sobre a resposta direta, e o ToT — que gasta")
    print("muito mais — não ganhou nada além disso. E note o item 1: a métrica exige o")
    print("formato 'Resposta: N'. Resposta certa escrita de outro jeito conta como erro,")
    print("então parte do ganho do CoT é aprender o formato, não a aritmética.")
    return resultado


if __name__ == "__main__":
    main()
