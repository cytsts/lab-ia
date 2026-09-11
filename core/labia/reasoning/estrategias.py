"""Estratégias de decodificação para raciocínio: direta, CoT e ToT (spec G5).

ToT: árvore de passos com busca gulosa limitada (expande o melhor nó folha a
cada rodada, k filhos por expansão), pontuada por log-probabilidade do próprio
modelo — sem oracle externo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

COT = "Vamos pensar passo a passo."
_PARSE = re.compile(r"Resposta:\s*(-?\d+)")
_NUM = re.compile(r"-?\d+")


def parse_resposta(texto: str) -> int | None:
    m = _PARSE.search(texto)
    return int(m.group(1)) if m else None


def carregar_benchmark(caminho: Path | str, split: str = "teste", limite: int | None = None) -> list[dict]:
    itens = [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l.strip()]
    itens = [i for i in itens if i["split"] == split]
    return itens[:limite] if limite else itens


@torch.no_grad()
def _gerar_tokens(
    modelo,
    ids: list[int],
    dispositivo,
    n_max: int = 48,
    temperatura: float = 0.7,
    topo_k: int = 40,
    guloso: bool = False,
    gerador: torch.Generator | None = None,
) -> tuple[list[int], float]:
    """Amostra continuação com log-prob acumulado. Retorna (ids_novos, logprob_soma).

    O corte estruturado (nova linha / "Resposta:") é pós-processado pelas estratégias.
    """
    ctx = modelo.cfg.janela_ctx
    idx = torch.tensor([ids], dtype=torch.long, device=dispositivo)
    novos: list[int] = []
    soma_lp = 0.0
    for _ in range(n_max):
        janela = idx[:, -ctx:]
        logits, _ = modelo(janela)
        ultimo = logits[:, -1, :].float()
        if guloso or temperatura <= 0:
            escolh = int(ultimo.argmax(-1))
            lp = float(torch.log_softmax(ultimo, dim=-1)[0, escolh])
        else:
            logits_t = ultimo / temperatura
            if topo_k:
                limiar = torch.topk(logits_t, min(topo_k, logits_t.shape[-1]), dim=-1).values[:, -1:]
                logits_t = logits_t.masked_fill(logits_t < limiar, float("-inf"))
            probs = torch.softmax(logits_t, dim=-1)
            escolh = int(torch.multinomial(probs, num_samples=1, generator=gerador))
            lp = float(torch.log_softmax(ultimo, dim=-1)[0, escolh])
        soma_lp += lp
        novos.append(escolh)
        idx = torch.cat([idx, torch.tensor([[escolh]], device=dispositivo)], dim=1)
    return novos, soma_lp


def _decodigtok(tok, ids: list[int]) -> str:
    return tok.decode(ids)


def responder_direta(modelo, tok, dispositivo, enunciado: str, semente: int = 1234) -> tuple[int | None, str]:
    ids = tok.encode(f"{enunciado} Resposta:", add_special_tokens=False).ids
    gen, _ = _gerar_tokens(modelo, ids, dispositivo, n_max=10, guloso=True)
    texto = _decodigtok(tok, gen)
    numero = _NUM.search(texto)
    return (int(numero.group(0)) if numero else None), texto


def responder_cot(
    modelo, tok, dispositivo, enunciado: str, semente: int = 1234, guloso: bool = True
) -> tuple[int | None, str]:
    ids = tok.encode(f"{enunciado}\n{COT}\n", add_special_tokens=False).ids
    novos, _ = _gerar_tokens(
        modelo,
        ids,
        dispositivo,
        n_max=64,
        temperatura=0.7,
        guloso=guloso,
        gerador=torch.Generator(device=dispositivo).manual_seed(semente),
    )
    texto = _decodigtok(tok, novos)
    if "Resposta:" in texto:
        texto = texto.split("Resposta:")[0] + "Resposta:" + texto.split("Resposta:")[1].split("\n")[0]
    return parse_resposta(texto), texto


def responder_tot(
    modelo,
    tok,
    dispositivo,
    enunciado: str,
    k: int = 3,
    profundidade: int = 4,
    semente: int = 1234,
) -> tuple[int | None, str]:
    """ToT limitado: nós = prefixos de passos; valor = log-prob médio do modelo."""
    gen = torch.Generator(device=dispositivo).manual_seed(semente)
    base_ids = tok.encode(f"{enunciado}\n{COT}\n", add_special_tokens=False).ids
    folhas: list[tuple[float, list[int]]] = [(0.0, list(base_ids))]
    terminais: list[tuple[float, str]] = []
    for _ in range(profundidade):
        if not folhas:
            break
        folhas.sort(key=lambda f: -f[0])
        melhor_lp, melhor_ids = folhas.pop(0)
        filhos = []
        for _j in range(k):
            novos, lp = _gerar_tokens(
                modelo,
                melhor_ids,
                dispositivo,
                n_max=16,
                temperatura=0.9,
                topo_k=40,
                gerador=gen,
            )
            texto = _decodigtok(tok, novos)
            corte = texto.split("\n")[0]
            ids_cortados = _recode_prefixo(tok, melhor_ids, corte)
            valor = (melhor_lp + lp) / max(1, len(ids_cortados) - len(base_ids) + 1)
            if "Resposta:" in texto:
                resposta = parse_resposta(texto)
                terminais.append((valor, texto))
            else:
                folhas.append((valor * (len(ids_cortados) - len(base_ids) + 1), ids_cortados))
    if terminais:
        terminais.sort(key=lambda t: -t[0])
        valor, texto = terminais[0]
        return parse_resposta(texto), texto
    resposta, texto = responder_cot(modelo, tok, dispositivo, enunciado, semente=semente + 1)
    return resposta, texto


def _recode_prefixo(tok, ids_base: list[int], texto_prefixo: str) -> list[int]:
    """Reconstrói ids até o prefixo textual aceito (corte no primeiro \\n)."""
    if not texto_prefixo:
        return list(ids_base)
    recod = tok.encode("«" + texto_prefixo, add_special_tokens=False).ids
    marcador = tok.encode("«", add_special_tokens=False).ids
    retorno = ids_base + recod[len(marcador) :]
    return retorno
