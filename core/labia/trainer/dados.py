"""Preparação de dados: split trem/validação determinístico e janelas de tokens."""
from __future__ import annotations

import random

import torch


def dividir_corpus(texto: str, frac_trem: float = 0.95, semente: int = 42) -> tuple[str, str]:
    """Divide por parágrafos (baralho com seed) — 95% trem / 5% validação."""
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    if not paragrafos:
        raise ValueError("corpus vazio")
    rnd = random.Random(semente)
    rnd.shuffle(paragrafos)
    corte = max(1, int(len(paragrafos) * frac_trem))
    trem = "\n\n".join(paragrafos[:corte])
    val = "\n\n".join(paragrafos[corte:]) or trem[:2000]  # evita split vazio em corpus mínimo
    return trem, val


def montar_dataset(
    tokenizer, texto: str, janela: int, stride: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Codifica o texto e corta janelas: X[:, t] -> Y[:, t] (alvo é o turno à esquerda).

    stride=None usa janela inteira (sem sobreposição); stride < janela aumenta o
    aproveitamento de corpus pequenos (janelas deslizantes).
    """
    passo = stride or janela
    ids = tokenizer.encode(texto, add_special_tokens=False).ids
    n = (len(ids) - janela - 1) // passo + 1  # toda janela precisa de alvo completo (janela+1 tokens)
    if n < 1:
        raise ValueError(f"tokens insuficientes para janela {janela} (obtive {len(ids)} tokens)")
    x = torch.stack([torch.tensor(ids[i * passo : i * passo + janela]) for i in range(n)])
    y = torch.stack([torch.tensor(ids[i * passo + 1 : i * passo + janela + 1]) for i in range(n)])
    return x, y


def lote_trem(x: torch.Tensor, y: torch.Tensor, passo: int, lote: int, semente: int, passos_por_epoch: int):
    """Lote determinístico do passo global: mesma permutação por epoch para qualquer processo."""
    epoch = passo // passos_por_epoch
    dentro = passo % passos_por_epoch
    gerador = torch.Generator().manual_seed(semente + 9_973 * epoch)
    ordem = torch.randperm(x.shape[0], generator=gerador)
    ini = dentro * lote
    idx = ordem[ini : ini + lote]
    return x[idx], y[idx]
