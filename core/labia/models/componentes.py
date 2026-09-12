r"""Peças de arquitetura modernas (pós-GPT-2), todas escritas aqui.

O laboratório nasceu com uma arquitetura só — decoder-only estilo GPT-2: LayerNorm,
embedding de posição aprendido, FFN com GELU, atenção multi-cabeça pura. Isso é o
começo da história, não o estado da arte. Este módulo traz o que mudou depois:

    RMSNorm   (Zhang & Sennrich 2019)  — normaliza sem subtrair a média e sem viés
    RoPE      (Su et al. 2021)         — posição por rotação, não por embedding
    SwiGLU    (Shazeer 2020)           — FFN com portão, 3 matrizes em vez de 2
    GQA       (Ainslie et al. 2023)    — menos cabeças de K/V que de Q

Cada peça é opcional na config, com o padrão igual ao comportamento antigo: os
checkpoints que já existem continuam carregando e gerando igual (há teste para isso).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Normalização por raiz da média quadrática: sem média, sem viés.

    LayerNorm subtrai a média e reescala pelo desvio. RMSNorm joga a média fora e
    divide pela raiz da média dos quadrados — menos contas, mesmo efeito prático, e é
    o que Llama, Mistral e companhia usam. O peso é inicializado em 1.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.peso = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        escala = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * escala * self.peso


def rope_frequencias(
    janela: int, dim_cabeca: int, base: float = 10000.0, dispositivo=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tabela de cossenos e senos por posição, no formato que Llama usa.

    Cada par de dimensões gira num ritmo diferente: pares iniciais giram rápido (posição
    local), pares finais giram devagar (posição global). O resultado tem forma
    (janela, dim_cabeca), com cada frequência duplicada para casar com rotate_half.
    """
    if dim_cabeca % 2:
        raise ValueError(f"dim_cabeca precisa ser par para RoPE (recebi {dim_cabeca})")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim_cabeca, 2, dtype=torch.float32) / dim_cabeca))
    posicoes = torch.arange(janela, dtype=torch.float32)
    angulos = torch.outer(posicoes, inv_freq)          # (janela, dim_cabeca/2)
    emb = torch.cat((angulos, angulos), dim=-1)        # (janela, dim_cabeca)
    return emb.cos().to(dispositivo), emb.sin().to(dispositivo)


def _rotacionar_metades(x: torch.Tensor) -> torch.Tensor:
    """Troca as metades trocando o sinal: usado pela rotação do RoPE."""
    primeira, segunda = x.chunk(2, dim=-1)
    return torch.cat((-segunda, primeira), dim=-1)


def aplicar_rope(x: torch.Tensor, cos: torch.Tensor, sen: torch.Tensor) -> torch.Tensor:
    """Gira q ou k pela posição. x tem forma (lote, cabeças, tokens, dim_cabeca)."""
    tokens = x.shape[-2]
    # cos/sen nascem em float32; sob autocast (bf16) é preciso acompanhar o dtype de x,
    # senão a multiplicação promove tudo para float32 e o ganho do autocast some.
    c = cos[:tokens].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    s = sen[:tokens].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return x * c + _rotacionar_metades(x) * s


def intermediario_swiglu(dim: int, multiplo: int = 16) -> int:
    """Tamanho da camada interna do SwiGLU: 8/3·dim (e não 4·dim), arredondado.

    Com três matrizes em vez de duas, usar 4·dim deixaria o SwiGLU 50% maior que a FFN
    densa. O 8/3 é o ajuste que mantém o custo parecido — é o que a Llama usa
    (arredondando para múltiplo de 256; aqui arredondamos para 16, que basta nestas
    dimensões pequenas).
    """
    bruto = 8 * dim / 3
    return int(math.ceil(bruto / multiplo) * multiplo)


class RedeSwiGLU(nn.Module):
    """FFN com portão: down(silu(gate(x)) * up(x)). Sem viés, como nos modelos modernos."""

    def __init__(self, cfg):
        super().__init__()
        self.intermediario = intermediario_swiglu(cfg.dim)
        self.w_gate = nn.Linear(cfg.dim, self.intermediario, bias=False)
        self.w_up = nn.Linear(cfg.dim, self.intermediario, bias=False)
        self.w_down = nn.Linear(self.intermediario, cfg.dim, bias=False)
        self.abandono = nn.Dropout(cfg.abandono)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return self.abandono(y), torch.zeros((), device=y.device)
