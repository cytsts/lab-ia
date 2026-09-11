"""Quantização nativa p/ o lab: int8 simétrico por linha e NF4 (QLoRA) por blocos.

Formatos persistidos são os reais (int8/uint8 empacotado + escalas fp16), então o
`state_dict` mede a economia de armazenamento de verdade.
"""
from __future__ import annotations

import torch

# Níveis NF4 da fórmula do paper QLoRA (quantisadores não-uniformes p/ gaussiana)
NF4_VALORES = torch.tensor(
    [
        -1.0,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_04,
        0.0,
        0.079_580_29,
        0.160_930_20,
        0.246_112_30,
        0.337_915_24,
        0.440_709_83,
        0.562_617_00,
        0.722_956_84,
        1.0,
    ]
)


def quant_int8(peso: torch.Tensor) -> dict[str, torch.Tensor]:
    """Simétrico por linha de saída: q = round(w/absmax·127)."""
    w = peso.detach().float()
    absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    q = torch.clamp((w / absmax * 127).round(), -127, 127).to(torch.int8)
    return {"peso": q, "escala": absmax.squeeze(1), "forma": torch.tensor(w.shape)}


def dequant_int8(buf: dict[str, torch.Tensor]) -> torch.Tensor:
    q = buf["peso"].float()
    escala = buf["escala"].float()
    saida = (q * escala.unsqueeze(1) / 127.0).reshape(tuple(buf["forma"].tolist()))
    return saida


def quant_nf4(peso: torch.Tensor, bloco: int = 64) -> dict[str, torch.Tensor]:
    """NF4 com escala absmax por bloco; códigos 0..15 empacotados 2/byte (hi<<4|lo)."""
    w = peso.detach().float()
    if w.dim() != 2:
        raise ValueError("quant_nf4 espera matriz 2D (saida, entrada)")
    if w.shape[1] % bloco:
        raise ValueError(f"entrada {w.shape[1]} não divisível pelo bloco {bloco}")
    out, ent = w.shape
    b = w.view(out, ent // bloco, bloco)
    absmax = b.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    xn = (b / absmax).clamp(-1.0, 1.0)
    niveis = NF4_VALORES.to(w.device)
    codigos = torch.argmin((xn.unsqueeze(-1) - niveis).abs(), dim=-1).to(torch.uint8)
    pares = codigos.view(out, ent // 2, 2)
    empacotado = (pares[..., 1] << 4) | pares[..., 0]
    return {
        "codigos": empacotado,
        "escalas": absmax.squeeze(-1).to(torch.float16),
        "forma": torch.tensor(w.shape),
    }


def dequant_nf4(buf: dict[str, torch.Tensor], bloco: int = 64) -> torch.Tensor:
    empacotado = buf["codigos"]
    out, ent2 = empacotado.shape
    ent = ent2 * 2
    lo = empacotado & 0x0F
    hi = empacotado >> 4
    codigos = torch.stack([lo, hi], dim=-1).view(out, ent // bloco, bloco).long()
    valores = NF4_VALORES.to(empacotado.device)[codigos]
    escalas = buf["escalas"].float().unsqueeze(-1)
    return (valores * escalas).reshape(out, ent)


TIPOS = {"int8": quant_int8, "nf4": quant_nf4}


def quantizar(modo: str, peso: torch.Tensor) -> dict[str, torch.Tensor]:
    if modo not in TIPOS:
        raise ValueError(f"modo de quantização desconhecido: {modo!r} (use {sorted(TIPOS)})")
    return TIPOS[modo](peso)


def tamanho_state_dict(state: dict) -> int:
    """Bytes efetivos de um state_dict (tensors)."""
    return sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor))
