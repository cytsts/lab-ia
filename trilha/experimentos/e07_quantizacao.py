r"""Lição 7 — quantização: o que se perde ao encolher os pesos.

Experimento: compara int8 e NF4 na mesma matriz, medindo tamanho, erro e bits
efetivos por parâmetro — em vez de aceitar "4 bits é melhor" por fé.

Rode:  .venv\Scripts\python trilha\experimentos\e07_quantizacao.py
"""
from __future__ import annotations

import torch

from labia.models.quant import NF4_VALORES, dequant_int8, dequant_nf4, quant_int8, quant_nf4


def matriz_de_teste(linhas: int = 256, colunas: int = 256, semente: int = 0) -> torch.Tensor:
    """Pesos com a cara de uma rede de verdade: gaussianos, com alguns outliers."""
    gerador = torch.Generator().manual_seed(semente)
    w = torch.randn(linhas, colunas, generator=gerador) * 0.05
    w[:, 0] *= 40.0  # coluna outlier, como acontece de fato em transformers
    return w


def comparar_modos(w: torch.Tensor) -> list[dict]:
    linhas = []
    for modo, quantizar, dequantizar in (
        ("int8", quant_int8, dequant_int8),
        ("nf4", quant_nf4, dequant_nf4),
    ):
        buf = quantizar(w)
        recuperado = dequantizar(buf)
        bytes_quant = sum(t.numel() * t.element_size() for t in buf.values())
        bytes_fp32 = w.numel() * 4
        erro = (w - recuperado).norm() / w.norm().clamp_min(1e-8)
        linhas.append(
            {
                "modo": modo,
                "bytes_fp32": bytes_fp32,
                "bytes_quantizado": bytes_quant,
                "fator": round(bytes_fp32 / bytes_quant, 2),
                "bits_por_parametro": round(8 * bytes_quant / w.numel(), 3),
                "erro_relativo": round(float(erro), 5),
                "erro_maximo_absoluto": round(float((w - recuperado).abs().max()), 6),
            }
        )
    return linhas


def niveis_do_nf4() -> dict:
    """NF4 não usa passos iguais: os níveis seguem os quantis de uma gaussiana."""
    v = NF4_VALORES.tolist()
    espacos = [round(v[i + 1] - v[i], 4) for i in range(len(v) - 1)]
    return {
        "niveis": len(v),
        "menor": min(v),
        "maior": max(v),
        "espaco_no_centro": espacos[len(espacos) // 2],
        "espaco_na_borda": espacos[0],
    }


def erro_por_faixa(w: torch.Tensor) -> dict:
    """Onde a quantização erra mais: nos pesos grandes ou nos pequenos?"""
    recuperado = dequant_nf4(quant_nf4(w))
    erro = (w - recuperado).abs()
    magnitudes = w.abs()
    limiar = float(magnitudes.median())
    pequenos = erro[magnitudes <= limiar].mean()
    grandes = erro[magnitudes > limiar].mean()
    return {
        "erro_medio_pesos_pequenos": round(float(pequenos), 6),
        "erro_medio_pesos_grandes": round(float(grandes), 6),
        "limiar_mediana": round(limiar, 5),
    }


def main() -> dict:
    w = matriz_de_teste()
    resultado = {
        "matriz": {"linhas": w.shape[0], "colunas": w.shape[1], "parametros": w.numel()},
        "modos": comparar_modos(w),
        "nf4": niveis_do_nf4(),
        "erro_por_faixa": erro_por_faixa(w),
    }
    print(f"matriz de teste: {w.shape[0]}x{w.shape[1]} ({w.numel():,} parâmetros)".replace(",", "."))
    print()
    print(f"{'modo':>6} {'bytes fp32':>11} {'bytes quant':>12} {'fator':>7} {'bits/param':>11} {'erro relativo':>14}")
    for linha in resultado["modos"]:
        print(
            f"{linha['modo']:>6} {linha['bytes_fp32']:>11} {linha['bytes_quantizado']:>12} "
            f"{linha['fator']:>6}x {linha['bits_por_parametro']:>11} {linha['erro_relativo']:>14}"
        )
    print()
    n = resultado["nf4"]
    print(f"NF4 tem {n['niveis']} níveis entre {n['menor']} e {n['maior']}; o espaço entre eles é")
    print(f"  {n['espaco_na_borda']} na borda e {n['espaco_no_centro']} no centro — quantis de gaussiana,")
    print("  não passos iguais. É por isso que 4 bits não-uniformes competem com 8 bits uniformes.")
    print()
    e = resultado["erro_por_faixa"]
    print("onde o erro cai (NF4):")
    print(f"  pesos pequenos: {e['erro_medio_pesos_pequenos']}")
    print(f"  pesos grandes : {e['erro_medio_pesos_grandes']}")
    print()
    print("Leia assim: quantizar não é 'arredondar tudo'. int8 é simples e barato de")
    print("implementar; NF4 gasta os poucos níveis onde há mais peso na distribuição e")
    print("ainda guarda uma escala por bloco de 64 para não perder os outliers.")
    return resultado


if __name__ == "__main__":
    main()
