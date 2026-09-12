r"""Lição 1 — o tensor, o grafo e o gradiente (autograd).

Experimento: mede três coisas que costumam ser aceitas por fé.
  1. o gradiente do autograd bate com a derivada analítica;
  2. o gradiente ACUMULA se você esquecer zero_grad;
  3. o autograd bate com diferença finita (o teste que pega gradiente errado).

Rode:  .venv\Scripts\python trilha\experimentos\e01_autograd.py
"""
from __future__ import annotations

import torch


def gradiente_analitico() -> dict:
    """f(x) = soma(x^3) → df/dx = 3x^2. Compara autograd com a conta no papel."""
    x = torch.tensor([2.0, -3.0, 0.5], requires_grad=True)
    y = (x**3).sum()
    y.backward()
    esperado = 3 * torch.tensor([2.0, -3.0, 0.5]) ** 2
    return {
        "x": x.detach().tolist(),
        "autograd": x.grad.tolist(),
        "analitico": esperado.tolist(),
        "erro_maximo": float((x.grad - esperado).abs().max()),
    }


def acumulacao_de_gradiente() -> dict:
    """Sem zerar, cada backward SOMA no .grad — a causa nº 1 de treino que não aprende."""
    x = torch.tensor([1.5], requires_grad=True)
    historico = []
    for _ in range(3):
        ((x**2).sum()).backward()          # sem zero_grad de propósito
        historico.append(round(float(x.grad), 4))
    x.grad = None                           # equivalente a zero_grad(set_to_none=True)
    ((x**2).sum()).backward()
    depois_de_zerar = round(float(x.grad), 4)
    return {
        "grad_apos_cada_backward": historico,
        "esperado_um_passo": round(2 * 1.5, 4),
        "apos_zero_grad": depois_de_zerar,
    }


def diferenca_finita(f=None, x0: float = 1.3, h: float = 1e-5) -> dict:
    """Confere o autograd contra (f(x+h) - f(x-h)) / 2h — o teste que pega gradiente errado."""
    f = f or (lambda t: torch.sin(t) * t**2)
    x = torch.tensor([x0], requires_grad=True, dtype=torch.float64)
    f(x).backward()
    autograd = float(x.grad)
    numerico = (float(f(torch.tensor([x0 + h], dtype=torch.float64))) - float(f(torch.tensor([x0 - h], dtype=torch.float64)))) / (2 * h)
    return {
        "ponto": x0,
        "autograd": round(autograd, 8),
        "diferenca_finita": round(numerico, 8),
        "erro_relativo": round(abs(autograd - numerico) / max(1e-12, abs(autograd)), 8),
    }


def main() -> dict:
    resultado = {
        "analitico": gradiente_analitico(),
        "acumulacao": acumulacao_de_gradiente(),
        "numerico": diferenca_finita(),
    }
    a, b, c = resultado["analitico"], resultado["acumulacao"], resultado["numerico"]
    print("1) autograd x derivada analítica  (f = soma(x^3))")
    print(f"   x        = {a['x']}")
    print(f"   autograd = {a['autograd']}")
    print(f"   analítico= {a['analitico']}")
    print(f"   erro máximo = {a['erro_maximo']:.2e}")
    print()
    print("2) acúmulo de gradiente sem zero_grad  (f = x^2 em x=1.5, esperado 3.0)")
    print(f"   .grad depois de cada backward: {b['grad_apos_cada_backward']}")
    print(f"   depois de zerar             : {b['apos_zero_grad']}")
    print()
    print("3) autograd x diferença finita  (f = sin(x)·x²)")
    print(f"   autograd        = {c['autograd']}")
    print(f"   diferença finita= {c['diferenca_finita']}")
    print(f"   erro relativo   = {c['erro_relativo']:.2e}")
    return resultado


if __name__ == "__main__":
    main()
