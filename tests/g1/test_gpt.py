"""Testes da arquitetura GPT (spec G1, base de RF1/RF6)."""
from __future__ import annotations

import torch

from labia.models.gpt import ConfigGPT, GPT


def _cfg_minima():
    return ConfigGPT(vocab=100, dim=32, camadas=2, cabecas=2, janela_ctx=16, abandono=0.0)


def test_formatos_de_saida():
    modelo = GPT(_cfg_minima())
    idx = torch.randint(0, 100, (3, 12))
    logits, perda = modelo(idx, alvos=idx.clone())
    assert logits.shape == (3, 12, 100)
    assert perda is not None and perda.dim() == 0


def test_parametros_positivos():
    modelo = GPT(_cfg_minima())
    assert modelo.contar_parametros() > 10_000


def test_causalidade_mudanca_futura():
    """Mudar tokens futuros não pode alterar logits de posições passadas (mask causal)."""
    torch.manual_seed(0)
    modelo = GPT(_cfg_minima())
    idx = torch.randint(0, 100, (1, 10))
    alterado = idx.clone()
    alterado[0, 6:] = torch.randint(0, 100, (4,))
    modelo.eval()
    with torch.no_grad():
        a, _ = modelo(idx)
        b, _ = modelo(alterado)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)


def test_gradientes_fluxionam():
    modelo = GPT(_cfg_minima())
    idx = torch.randint(0, 100, (2, 8))
    _, perda = modelo(idx, alvos=idx.clone())
    perda.backward()
    assert modelo.wte.weight.grad is not None
    assert torch.isfinite(modelo.wte.weight.grad).all()


def test_geracao_produz_tokens_extras():
    modelo = GPT(_cfg_minima())
    idx = torch.randint(0, 100, (1, 4))
    saida = modelo.gerar(idx, passos_max=6, temperatura=0.0)
    assert saida.shape == (1, 10)
    assert torch.equal(saida[:, :4], idx)
