"""Testes do LoRA nativo (spec G2, RF1–RF4)."""
from __future__ import annotations

import copy

import pytest
import torch

from labia.models.gpt import ConfigGPT, GPT
from labia.models.lora import (
    aplicar_lora,
    carregar_adaptador,
    estatisticas_lora,
    mesclar_lora,
    salvar_adaptador,
)


def _base(dim=64, camadas=2, cabecas=2):
    cfg = ConfigGPT(vocab=100, dim=dim, camadas=camadas, cabecas=cabecas, janela_ctx=32, abandono=0.0)
    m = GPT(cfg)
    m.init_pesos(1)
    return m.eval()


def _logits(m, x):
    with torch.no_grad():
        return m(x)[0]


def test_adaptador_comece_identico_a_base():
    """RF1/CA: B zerado => saída do modelo com LoRA é a saída da base."""
    torch.manual_seed(0)
    m = _base()
    x = torch.randint(0, 100, (2, 8))
    antes = _logits(m, x)
    aplicar_lora(m, r=4, alpha=8)
    depois = _logits(m, x)
    assert torch.allclose(antes, depois, atol=1e-5)


def test_so_adaptador_treinavel():
    torch.manual_seed(0)
    m = _base()
    aplicar_lora(m, r=4, alpha=8)
    for nome, p in m.named_parameters():
        eh_adaptador = nome.endswith(".lora_A") or nome.endswith(".lora_B")
        assert p.requires_grad == eh_adaptador, nome


def test_treinaveis_ate_5_por_cento_na_config_g1():
    m = _base(dim=256, camadas=6, cabecas=8)
    stats = aplicar_lora(m, r=8, alpha=16)
    assert stats["proporcao"] <= 0.05, f"adaptador grande demais: {stats}"


def test_salvar_e_carregar_adaptador(tmp_path):
    torch.manual_seed(0)
    m = _base()
    aplicar_lora(m, r=4, alpha=8)
    with torch.no_grad():  # dá trabalho ao adaptador
        for nome, p in m.named_parameters():
            if nome.endswith(".lora_A"):
                p.add_(torch.randn_like(p) * 0.05)
        for nome, p in m.named_parameters():
            if nome.endswith(".lora_B"):
                p.add_(torch.randn_like(p) * 0.05)
    x = torch.randint(0, 100, (1, 8))
    esperado = _logits(m, x)

    salvar_adaptador(m, tmp_path / "adaptador", meta_extra={"base": "g1-teste"})
    m2 = _base()
    stats = aplicar_lora(m2, r=4, alpha=8)
    carregar_adaptador(m2, tmp_path / "adaptador")
    assert torch.allclose(esperado, _logits(m2, x), atol=1e-5)
    import json

    meta = json.loads((tmp_path / "adaptador" / "meta.json").read_text(encoding="utf-8"))
    assert meta["r"] == 4 and meta["alpha"] == 8 and meta["base"] == "g1-teste"


def test_mesclar_preserva_saida(tmp_path):
    """RF4/CA3: modelo mesclado idêntico a base+adaptador, sem ramos LoRA."""
    torch.manual_seed(0)
    m = _base()
    aplicar_lora(m, r=4, alpha=8)
    with torch.no_grad():
        for nome, p in m.named_parameters():
            if nome.endswith(".lora_B"):
                p.copy_(torch.randn_like(p) * 0.1)
    x = torch.randint(0, 100, (1, 8))
    esperado = _logits(m, x)
    mesclar_lora(m)
    assert torch.allclose(esperado, _logits(m, x), rtol=1e-4, atol=1e-5)
    assert not any(nome.endswith(".lora_A") for nome, _ in m.named_parameters())


def test_ramos_lora_nascem_no_dispositivo_da_base():
    """Regressão (caderno 08): aplicar LoRA em modelo já em CUDA quebrava o forward
    — lora_A/lora_B eram criados na CPU e estouravam em matmul de dispositivos mistos."""
    if not torch.cuda.is_available():
        pytest.skip("requer CUDA")
    m = _base().cuda()
    aplicar_lora(m, r=4, alpha=8)
    for nome, p in m.named_parameters():
        if nome.endswith(".lora_A") or nome.endswith(".lora_B"):
            assert p.device.type == "cuda", nome
    x = torch.randint(0, 100, (2, 8), device="cuda")
    assert _logits(m, x).device.type == "cuda"


def test_salvar_adaptador_sem_ramos_falha_em_vez_de_gravar_vazio(tmp_path):
    """Regressão (caderno 08): salvar depois de fundir gravava meta={} silencioso."""
    m = _base()
    with pytest.raises(ValueError, match="nenhum adaptador"):
        salvar_adaptador(m, tmp_path / "adaptador")
