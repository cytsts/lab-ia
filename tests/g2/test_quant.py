"""Testes de quantização nativa int8/NF4 (spec G2 RF5; base da G3)."""
from __future__ import annotations

import torch

from labia.models.quant import dequant_nf4, dequant_int8, quant_nf4, quant_int8, tamanho_state_dict


def _matriz(out=96, entrada=192, semente=3):
    g = torch.Generator().manual_seed(semente)
    return torch.randn(out, entrada, generator=g)


def test_int8_roundtrip_e_armazenamento():
    w = _matriz()
    buf = quant_int8(w)
    assert set(buf) >= {"peso", "escala"}
    assert buf["peso"].dtype == torch.int8
    w2 = dequant_int8(buf)
    erro = (w - w2).norm() / w.norm()
    assert erro < 0.02, f"erro relativo int8 alto: {erro}"
    bytes_quant = buf["peso"].numel() * 1 + buf["escala"].numel() * 4
    assert bytes_quant * 2 < w.numel() * 4, "int8 deve reduzir armazenamento em ≥2x"


def test_nf4_roundtrip_e_armazenamento():
    w = _matriz()
    buf = quant_nf4(w, bloco=64)
    assert buf["codigos"].dtype == torch.uint8
    assert buf["codigos"].numel() == w.numel() // 2, "NF4 deve empacotar 2 códigos por byte"
    w2 = dequant_nf4(buf, bloco=64)
    erro = (w - w2).norm() / w.norm()
    assert erro < 0.10, f"erro relativo NF4 alto: {erro}"
    bytes_quant = buf["codigos"].numel() + buf["escalas"].numel() * 2
    assert bytes_quant * 3 < w.numel() * 4, "NF4 deve reduzir armazenamento em ≥3x"


def test_bloco_invalido_rejeitado():
    w = _matriz(out=64, entrada=100)  # 100 não divisível por 64
    try:
        quant_nf4(w, bloco=64)
    except ValueError:
        return
    raise AssertionError("deveria exigir entrada divisível pelo bloco")


def test_qlora_congela_base_quantizada():
    """RF5: base quantizada + LoRA — só adaptador tem grad."""
    from labia.models.gpt import ConfigGPT, GPT
    from labia.models.lora import aplicar_lora, estatisticas_lora

    torch.manual_seed(0)
    m = GPT(ConfigGPT(vocab=64, dim=64, camadas=2, cabecas=2, janela_ctx=32, abandono=0.0))
    m.init_pesos(1)
    stats = aplicar_lora(m, r=4, alpha=8, quant="int8")
    assert stats["quant"] == "int8"
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(".lora_A") or n.endswith(".lora_B") for n in trainable)
    x = torch.randint(0, 64, (1, 8))
    logits, _ = m(x)
    assert torch.isfinite(logits).all()
