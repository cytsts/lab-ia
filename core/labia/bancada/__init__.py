"""Bancada de experimentos do Lab-IA: dados próprios, configs assistidas, comparação.

Este pacote é a camada que transforma o laboratório de *demo fechada* em
*bancada de uso contínuo*: trazer seus dados, gerar configurações explicadas,
comparar runs lado a lado, varrer hiperparâmetros e desenhar curvas.

Módulos:
    dados    — ingestão, limpeza, deduplicação, split e manifesto de corpus próprio
    presets  — presets de modelo, estimativas (parâmetros/VRAM/tempo) e YAML comentado
"""
from __future__ import annotations

from . import dados, presets

__all__ = ["dados", "presets"]
