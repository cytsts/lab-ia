"""Logica proposicional: geracao de problemas com resposta verificada por tabela-verdade.

O nome do modulo diz o que ele gera: 'gerador_proposicional' produz datasets de
problemas de logica proposicional (avaliacao, tautologia, satisfatibilidade,
equivalencia e consequencia logica), nao 'logica' em geral.
"""
from __future__ import annotations

from . import gerador_proposicional

__all__ = ["gerador_proposicional"]
