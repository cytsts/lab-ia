"""Métrica simples de estilo: sobreposição de n-gramas de palavras (geração × corpus)."""
from __future__ import annotations

import re

_PADRAO = re.compile(r"\w+", re.UNICODE)


def _palavras(texto: str) -> list[str]:
    return _PADRAO.findall(texto.lower())


def ngramas(texto: str, n: int = 4) -> set[tuple[str, ...]]:
    p = _palavras(texto)
    return {tuple(p[i : i + n]) for i in range(len(p) - n + 1)}


def sobreposicao_ngramas(texto: str, corpus: str, n: int = 4) -> float:
    """Fração dos n-gramas únicos de `texto` que aparecem em `corpus` (0..1)."""
    g = ngramas(texto, n)
    if not g:
        return 0.0
    return len(g & ngramas(corpus, n)) / len(g)
