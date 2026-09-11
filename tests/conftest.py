"""Fixtures compartilhadas dos testes da G1: corpus sintético pt-BR."""
from __future__ import annotations

from pathlib import Path

import pytest

from common import gerar_corpus


@pytest.fixture(scope="session")
def corpus_arquivo(tmp_path_factory) -> Path:
    destino = tmp_path_factory.mktemp("dados") / "corpus_micro.txt"
    destino.write_text(gerar_corpus(), encoding="utf-8")
    return destino
