"""Tokenizador BPE pt-BR (HF tokenizers), salvo junto ao run para reprodutibilidade."""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ESPECIAIS = ["<unk>", "<fim>"]  # <fim>: marcador de fim de trecho/mostr


def treinar_tokenizer_ptbr(
    textos: list[str], vocab_tam: int, destino: Path
) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    treinador = trainers.BpeTrainer(
        vocab_size=vocab_tam,
        special_tokens=ESPECIAIS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(textos, treinador)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(destino))
    return tok


def carregar_tokenizer(caminho: Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(caminho) / "tokenizer.json"))
