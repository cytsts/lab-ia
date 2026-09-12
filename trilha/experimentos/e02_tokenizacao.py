r"""Lição 2 — tokenização BPE: por que o modelo não vê letras nem palavras.

Experimento: treina BPE de verdade (a mesma função que o treino usa) e mede
caracteres por token para vários tamanhos de vocabulário.

Rode:  .venv\Scripts\python trilha\experimentos\e02_tokenizacao.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from labia.trainer.tokenizacao import carregar_tokenizer, treinar_tokenizer_ptbr

FRASE = "A noite estava fria e chuvosa quando ele decidiu partir."
CORPUS_PADRAO = Path("data/corpus_ptbr.txt")
SINTETICO = (
    "A noite estava fria e chuvosa quando ele decidiu partir.\n\n"
    "O mar batia com força contra as pedras do quebra-mar.\n\n"
    "Ela sorriu sem dizer nada e guardou a carta na gaveta.\n\n"
) * 40


def texto_de_trabalho(corpus: Path | str | None = None) -> tuple[str, str]:
    """Usa o corpus do laboratório se existir; senão, um texto sintético repetido."""
    caminho = Path(corpus) if corpus else CORPUS_PADRAO
    if caminho.exists():
        return caminho.read_text(encoding="utf-8")[:200_000], str(caminho)
    return SINTETICO, "sintético (data/corpus_ptbr.txt não encontrado)"


def comparar_vocabularios(corpus: str, tamanhos=(128, 512, 4096)) -> list[dict]:
    """Para cada tamanho de vocabulário: caracteres/token e quantos tokens a frase gasta."""
    resultados = []
    with tempfile.TemporaryDirectory() as pasta:
        for tamanho in tamanhos:
            destino = Path(pasta) / f"tok{tamanho}"
            destino.mkdir(parents=True, exist_ok=True)
            treinar_tokenizer_ptbr([corpus], tamanho, destino / "tokenizer.json")
            tokenizer = carregar_tokenizer(destino)
            vocab_real = tokenizer.get_vocab_size()
            ids = tokenizer.encode(corpus, add_special_tokens=False).ids
            ids_frase = tokenizer.encode(FRASE, add_special_tokens=False).ids
            pedacos = tokenizer.encode(FRASE, add_special_tokens=False).tokens
            resultados.append(
                {
                    "vocab_pedido": tamanho,
                    "vocab_real": vocab_real,
                    "chars_por_token": round(len(corpus) / max(1, len(ids)), 2),
                    "tokens_da_frase": len(ids_frase),
                    "pedacos_da_frase": pedacos,
                }
            )
    return resultados


def main(corpus: str | None = None, tamanhos=(128, 512, 4096)) -> dict:
    texto, origem = texto_de_trabalho(corpus)
    tabela = comparar_vocabularios(texto, tamanhos)
    print(f"corpus: {origem} ({len(texto)} caracteres)")
    print()
    print(f"{'vocab':>7} {'real':>7} {'chars/token':>12} {'tokens da frase':>16}")
    for linha in tabela:
        print(
            f"{linha['vocab_pedido']:>7} {linha['vocab_real']:>7} "
            f"{linha['chars_por_token']:>12} {linha['tokens_da_frase']:>16}"
        )
    print()
    print(f"frase: {FRASE}")
    print(f"tokenizada com o maior vocabulário: {tabela[-1]['pedacos_da_frase']}")
    print()
    print("Leia assim: token é a unidade que o modelo realmente vê. Com 128 peças o")
    print("texto sai picado em muitos tokens; com 4096 cada token cobre mais caracteres.")
    print("Vocabulário maior encurta a sequência, mas engorda o embedding (vocab x dim)")
    print("e o custo da camada final — é troca, não ganho de graça.")
    return {"origem": origem, "tabela": tabela, "frase": FRASE}


if __name__ == "__main__":
    main()
