"""Baixa corpus pt-BR de domínio público (Project Gutenberg) para data/.

Uso: python scripts/prepara_corpus.py
Vende o resultado em data/corpus_ptbr.txt + data/corpus_manifest.json.
Licença dos textos: domínio público (autores mortos há >70 anos; Machado de Assis).
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "data"

# IDs verificados via Gutendex (2026-09-11): título + idioma "pt".
BUSCAS = [
    "55752",  # Dom Casmurro — Machado de Assis (pt)
    "69187",  # O Cortiço — Aluísio Azevedo (pt)
]

def baixar(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "lab-ia corpus prep"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

def limpar(texto: str) -> str:
    # Remove cabeçalho/rodapé do Gutenberg entre marcos "*** START/END".
    m = re.search(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG.*?\*\*\*(.*?)\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG", texto, re.S)
    if m:
        texto = m.group(1)
    texto = re.sub(r"\r\n?", "\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

def main() -> int:
    DADOS.mkdir(exist_ok=True)
    partes: list[str] = []
    obras: list[dict] = []
    for ebook_id in BUSCAS:
        meta_url = f"https://gutendex.com/books/{ebook_id}"
        meta = json.loads(baixar(meta_url))
        if "pt" not in meta["languages"]:
            raise RuntimeError(f"ebook {ebook_id} não é pt: {meta['languages']} ({meta['title']})")
        url = meta["formats"]["text/plain; charset=utf-8"]
        bruto = baixar(url).decode("utf-8", errors="replace")
        limpo = limpar(bruto)
        partes.append(limpo)
        obras.append({
            "titulo": meta["title"],
            "autores": [a["name"] for a in meta["authors"]],
            "id_gutenberg": ebook_id,
            "caracteres": len(limpo),
        })
        print(f"[ok] {meta['title']}: {len(limpo):,} chars")
    destino = DADOS / "corpus_ptbr.txt"
    destino.write_text("\n\n".join(partes), encoding="utf-8")
    manifesto = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "licenca": "domínio público (Project Gutenberg)",
        "obras": obras,
        "total_caracteres": destino.stat().st_size,
    }
    (DADOS / "corpus_manifest.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] corpus -> {destino} ({destino.stat().st_size:,} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
