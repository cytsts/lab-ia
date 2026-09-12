"""Monta a pasta pacote/ para o laboratório portátil (spec B4).

O app Electron portátil precisa levar três coisas para outra máquina:

    pacote/nucleo/    núcleo congelado (PyInstaller) — o Python do laboratório
    pacote/semente/   configs, docs e data iniciais — copiados para o workspace do
                      usuário na primeira execução (nunca sobrescrevem nada)
    (a SPA e o shell vão dentro do próprio pacote do electron-builder)

Uso:
    .venv\\Scripts\\python scripts/monta_portatil.py [--nucleo .qwen/pyi-dist/lab-ia] [--limpar]

O tamanho importa: com CUDA o núcleo passa de 4 GB (só o torch leva a maior parte).
O script mede e imprime isso em vez de deixar a surpresa para o dia do build.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTA_PACOTE = RAIZ / "pacote"
SEMENTE_DIRS = ("configs", "docs", "data")


def _tamanho(caminho: Path) -> int:
    if caminho.is_file():
        return caminho.stat().st_size
    return sum(p.stat().st_size for p in caminho.rglob("*") if p.is_file())


def _mb(bytes_: int) -> str:
    """Tamanho legível em pt-BR; acima de 1 GB mostra GB (senão '4.957 MB' engana)."""
    megas = bytes_ / 1024 / 1024
    if megas >= 1000:
        return f"{megas / 1024:.2f} GB".replace(".", ",")
    return f"{megas:,.0f} MB".replace(",", ".")


def montar_nucleo(origem: Path, destino: Path, limpar: bool) -> int:
    exe = origem / ("lab-ia.exe" if sys.platform == "win32" else "lab-ia")
    if not exe.is_file():
        raise SystemExit(
            f"núcleo congelado não encontrado em {exe}\n"
            "construa antes:  .venv\\Scripts\\pyinstaller --noconfirm "
            "--distpath .qwen\\pyi-dist --workpath .qwen\\pyi-work lab-ia.spec"
        )
    if destino.exists() and limpar:
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copytree(origem, destino, dirs_exist_ok=True)
    return _tamanho(destino)


def montar_semente(destino: Path, limpar: bool) -> dict[str, int]:
    """Copia configs/, docs/ e os arquivos soltos de data/ (sem os datasets derivados)."""
    if destino.exists() and limpar:
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)
    tamanhos: dict[str, int] = {}
    for pasta in SEMENTE_DIRS:
        origem = RAIZ / pasta
        if not origem.is_dir():
            continue
        alvo = destino / pasta
        alvo.mkdir(parents=True, exist_ok=True)
        for item in sorted(origem.iterdir()):
            # datasets preparados (data/<id>/) são derivados: ficam fora do pacote
            if item.is_dir():
                continue
            shutil.copy2(item, alvo / item.name)
        tamanhos[pasta] = _tamanho(alvo)
    return tamanhos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monta_portatil.py", description="Monta pacote/ para o app portátil")
    parser.add_argument("--nucleo", default=".qwen/pyi-dist/lab-ia", help="pasta do núcleo congelado")
    parser.add_argument("--limpar", action="store_true", help="apaga pacote/nucleo e pacote/semente antes")
    args = parser.parse_args(argv)

    origem = (RAIZ / args.nucleo).resolve()
    print(f"[1/2] núcleo: {origem}")
    tamanho_nucleo = montar_nucleo(origem, PASTA_PACOTE / "nucleo", args.limpar)
    print(f"      copiado: {_mb(tamanho_nucleo)}")

    print("[2/2] semente do workspace")
    tamanhos = montar_semente(PASTA_PACOTE / "semente", args.limpar)
    for pasta, tamanho in tamanhos.items():
        print(f"      {pasta}: {_mb(tamanho)}")

    total = tamanho_nucleo + sum(tamanhos.values())
    print()
    print(f"pacote/ pronto: {_mb(total)} no total ({_mb(tamanho_nucleo)} são o núcleo).")
    print("O grosso é o torch com CUDA: uma máquina sem GPU pode usar um núcleo CPU (~10x menor).")
    print()
    print("próximo: pnpm --dir ui package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
