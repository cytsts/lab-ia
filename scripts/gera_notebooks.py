r"""Gera os notebooks da trilha a partir dos experimentos (spec B6).

Regra deste projeto: uma fonte de verdade só. O experimento é o arquivo .py que a CLI
roda e que os testes verificam; o notebook é gerado dele, célula a célula, sem
reescrever o conteúdo à mão. Assim não existe a duplicação que apodrece — se o
experimento muda, o notebook é regenerado (e o teste da trilha cobra isso).

Uso:
    .venv\Scripts\python scripts\gera_notebooks.py            # escreve trilha/notebooks/
    .venv\Scripts\python scripts\gera_notebooks.py --checar    # falha se estiver desatualizado

Honestidade sobre verificação: sem rede não foi possível instalar o jupyter neste
ambiente, então os notebooks NÃO foram executados aqui. O que está verificado é a
estrutura (JSON do nbformat) e a correspondência com o experimento.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
EXPERIMENTOS = RAIZ / "trilha" / "experimentos"
NOTEBOOKS = RAIZ / "trilha" / "notebooks"
LICOES = RAIZ / "trilha" / "licoes"


def _celula_markdown(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def _celula_codigo(texto: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True) if texto else [],
    }


def _licao_do_experimento(experimento: Path) -> Path | None:
    """Lição que cita este experimento (para o notebook apontar de volta)."""
    if not LICOES.is_dir():
        return None
    for arquivo in sorted(LICOES.glob("*.md")):
        if experimento.name in arquivo.read_text(encoding="utf-8"):
            return arquivo
    return None


def gerar(experimento: Path) -> dict:
    """Monta o notebook: contexto, imports, uma célula por função, e a chamada final."""
    fonte = experimento.read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    linhas = fonte.splitlines(keepends=True)
    licao = _licao_do_experimento(experimento)

    titulo = "# " + experimento.stem.replace("_", " ")
    nota = f"Lição correspondente: [{licao.name}](../licoes/{licao.name})" if licao else ""
    contexto = (
        f"{titulo}\n\n{nota}\n\n"
        "Notebook gerado a partir de `trilha/experimentos/"
        f"{experimento.name}` — a fonte de verdade é o arquivo .py; este caderno é o mesmo\n"
        "código, em células, para você mexer e rodar passo a passo.\n\n"
        "**Rode a partir da raiz do laboratório** (a pasta que tem `core/` e `data/`).\n"
        "Cada função é uma medição: rode, olhe o número, mude uma linha, rode de novo.\n"
    )
    celulas = [_celula_markdown(contexto)]

    preparo = (
        "from pathlib import Path\n"
        "import sys\n\n"
        "RAIZ = Path.cwd()\n"
        "if not (RAIZ / 'core' / 'labia').is_dir():\n"
        "    raise RuntimeError('abra o notebook na raiz do laboratorio (a pasta com core/ e data/)')\n"
        "if str(RAIZ / 'core') not in sys.path:\n"
        "    sys.path.insert(0, str(RAIZ / 'core'))\n"
    )
    celulas.append(_celula_codigo(preparo))

    trechos_import = []
    for no in arvore.body:
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            trechos_import.append("".join(linhas[no.lineno - 1 : no.end_lineno]))
    if trechos_import:
        celulas.append(_celula_codigo("\n".join(t.strip() for t in trechos_import) + "\n"))

    for no in arvore.body:
        if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef)):
            celulas.append(_celula_codigo("".join(linhas[no.lineno - 1 : no.end_lineno]) + "\n"))

    celulas.append(_celula_codigo("# roda a medição completa e imprime os números\nresultado = main()\n"))

    return {
        "cells": celulas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
            "labia": {
                "experimento": experimento.name,
                "licao": licao.name if licao else None,
                "gerado_por": "scripts/gera_notebooks.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def gerar_todos(raiz: Path = RAIZ) -> dict[Path, dict]:
    experimentos = sorted((raiz / "trilha" / "experimentos").glob("e*.py"))
    return {raiz / "trilha" / "notebooks" / f"{e.stem}.ipynb": gerar(e) for e in experimentos}


def _serializar(notebook: dict) -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gera_notebooks.py", description="Gera os notebooks da trilha")
    parser.add_argument("--checar", action="store_true", help="não escreve: falha se algum notebook estiver desatualizado")
    args = parser.parse_args(argv)

    notebooks = gerar_todos()
    if args.checar:
        desatualizados = [
            destino.name
            for destino, notebook in notebooks.items()
            if not destino.exists() or destino.read_text(encoding="utf-8") != _serializar(notebook)
        ]
        if desatualizados:
            print("notebooks desatualizados: " + ", ".join(desatualizados))
            print("rode: .venv\\Scripts\\python scripts\\gera_notebooks.py")
            return 1
        print(f"{len(notebooks)} notebooks em dia")
        return 0

    for destino, notebook in notebooks.items():
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(_serializar(notebook), encoding="utf-8")
        print(f"escrito: {destino.relative_to(RAIZ)} ({len(notebook['cells'])} células)")
    print()
    print("para abrir: instale o jupyter (pip install jupyterlab) e rode 'jupyter lab trilha/notebooks'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
