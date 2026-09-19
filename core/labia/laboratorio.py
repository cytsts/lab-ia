"""Espaço de trabalho para experimentos autorais em JupyterLab.

O núcleo do Lab-IA continua sendo uma biblioteca Python. Este módulo só prepara
um lugar seguro para os cadernos do usuário e fornece o comando que abre o
JupyterLab; não tenta transformar cada experimento em um formulário da API.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

PASTA_CADERNOS = "cadernos"
PASTA_EXEMPLOS = "exemplos"
PASTA_RESULTADOS = "resultados"
NOME_PRIMEIRO_CADERNO = "00-primeiro-experimento.ipynb"


def _celula(tipo: str, texto: str) -> dict:
    """Cria uma célula nbformat simples sem exigir Jupyter para preparar arquivos."""
    return {
        "id": hashlib.sha1(f"{tipo}\0{texto}".encode()).hexdigest()[:8],
        "cell_type": tipo,
        "metadata": {},
        "source": [linha + "\n" for linha in texto.splitlines()],
        **({"execution_count": None, "outputs": []} if tipo == "code" else {}),
    }


def caderno_inicial() -> dict:
    """Exemplo copiável que prova estado compartilhado e acesso ao núcleo."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python (Lab-IA)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "cells": [
            _celula(
                "markdown",
                "# Meu primeiro experimento\n\n"
                "Este arquivo é seu. Altere, adicione ou apague células livremente. "
                "Salve checkpoints e resultados em `cadernos/resultados/`.",
            ),
            _celula(
                "code",
                "from pathlib import Path\n"
                "import sys\n"
                "import torch\n"
                "import labia\n\n"
                "# O kernel abre com cwd na pasta do caderno; a raiz do laboratório\n"
                "# é o primeiro ancestral que contém a pasta 'cadernos'.\n"
                "pasta_atual = Path.cwd().resolve()\n"
                "raiz = next((p for p in (pasta_atual, *pasta_atual.parents) if (p / 'cadernos').is_dir()), pasta_atual)\n"
                "resultados = raiz / 'cadernos' / 'resultados'\n"
                "resultados.mkdir(parents=True, exist_ok=True)\n"
                "print(f'Lab-IA importado de: {Path(labia.__file__).resolve()}')\n"
                "print(f'PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}')",
            ),
            _celula(
                "code",
                "# Esta variável continua viva para as próximas células desta sessão.\n"
                "hipotese = 'Uma perda quadrática diminui neste exemplo mínimo.'\n"
                "amostras = torch.tensor([1.0, 2.0, 3.0])\n"
                "alvo = torch.tensor([2.0, 4.0, 6.0])\n"
                "print(hipotese)",
            ),
            _celula(
                "code",
                "# Mude a perda, o modelo ou os dados e execute novamente.\n"
                "peso = torch.nn.Parameter(torch.tensor(0.0))\n"
                "otimizador = torch.optim.SGD([peso], lr=0.1)\n"
                "for passo in range(40):\n"
                "    perda = torch.mean((peso * amostras - alvo) ** 2)\n"
                "    otimizador.zero_grad()\n"
                "    perda.backward()\n"
                "    otimizador.step()\n"
                "print({'peso': float(peso.detach()), 'perda': float(perda.detach())})",
            ),
            _celula(
                "code",
                "# O núcleo do Lab-IA também está disponível como biblioteca.\n"
                "from labia.models.gpt import ConfigGPT, GPT\n\n"
                "torch.manual_seed(7)\n"
                "config = ConfigGPT(vocab=32, dim=16, camadas=1, cabecas=2, janela_ctx=8, abandono=0.0)\n"
                "modelo = GPT(config)\n"
                "modelo.init_pesos(semente=7)\n"
                "tokens = torch.randint(0, config.vocab, (2, 8))  # troque pelos seus tokens ou dados\n"
                "logits, perda_gpt = modelo(tokens, tokens)\n"
                "print({'parâmetros': modelo.contar_parametros(), 'perda': float(perda_gpt.detach())})",
            ),
            _celula(
                "code",
                "# Salvar artefatos é explícito; reiniciar o kernel limpa só a memória.\n"
                "checkpoint = resultados / 'primeiro-experimento.pt'\n"
                "torch.save({'peso': peso.detach(), 'modelo': modelo.state_dict(), 'config': config.para_dict(), 'hipotese': hipotese}, checkpoint)\n"
                "restaurado = torch.load(checkpoint, weights_only=True)\n"
                "print(f'Checkpoint salvo e relido: {checkpoint}')\n"
                "restaurado",
            ),
        ],
    }


def preparar(raiz: str | Path) -> dict[str, str | bool]:
    """Cria o espaço autoral sem sobrescrever nenhum arquivo do usuário."""
    raiz = Path(raiz).resolve()
    cadernos = raiz / PASTA_CADERNOS
    exemplos = cadernos / PASTA_EXEMPLOS
    resultados = cadernos / PASTA_RESULTADOS
    exemplos.mkdir(parents=True, exist_ok=True)
    resultados.mkdir(parents=True, exist_ok=True)
    inicial = exemplos / NOME_PRIMEIRO_CADERNO
    criado = False
    if not inicial.exists():
        inicial.write_text(json.dumps(caderno_inicial(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        criado = True
    return {
        "raiz": str(raiz),
        "cadernos": str(cadernos),
        "exemplo": str(inicial),
        "exemplo_criado": criado,
        "resultados": str(resultados),
    }


def comando_jupyter(raiz: str | Path, porta: int = 8889, token: str | None = None) -> list[str]:
    """Argumentos para um JupyterLab local, restrito ao loopback."""
    raiz = Path(raiz).resolve()
    args = [
        sys.executable,
        "-m",
        "jupyterlab",
        f"--ServerApp.root_dir={raiz}",
        "--ServerApp.ip=127.0.0.1",
        f"--ServerApp.port={porta}",
        "--ServerApp.port_retries=0",
        "--ServerApp.open_browser=False",
    ]
    if token:
        args.append(f"--ServerApp.token={token}")
    return args


def abrir(raiz: str | Path, porta: int = 8889) -> int:
    """Prepara o espaço e entrega o processo ao JupyterLab no terminal atual."""
    preparar(raiz)
    try:
        import jupyterlab  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError("instale o suporte com: pip install -e \"core[laboratorio]\"") from e
    return subprocess.call(comando_jupyter(raiz, porta), cwd=Path(raiz))
