"""Contrato do espaço autoral da B14."""
from __future__ import annotations

import json
import sys

import pytest

from labia.laboratorio import NOME_PRIMEIRO_CADERNO, comando_jupyter, preparar


def test_preparar_cria_exemplo_editavel_sem_sobrescrever(tmp_path):
    primeiro = preparar(tmp_path)
    caminho = tmp_path / "cadernos" / "exemplos" / NOME_PRIMEIRO_CADERNO
    assert primeiro["exemplo_criado"] is True
    caderno = json.loads(caminho.read_text(encoding="utf-8"))
    assert caderno["nbformat"] == 4
    fontes = "\n".join("".join(celula["source"]) for celula in caderno["cells"])
    assert "import labia" in fontes
    assert "hipotese" in fontes
    assert "ConfigGPT" in fontes
    assert "torch.save" in fontes

    caminho.write_text("conteúdo autoral", encoding="utf-8")
    segundo = preparar(tmp_path)
    assert segundo["exemplo_criado"] is False
    assert caminho.read_text(encoding="utf-8") == "conteúdo autoral"


def test_comando_jupyter_usa_loopback_raiz_e_token(tmp_path):
    args = comando_jupyter(tmp_path, porta=8899, token="segredo")
    assert args[:3] == [sys.executable, "-m", "jupyterlab"]
    assert f"--ServerApp.root_dir={tmp_path.resolve()}" in args
    assert "--ServerApp.ip=127.0.0.1" in args
    assert "--ServerApp.port=8899" in args
    assert "--ServerApp.token=segredo" in args


def test_caderno_executa_estado_gpt_e_checkpoint(tmp_path):
    """O aceite L0/L2/L3 usa um kernel real, não um simulador de células."""
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    info = preparar(tmp_path)
    caderno = nbformat.read(info["exemplo"], as_version=4)
    NotebookClient(
        caderno,
        kernel_name="python3",
        timeout=120,
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    saidas = "\n".join(
        saida.get("text", "")
        for celula in caderno.cells
        if celula.cell_type == "code"
        for saida in celula.get("outputs", [])
        if saida.output_type == "stream"
    )
    assert "Lab-IA importado de:" in saidas
    assert "parâmetros" in saidas
    assert (tmp_path / "cadernos" / "resultados" / "primeiro-experimento.pt").exists()


def test_caderno_salva_na_raiz_quando_kernel_abre_na_pasta_do_caderno(tmp_path):
    """JupyterLab real executa o kernel com cwd na pasta do caderno.

    Regressão: com `raiz = Path.cwd()` o checkpoint era salvo em
    `cadernos/exemplos/cadernos/resultados/`, aninhado e fora do lugar.
    """
    from pathlib import Path

    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    info = preparar(tmp_path)
    pasta_do_caderno = Path(info["exemplo"]).parent
    caderno = nbformat.read(info["exemplo"], as_version=4)
    NotebookClient(
        caderno,
        kernel_name="python3",
        timeout=120,
        resources={"metadata": {"path": str(pasta_do_caderno)}},
    ).execute()
    assert (tmp_path / "cadernos" / "resultados" / "primeiro-experimento.pt").exists()
    assert not (pasta_do_caderno / "cadernos").exists()
