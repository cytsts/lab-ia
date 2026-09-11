"""G7/RF3 — regressão do pacote congelado.

O defeito real (medido 2026-09-11): o PyInstaller resolveu `msvcp140.dll` pelo PATH
e arrastou a cópia de 2018 do JDK 11 para a raiz do bundle. No app congelado,
`_internal` entra na busca de dependências e sombreia o CRT do sistema; `c10.dll`
exige um `msvcp140` mais novo e a inicialização falha com WinError 1114.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "verifica_pacote.py"
BUNDLE = Path(os.environ.get("LAB_IA_BUNDLE", ".qwen/pyi-dist/lab-ia")).resolve()

# A cópia antiga que causou o defeito nesta máquina; sem ela, o detector não tem fixture.
CRT_VELHA = Path(r"C:\Program Files\Microsoft\jdk-11.0.16.101-hotspot\bin\msvcp140.dll")

if os.name != "nt":
    pytest.skip("empacotamento do núcleo é verificação específica de Windows", allow_module_level=True)

_espe = importlib.util.spec_from_file_location("verifica_pacote", SCRIPT)
mod = importlib.util.module_from_spec(_espe)
_espe.loader.exec_module(mod)


def _bundle_falso(tmp_path, nome: str, com_crt: bool) -> Path:
    bundle = tmp_path / nome
    interna = bundle / "_internal"
    interna.mkdir(parents=True, exist_ok=True)
    if com_crt:
        shutil.copy2(CRT_VELHA, interna / "msvcp140.dll")
    return bundle


def test_primitiva_de_versao_le_disco():
    v = mod._versao_arquivo(mod.SYSTEM32 / "kernel32.dll")
    assert v is not None and len(v) == 4, "não consegui ler FileVersion — o detector fica cego"
    assert v[0] >= 6
    assert mod._formato(v) == ".".join(str(parte) for parte in v)


@pytest.mark.skipif(not CRT_VELHA.is_file(), reason="sem a CRT antiga de fixture nesta máquina")
def test_detector_discrimina_a_crt_sombreadora(tmp_path, capsys):
    """Com a CRT de 2018 na raiz o detector precisa acusar; sem ela, não."""
    capsys.readouterr()

    mod._falhas.clear()
    mod.c2_sombreamento_crt(_bundle_falso(tmp_path, "com-bug", com_crt=True))
    com_bug = list(mod._falhas)

    mod._falhas.clear()
    mod.c2_sombreamento_crt(_bundle_falso(tmp_path, "sem-crt", com_crt=False))
    sem_bug = list(mod._falhas)

    mod._falhas.clear()
    atual = _bundle_falso(tmp_path, "crt-atual", com_crt=False)
    shutil.copy2(mod.SYSTEM32 / "msvcp140.dll", atual / "_internal" / "msvcp140.dll")
    mod.c2_sombreamento_crt(atual)
    com_crt_boa = list(mod._falhas)

    assert len(com_bug) == 1, f"detector cego ao defeito: {com_bug}"
    assert "msvcp140.dll" in com_bug[0]
    assert "14.16" in com_bug[0], f"versão medida ausente da mensagem: {com_bug[0]}"
    assert sem_bug == [], f"falso positivo sem CRT no bundle: {sem_bug}"
    assert com_crt_boa == [], f"falso positivo com a CRT do sistema no bundle: {com_crt_boa}"


@pytest.mark.slow
@pytest.mark.skipif(not BUNDLE.is_dir(), reason=f"bundle congelado ausente: {BUNDLE}")
def test_pacote_congelado_passa_na_verificacao_completa():
    """CA2 real: exe do núcleo responde e usa a GPU de verdade."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(BUNDLE), "--cuda"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    assert r.returncode == 0, f"verificador falhou:\n{r.stdout}\n{r.stderr}"
    assert "0 falha(s)" in r.stdout
