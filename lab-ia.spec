# -*- mode: python ; coding: utf-8 -*-
"""Empacota o nucleo headless do lab-ia (spec G7, RF3).

Build:
    .venv\\Scripts\\pyinstaller --noconfirm --distpath .qwen/pyi-dist --workpath .qwen/pyi-work lab-ia.spec
Validacao:
    .venv\\Scripts\\python scripts/verifica_pacote.py .qwen/pyi-dist/lab-ia
"""
import os

from PyInstaller.utils.hooks import collect_all

# DLLs do redistributavel MSVC que nao podem vir de qualquer lugar do PATH.
CRT_DESTINOS = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codec_convert_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
    "concrt140.dll",
    "vcomp140.dll",
}

datas, binaries, hiddenimports = [], [], []
for modulo in ("torch", "labia"):
    d, b, h = collect_all(modulo)
    datas += d
    binaries += b
    hiddenimports += h

# collect_all('nvidia') e intencionalmente ausencia aqui: no wheel Windows cu128
# as DLLs CUDA vivem em torch/lib, nao em site-packages/nvidia (medido 2026-09-11).

a = Analysis(
    [os.path.join("scripts", "entry_nucleo.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)


def _arquivo_crt_sistema(nome):
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    caminho = os.path.join(system32, nome)
    return caminho if os.path.isfile(caminho) else None


substituidas = []
rejeitadas = []
mantidas = []
for entrada in a.binaries:
    destino = os.path.basename(str(entrada[0]).replace("\\", os.sep)).lower()
    if destino not in CRT_DESTINOS:
        mantidas.append(entrada)
        continue
    origem_sistema = _arquivo_crt_sistema(destino)
    if origem_sistema and os.path.normcase(origem_sistema) != os.path.normcase(entrada[1]):
        rejeitadas.append((destino, entrada[1]))
        substituidas.append((str(entrada[0]), origem_sistema, "BINARY"))
    else:
        mantidas.append(entrada)

a.binaries = mantidas + substituidas

for destino, origem in rejeitadas:
    print(f"[lab-ia.spec] CRT descartada do bundle (veio do PATH): {origem}")
    print(f"[lab-ia.spec]   no lugar: {os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', destino)}")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lab-ia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="lab-ia",
)
