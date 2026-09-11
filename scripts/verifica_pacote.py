"""Verifica o pacote do núcleo congelado pelo PyInstaller (spec G7, RF3/CA2).

Não confia em status auto-declarado: cada checagem observa um fato externo.

Uso:
    .venv\\Scripts\\python scripts/verifica_pacote.py [.qwen/pyi-dist/lab-ia] [--cuda]

Checagens:
    C1  o executável do bundle existe
    C2  nenhuma DLL do redistributável MSVC na raiz do bundle é mais antiga que a
        do sistema — é o sombreamento de CRT que derruba c10.dll com WinError 1114
    C3  `lab-ia.exe agentes` responde sem traceback
    C4  (--cuda) uma geração real roda e o PID do exe aparece como processo CUDA
        em `nvidia-smi`, provando uso efetivo da GPU pelo binário congelado
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

NOMES_CRT = {
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

SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"

# O console pode ser cp1252; a saída do exe congelado tem acentos pt-BR.
for _fluxo in (sys.stdout, sys.stderr):
    if hasattr(_fluxo, "reconfigure"):
        _fluxo.reconfigure(encoding="utf-8", errors="replace")

_falhas: list[str] = []


def _ok(mensagem: str) -> None:
    print(f"  OK   {mensagem}")


def _nao(mensagem: str) -> None:
    _falhas.append(mensagem)
    print(f"  FALHA {mensagem}")


def _hi(word: int) -> int:
    return (word >> 16) & 0xFFFF


def _lo(word: int) -> int:
    return word & 0xFFFF


class _FIXINFO(ctypes.Structure):
    _fields_ = [
        ("dwSignature", wintypes.DWORD),
        ("dwStrucVersion", wintypes.DWORD),
        ("dwFileVersionMS", wintypes.DWORD),
        ("dwFileVersionLS", wintypes.DWORD),
    ]


def _versao_arquivo(caminho: Path) -> tuple[int, ...] | None:
    """FileVersion via version.dll; None se o arquivo não trouxer o recurso."""
    size = ctypes.windll.version.GetFileVersionInfoSizeW
    get = ctypes.windll.version.GetFileVersionInfoW
    query = ctypes.windll.version.VerQueryValueW

    total = size(str(caminho), None)
    if not total:
        return None
    buffer = ctypes.create_string_buffer(total)
    if not get(str(caminho), 0, total, buffer):
        return None
    ponteiro = ctypes.c_void_p()
    comprimento = ctypes.c_uint()
    if not query(buffer, "\\", ctypes.byref(ponteiro), ctypes.byref(comprimento)):
        return None
    info = ctypes.cast(ponteiro, ctypes.POINTER(_FIXINFO)).contents
    return (_hi(info.dwFileVersionMS), _lo(info.dwFileVersionMS),
            _hi(info.dwFileVersionLS), _lo(info.dwFileVersionLS))


def _formato(v: tuple[int, ...] | None) -> str:
    return ".".join(str(parte) for parte in v) if v else "sem versão"


def _executar(exe: Path, *argumentos: str, timeout: int = 600) -> tuple[int, str, str]:
    proc = subprocess.run(
        [str(exe), *argumentos], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _tem_traceback(texto: str) -> bool:
    return "Traceback (most recent call last)" in texto or "WinError 1114" in texto


def _cauda(texto: str, n: int = 3) -> str:
    return " | ".join(texto.strip().splitlines()[-n:])


def c1_executavel(bundle: Path) -> Path:
    exe = bundle / "lab-ia.exe"
    if exe.is_file():
        _ok(f"C1 {exe.name} existe ({exe.stat().st_size / 1024 / 1024:.1f} MiB)")
    else:
        _nao(f"C1 executável ausente em {exe}")
    return exe


def c2_sombreamento_crt(bundle: Path) -> None:
    raiz = bundle / "_internal"
    if not raiz.is_dir():
        raiz = bundle
    crt_no_bundle = sorted(p for p in raiz.glob("*.dll") if p.name.lower() in NOMES_CRT)
    if not crt_no_bundle:
        _ok("C2 nenhuma CRT do redistributável na raiz do bundle (o exe resolve pela do sistema)")
        return
    atrasadas = []
    for p in crt_no_bundle:
        do_sistema = SYSTEM32 / p.name
        v_bundle = _versao_arquivo(p)
        v_sistema = _versao_arquivo(do_sistema) if do_sistema.is_file() else None
        if v_sistema is not None and v_bundle is not None and v_bundle < v_sistema:
            atrasadas.append(f"{p.name} bundle={_formato(v_bundle)} < sistema={_formato(v_sistema)}")
        else:
            _ok(f"C2 {p.name} {_formato(v_bundle)} (sistema: {_formato(v_sistema)})")
    if atrasadas:
        _nao("C2 CRT do bundle mais antiga que a do sistema sombreia dependências de c10.dll -> "
             + "; ".join(atrasadas))


def c3_agentes(exe: Path) -> None:
    rc, stdout, stderr = _executar(exe, "agentes", timeout=180)
    if rc != 0 or _tem_traceback(stderr) or _tem_traceback(stdout):
        _nao(f"C3 `agentes` rc={rc}: {_cauda(stderr or stdout)}")
        return
    _ok(f"C3 `agentes` respondeu (rc=0, {len(stdout.splitlines())} linhas): {_cauda(stdout, 1)}")


def _pids_cuda() -> set[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if r.returncode != 0:
        return set()
    return {int(linha) for linha in (l.strip() for l in r.stdout.splitlines()) if linha.isdigit()}


def c4_uso_cuda(exe: Path) -> None:
    argumentos = [
        "agente", "avaliador", "evaluate_model", "--json",
        '{"run":"g1-treino-zero","prompt":"Uma noite","passos_max":400}',
    ]
    vistos: set[int] = set()
    encerrado = threading.Event()

    def vigiar() -> None:
        while not encerrado.is_set():
            vistos.update(_pids_cuda())
            time.sleep(0.2)

    vigia = threading.Thread(target=vigiar, daemon=True)
    vigia.start()
    try:
        proc = subprocess.Popen(
            [str(exe), *argumentos], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            stdout, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            _nao("C4 geração não terminou em 600s")
            return
    finally:
        encerrado.set()
        vigia.join(timeout=10)

    if proc.returncode != 0 or _tem_traceback(stderr) or _tem_traceback(stdout):
        _nao(f"C4 geração rc={proc.returncode}: {_cauda(stderr or stdout)}")
        return
    if proc.pid in vistos:
        _ok(f"C4 PID {proc.pid} apareceu como processo CUDA no nvidia-smi durante a geração")
    else:
        _nao(f"C4 o exe gerou texto mas nunca ocupou a GPU (PID {proc.pid}; "
             f"observados: {sorted(vistos) or 'nenhum'})")


def main(argv: list[str]) -> int:
    pos = [a for a in argv if not a.startswith("--")]
    bundle = Path(pos[0]).resolve() if pos else Path(".qwen/pyi-dist/lab-ia").resolve()
    print(f"=== verificando pacote: {bundle} ===")
    if not bundle.is_dir():
        print(f"ERRO: diretório do bundle não existe: {bundle}")
        return 2

    exe = c1_executavel(bundle)
    c2_sombreamento_crt(bundle)
    c3_agentes(exe)
    if "--cuda" in argv:
        c4_uso_cuda(exe)

    print(f"\n=== resultado: {len(_falhas)} falha(s) ===")
    for falha in _falhas:
        print(f"  - {falha}")
    return 1 if _falhas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
