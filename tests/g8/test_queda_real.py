"""G8: queda REAL por kill de processo + retomada íntegra (spec G8).

O subprocesso morre sem nenhuma chance de limpeza — exatamente o cenário de
queda de energia. Verifica invariantes de atomicidade e identidade do resultado.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from common import config_micro, criar_dir_run

RAIZ_REPO = Path(__file__).resolve().parents[2]


def _escrever_config(tmp: Path, corpus: Path, run: Path, passos: int) -> Path:
    import yaml

    cfg = config_micro(corpus, run, passos=passos, avaliar_a_cada=passos // 2, salvar_a_cada=10, lr=3e-3)
    caminho = tmp / f"cfg-{passos}.yaml"
    caminho.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return caminho


def _launch(config: Path, run_id: str, raiz: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(config),
         "--run-id", run_id, "--raiz", str(raiz)],
        cwd=RAIZ_REPO, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def _esperar_checkpoint(run: Path, minimo: int, timeout: float = 120.0) -> int:
    fim = time.time() + timeout
    while time.time() < fim:
        cps = sorted(
            int(p.stem.split("-")[1]) for p in (run / "ckpt").glob("passo-*.pt")
        )
        if cps and cps[-1] >= minimo:
            return cps[-1]
        # 20 ms de espera: o modelo micro termina os passos que faltam em poucas
        # dezenas de ms, então uma janela de 100 ms deixava o processo escapar vivo
        # de vez em quando e a queda deixava de ser testada
        time.sleep(0.02)
    raise AssertionError(f"checkpoint >= {minimo} não apareceu em {timeout}s")


def _ler_metricas(run: Path) -> list[dict]:
    arquivo = run / "metricas.jsonl"
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("passos_ate_kill", [20, 35])
def test_queda_e_retomada_identica(tmp_path, corpus_arquivo, passos_ate_kill):
    total = 60
    dir_cont = criar_dir_run(tmp_path / "runs" / "continuo")
    dir_qeda = criar_dir_run(tmp_path / "runs" / "queda")
    cfg_cont = _escrever_config(tmp_path, corpus_arquivo, dir_cont, total)
    cfg_qeda = _escrever_config(tmp_path, corpus_arquivo, dir_qeda, total)

    # 1) run contínuo de referência
    assert subprocess.run(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(cfg_cont),
         "--run-id", "continuo", "--raiz", str(tmp_path)],
        cwd=RAIZ_REPO, timeout=600, capture_output=True, text=True,
    ).returncode == 0

    # 2) run vítima: mata-se de verdade após o checkpoint alvo
    proc = _launch(cfg_qeda, "queda", tmp_path)
    _esperar_checkpoint(dir_qeda, passos_ate_kill)
    if proc.poll() is None:  # máquina lenta pode terminar antes; o kill segue sendo real quando vivo
        proc.kill()  # TerminateProcess no Windows / SIGKILL no POSIX — sem cleanup
        proc.wait(timeout=60)
        assert proc.returncode != 0, "processo deveria morrer por sinal"

    # RF1/CA2: estado legível, progresso incompleto; .tmp órfão é aceitável (nunca é
    # lido), mas todo passo-*.pt visível TEM de carregar inteiro.
    import torch

    estado = json.loads((dir_qeda / "estado.json").read_text(encoding="utf-8"))
    assert estado["concluido"] is False
    for p in (dir_qeda / "ckpt").glob("passo-*.pt"):
        torch.load(p, map_location="cpu", weights_only=True)  # levanta se truncado
    for p in (dir_qeda / "ckpt").glob("passo-*.pt"):
        m = __import__("re").fullmatch(r"passo-\d+\.pt", p.name)
        assert m, f"arquivo de checkpoint com nome inválido: {p.name}"

    # 3) retomada
    assert subprocess.run(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(cfg_qeda),
         "--run-id", "queda", "--raiz", str(tmp_path), "--retomar"],
        cwd=RAIZ_REPO, timeout=600, capture_output=True, text=True,
    ).returncode == 0

    reg_c = [r for r in _ler_metricas(dir_cont) if r["passo"]]
    reg_q = [r for r in _ler_metricas(dir_qeda) if r["passo"]]
    assert [r["passo"] for r in reg_q] == [r["passo"] for r in reg_c], "métricas duplicadas/puladas"
    for rc, rq in zip(reg_c, reg_q):
        assert abs(rc["loss_val"] - rq["loss_val"]) < 1e-6
    estado_final = json.loads((dir_qeda / "estado.json").read_text(encoding="utf-8"))
    assert estado_final["concluido"] is True and estado_final["passo"] == total

def test_retomada_fecha_run_que_morreu_depois_do_ultimo_checkpoint(tmp_path, corpus_arquivo):
    """Corrida real (1 falha em 6 execuções): morrer entre gravar o ÚLTIMO checkpoint e
    fechar o estado deixava o run 'não concluído' para sempre — toda retomada seguinte
    virava no-op silencioso. Reproduzido de forma determinística, sem depender de timing.
    """
    dir_run = criar_dir_run(tmp_path / "runs" / "gap")
    cfg = _escrever_config(tmp_path, corpus_arquivo, dir_run, 20)
    assert subprocess.run(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(cfg),
         "--run-id", "gap", "--raiz", str(tmp_path)],
        cwd=RAIZ_REPO, timeout=600, capture_output=True, text=True,
    ).returncode == 0

    # estado como ficaria se o processo tivesse morrido logo após o checkpoint final:
    # checkpoint 20 no disco, estado ainda apontando o fechamento anterior
    estado = json.loads((dir_run / "estado.json").read_text(encoding="utf-8"))
    assert estado["concluido"] is True
    estado["concluido"] = False
    estado["passo"] = estado["passos_totais"] - 10
    (dir_run / "estado.json").write_text(json.dumps(estado), encoding="utf-8")
    metricas_antes = [r["passo"] for r in _ler_metricas(dir_run)]

    assert subprocess.run(
        [sys.executable, "-m", "labia.cli", "train", "--config", str(cfg),
         "--run-id", "gap", "--raiz", str(tmp_path), "--retomar"],
        cwd=RAIZ_REPO, timeout=600, capture_output=True, text=True,
    ).returncode == 0

    final = json.loads((dir_run / "estado.json").read_text(encoding="utf-8"))
    assert final["concluido"] is True, "retomada no último passo tem de fechar o run"
    assert final["passo"] == 20
    assert [r["passo"] for r in _ler_metricas(dir_run)] == metricas_antes, "retomada não repete nem pula métrica"

