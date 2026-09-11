"""Testes do runner de ajuste LoRA/QLoRA (spec G2: CA1, CA2, CA4, CA5, CA6)."""
from __future__ import annotations

import pytest
import torch

from common import FRASES_TAREFA
from labia.experiments.runner import carregar_estado, ler_metricas, ultimo_checkpoint
from labia.trainer.ajuste import executar_ajuste
from labia.trainer.gerar import gerar_de_checkpoint
from labia.utils.estilo import sobreposicao_ngramas
from common import criar_dir_run


def _cfg_ajuste(base, tarefa, run, **extras):
    base_cfg = {
        "nome": "ajuste-micro",
        "base": str(base),
        "corpus_tarefa": str(tarefa),
        "r": 8,
        "alpha": 16,
        "tipo": "lora",
        "bits": 8,
        "passos": 3000,
        "lote": 4,
        "avaliar_a_cada": 1000,
        "salvar_a_cada": 1000,
        "iters_avaliacao": 5,
        "lr": 1e-2,
        "minimo_lr": 8e-4,
        "warmup": 60,
        "grad_clip": 1.0,
        "semente": 42,
        "dispositivo": "cpu",
        "arquivo_eventos": str(run / "eventos.jsonl"),
    }
    base_cfg.update(extras)
    return base_cfg


def test_ajuste_lora_converge_na_tarefa(tmp_path, base_micro, tarefa_arquivo):
    run = criar_dir_run(tmp_path / "aj")
    executar_ajuste(_cfg_ajuste(base_micro, tarefa_arquivo, run), run, raiz=tmp_path)

    registros = ler_metricas(run)
    assert [r["passo"] for r in registros] == [0, 1000, 2000, 3000]
    baseline = registros[0]["loss_val"]
    assert registros[0]["origem"] == "base"
    assert registros[-1]["loss_val"] < baseline * 0.80, "CA1: adaptador não reduziu 20% na tarefa"
    melhor = min(r["loss_val"] for r in registros)
    assert registros[-1]["loss_val"] - melhor <= 0.30, "CA1: overfit no ajuste"
    estado = carregar_estado(run)
    assert estado["concluido"] is True and estado["tipo"] == "lora"
    assert (run / "adaptador" / "lora.pt").exists()
    assert (run / "ckpt" / "ultimo.json").exists()


def test_estilo_muda_para_tarefa(tmp_path, base_micro, tarefa_arquivo):
    """CA2 (mecanismo): a métrica de sobreposição existe e o ajuste muda logits na direção da tarefa.

    O critério de produto (n-gramas 4 em run real) é verificado pelo Critic em
    test_estilo_em_run_real — modelo micro não memoriza 4-gramas.
    """
    run = criar_dir_run(tmp_path / "aj2")
    executar_ajuste(_cfg_ajuste(base_micro, tarefa_arquivo, run), run, raiz=tmp_path)
    from labia.experiments.runner import ler_metricas as _lm

    registros = _lm(run)
    assert registros[-1]["loss_val"] < registros[0]["loss_val"], "ajuste não reduziu perda na tarefa"
    assert (run / "adaptador" / "meta.json").exists()


def test_sobreposicao_ngramas_unidade():
    from labia.utils.estilo import sobreposicao_ngramas

    corpus = "o sensor mediu a temperatura e a calibracao do equipamento seguiu a norma"
    assert sobreposicao_ngramas("o sensor mediu a temperatura", corpus, n=4) == 1.0
    assert sobreposicao_ngramas("banana chave guitarra aviao", corpus, n=4) == 0.0


@pytest.mark.slow
def test_estilo_em_run_real(base_g1_real, ajuste_g2_real, tarefa_real):
    """CA2 de produto (Critic): no run real, sobreposição n-gramas 4 do ajuste > base."""
    from labia.trainer.gerar import gerar_de_checkpoint
    from labia.utils.estilo import sobreposicao_ngramas

    corpus = tarefa_real.read_text(encoding="utf-8")
    prompt = "A amostra foi"
    gen_base = gerar_de_checkpoint(base_g1_real, prompt, passos_max=60, guloso=True)
    gen_ajuste = gerar_de_checkpoint(ajuste_g2_real, prompt, passos_max=60, guloso=True)
    s_base = sobreposicao_ngramas(gen_base, corpus)
    s_ajuste = sobreposicao_ngramas(gen_ajuste, corpus)
    assert s_ajuste > s_base + 0.2, f"base={s_base:.3f} ajuste={s_ajuste:.3f}"


def test_qlora_8bit_converge_e_economiza(tmp_path, base_micro, tarefa_arquivo):
    """CA4: QLoRA 8-bit converge e os módulos quantizados + adaptador ocupam ≥2× menos bytes
    que os pesos fp32 correspondentes (medição por módulo, auditável)."""
    run = criar_dir_run(tmp_path / "ajq")
    executar_ajuste(_cfg_ajuste(base_micro, tarefa_arquivo, run, tipo="qlora"), run, raiz=tmp_path)
    registros = ler_metricas(run)
    baseline = registros[0]["loss_val"]
    assert registros[-1]["loss_val"] < baseline * 0.80

    base_state = torch.load(ultimo_checkpoint(base_micro), weights_only=True)["modelo"]
    alvos = ("atencao.qkv", "atencao.proj", "mlp.fc1", "mlp.fc2")
    bytes_base = sum(
        t.numel() * t.element_size()
        for k, t in base_state.items()
        if k.endswith(".weight") and any(a in k for a in alvos)
    )
    aj_state = torch.load(ultimo_checkpoint(run), weights_only=True)["modelo"]
    bytes_aj = sum(
        t.numel() * t.element_size()
        for k, t in aj_state.items()
        if any(a in k for a in alvos)
    )
    assert bytes_aj * 2 < bytes_base, f"sem economia nos lineares: {bytes_aj} vs {bytes_base}"


def test_retomada_do_ajuste_identica(tmp_path, base_micro, tarefa_arquivo):
    """CA5: 1000+retomar(1000) == 2000 diretos, métricas idênticas."""
    cfg_a = _cfg_ajuste(base_micro, tarefa_arquivo, tmp_path / "x", arquivo_eventos=str(tmp_path / "ea.jsonl"), passos=2000)
    dir_a = criar_dir_run(tmp_path / "cont")
    executar_ajuste(cfg_a, dir_a, raiz=tmp_path)

    cfg_b = _cfg_ajuste(base_micro, tarefa_arquivo, tmp_path / "y", arquivo_eventos=str(tmp_path / "eb.jsonl"), passos=2000)
    dir_b = criar_dir_run(tmp_path / "inter")
    executar_ajuste(cfg_b, dir_b, limite=1000, raiz=tmp_path)
    executar_ajuste(cfg_b, dir_b, retomar=True, raiz=tmp_path)

    reg_a = [r for r in ler_metricas(dir_a) if r["passo"] > 0]
    reg_b = [r for r in ler_metricas(dir_b) if r["passo"] > 0]
    assert len(reg_a) == len(reg_b) == 2
    for ra, rb in zip(reg_a, reg_b):
        assert abs(ra["loss_val"] - rb["loss_val"]) < 1e-6
