"""Runner de fine-tuning LoRA/QLoRA sobre um run-base (spec G2, RF6)."""
from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from ..experiments.runner import (
    carregar_estado,
    registrar_metrica,
    salvar_checkpoint,
    salvar_estado,
    ultimo_checkpoint,
)
from ..models.gpt import ConfigGPT, GPT
from ..models.lora import aplicar_lora, salvar_adaptador
from ..utils.estado import LogEventos
from .dados import dividir_corpus, lote_trem, montar_dataset
from .tokenizacao import carregar_tokenizer
from .treino import _perda_media, escolher_dispositivo, fator_lr


@dataclass
class ConfigAjuste:
    nome: str = "g2-ajuste-estilo"
    base: str = "runs/g1-treino-zero"
    corpus_tarefa: str = "data/tarefa_ciencia.txt"
    tipo: str = "lora"  # lora | qlora
    bits: int = 8  # p/ qlora: 8 (int8) ou 4 (nf4)
    r: int = 8
    alpha: float = 16.0
    passos: int = 500
    lote: int = 32
    stride: int = 0
    avaliar_a_cada: int = 100
    salvar_a_cada: int = 100
    iters_avaliacao: int = 40
    lr: float = 3e-3
    minimo_lr: float = 3e-4
    warmup: int = 20
    grad_clip: float = 1.0
    semente: int = 42
    dispositivo: str = "auto"
    arquivo_eventos: str = ".lab-ia/eventos.jsonl"

    @classmethod
    def de_arquivo(cls, caminho: Path | str) -> "ConfigAjuste":
        dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {}
        nomes = {f.name for f in dataclasses.fields(cls)}
        desconhecidas = set(dados) - nomes
        if desconhecidas:
            raise ValueError(f"chaves desconhecidas na config de ajuste: {sorted(desconhecidas)}")
        return cls(**dados)


def executar_ajuste(
    config: ConfigAjuste | dict,
    run_dir: Path,
    retomar: bool = False,
    limite: int | None = None,
    raiz: Path | str = ".",
) -> Path:
    cfg = config if isinstance(config, ConfigAjuste) else ConfigAjuste(**dict(config))
    run_dir = Path(run_dir)
    random.seed(cfg.semente)
    torch.manual_seed(cfg.semente)
    dispositivo = escolher_dispositivo(cfg.dispositivo)
    eventos = LogEventos(
        Path(cfg.arquivo_eventos)
        if Path(cfg.arquivo_eventos).is_absolute()
        else Path(raiz) / cfg.arquivo_eventos
    )

    base_dir = Path(cfg.base) if Path(cfg.base).is_absolute() else Path(raiz) / cfg.base
    base_ckpt = ultimo_checkpoint(base_dir)
    if base_ckpt is None:
        raise RuntimeError(f"nenhum checkpoint na base {base_dir} — rode o treino G1 antes")
    peso_base = torch.load(base_ckpt, map_location=dispositivo, weights_only=True)
    cfgm = ConfigGPT.de_dict(peso_base["config_modelo"])
    tokenizer = carregar_tokenizer(base_dir / "tokens")

    modelo = GPT(cfgm)
    modelo.load_state_dict(peso_base["modelo"], strict=True)
    quant = None
    if cfg.tipo == "qlora":
        quant = "int8" if cfg.bits == 8 else "nf4"
    elif cfg.tipo != "lora":
        raise ValueError(f"tipo desconhecido: {cfg.tipo!r} (use lora|qlora)")
    stats = aplicar_lora(modelo, r=cfg.r, alpha=cfg.alpha, quant=quant)
    modelo.to(dispositivo)

    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otimizador = torch.optim.AdamW(treinaveis, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-8)

    texto = Path(cfg.corpus_tarefa).read_text(encoding="utf-8")
    trem_texto, val_texto = dividir_corpus(texto, semente=cfg.semente)
    x_trem, y_trem = montar_dataset(tokenizer, trem_texto, cfgm.janela_ctx, stride=cfg.stride or None)
    x_val, y_val = montar_dataset(tokenizer, val_texto, cfgm.janela_ctx)
    passos_por_epoch = max(1, x_trem.shape[0] // cfg.lote)
    lotes_val = list(range(min(cfg.iters_avaliacao, x_val.shape[0])))

    passo_atual = 0
    alvo = cfg.passos if limite is None else min(limite, cfg.passos)
    if retomar:
        ck_caminho = ultimo_checkpoint(run_dir)
        if ck_caminho is None:
            raise RuntimeError(f"retomada sem checkpoint em {run_dir/'ckpt'}")
        ck = torch.load(ck_caminho, map_location=dispositivo, weights_only=True)
        modelo.load_state_dict(ck["modelo"], strict=True)
        otimizador.load_state_dict(ck["otimizador"])
        passo_atual = int(ck["passo"])
        eventos.registrar("ajuste_retomado", run=run_dir.name, desde_passo=passo_atual)
    else:
        salvar_estado(
            run_dir,
            run_id=run_dir.name,
            tipo=cfg.tipo,
            bits=cfg.bits if quant else None,
            r=cfg.r,
            alpha=cfg.alpha,
            base=str(base_dir),
            passo=0,
            passos_totais=cfg.passos,
            concluido=False,
        )
        eventos.registrar("ajuste_iniciado", run=run_dir.name, base=base_dir.name, tipo=cfg.tipo, treinaveis=stats["treinaveis"])
        registrar_metrica(
            run_dir,
            {
                "passo": 0,
                "origem": "base",
                "loss_trem": None,
                "loss_val": _perda_media(modelo, x_val, y_val, lotes_val, dispositivo),
                "lr": 0.0,
                "tokens_por_s": 0.0,
                "tempo_s": 0.0,
                "dispositivo": str(dispositivo),
            },
        )

    inicio = time.time()
    perdas_trem: list[float] = []
    ultimo_reporte = inicio
    for passo in range(passo_atual, alvo):
        lr = fator_lr(cfg, passo)
        for grupo in otimizador.param_groups:
            grupo["lr"] = lr
        xb, yb = lote_trem(x_trem, y_trem, passo, cfg.lote, cfg.semente, passos_por_epoch)
        _, perda = modelo(xb.to(dispositivo), yb.to(dispositivo))
        perda.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(treinaveis, cfg.grad_clip)
        otimizador.step()
        otimizador.zero_grad(set_to_none=True)
        perdas_trem.append(float(perda.detach()))

        passo_feito = passo + 1
        avaliar = (passo_feito % cfg.avaliar_a_cada == 0) or (passo_feito == alvo)
        salvar = (passo_feito % cfg.salvar_a_cada == 0) or (passo_feito == alvo)
        if not (avaliar or salvar):
            continue
        agora = time.time()
        if avaliar:
            registrar_metrica(
                run_dir,
                {
                    "passo": passo_feito,
                    "loss_trem": sum(perdas_trem) / max(1, len(perdas_trem)),
                    "loss_val": _perda_media(modelo, x_val, y_val, lotes_val, dispositivo),
                    "lr": lr,
                    "tokens_por_s": cfg.lote * cfgm.janela_ctx * len(perdas_trem) / max(1e-6, agora - ultimo_reporte),
                    "tempo_s": round(agora - inicio, 2),
                    "dispositivo": str(dispositivo),
                },
            )
            perdas_trem = []
            ultimo_reporte = agora
        if salvar:
            salvar_checkpoint(
                run_dir,
                passo_feito,
                {"passo": passo_feito, "modelo": modelo.state_dict(), "otimizador": otimizador.state_dict()},
            )
            concluir = passo_feito >= cfg.passos
            salvar_estado(run_dir, passo=passo_feito, concluido=concluir, proporcao_adaptador=round(stats["proporcao"], 5))
            if concluir:
                salvar_adaptador(
                    modelo,
                    run_dir / "adaptador",
                    meta_extra={
                        "tipo": cfg.tipo,
                        "base": str(base_dir.resolve()),
                        "base_passo": int(peso_base["passo"]),
                        "semente": cfg.semente,
                        "nome_base": base_dir.name,
                    },
                )
                eventos.registrar("ajuste_concluido", run=run_dir.name, passo=passo_feito)
    return run_dir
