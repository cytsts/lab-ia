"""Laço de treinamento do zero: determinístico, retomável, métricas append-only."""
from __future__ import annotations

import dataclasses
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from ..experiments.runner import (
    carregar_estado,
    ler_metricas,
    registrar_metrica,
    salvar_checkpoint,
    salvar_estado,
    ultimo_checkpoint,
)
from ..models.gpt import ConfigGPT, GPT
from ..utils.estado import LogEventos
from .dados import dividir_corpus, lote_trem, montar_dataset
from .tokenizacao import carregar_tokenizer, treinar_tokenizer_ptbr


@dataclass
class ConfigTreino:
    nome: str = "g1-treino-zero"
    corpus: str = "data/corpus_ptbr.txt"
    vocab_bpe: int = 4096
    passos: int = 4000
    lote: int = 32
    avaliar_a_cada: int = 200
    salvar_a_cada: int = 200
    iters_avaliacao: int = 40
    lr: float = 3e-4
    minimo_lr: float = 3e-5
    warmup: int = 100
    peso_decay: float = 0.1
    grad_clip: float = 1.0
    semente: int = 42
    stride: int = 0  # 0 = janela inteira (sem sobreposição); <janela = janelas deslizantes
    dispositivo: str = "auto"  # auto | cuda | cpu
    modelo: dict = field(default_factory=dict)
    arquivo_eventos: str = ".lab-ia/eventos.jsonl"

    @classmethod
    def de_arquivo(cls, caminho: Path | str) -> "ConfigTreino":
        dados = yaml.safe_load(Path(caminho).read_text(encoding="utf-8")) or {}
        nomes = {f.name for f in dataclasses.fields(cls)}
        desconhecidas = set(dados) - nomes
        if desconhecidas:
            raise ValueError(f"chaves desconhecidas na config: {sorted(desconhecidas)}")
        return cls(**dados)


def escolher_dispositivo(preferencia: str) -> torch.device:
    if preferencia == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferencia == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("dispositivo 'cuda' solicitado mas indisponível")
    return torch.device(preferencia)


def fator_lr(cfg: ConfigTreino, passo: int) -> float:
    """Warmup linear + cosseno até minimo_lr."""
    if passo < cfg.warmup:
        return cfg.lr * (passo + 1) / max(1, cfg.warmup)
    span = max(1, cfg.passos - cfg.warmup)
    coef = 0.5 * (1.0 + math.cos(math.pi * (passo - cfg.warmup) / span))
    return cfg.minimo_lr + (cfg.lr - cfg.minimo_lr) * coef


@torch.no_grad()
def _perda_media(modelo: GPT, x: torch.Tensor, y: torch.Tensor, lotes: list, dispositivo) -> float:
    modelo.eval()
    totais = 0.0
    for ix in lotes:
        logits, perda = modelo(x[ix : ix + 1].to(dispositivo), y[ix : ix + 1].to(dispositivo))
        totais += float(perda.detach())
    modelo.train()
    return totais / max(1, len(lotes))


def executar_treino(
    config: ConfigTreino | dict,
    run_dir: Path,
    retomar: bool = False,
    limite: int | None = None,
    raiz: Path | str = ".",
) -> Path:
    """Treina (ou retoma) um run. `limite` encerra antecipadamente salvando estado (uso de teste)."""
    cfg = config if isinstance(config, ConfigTreino) else ConfigTreino(**dict(config))
    run_dir = Path(run_dir)
    random.seed(cfg.semente)
    torch.manual_seed(cfg.semente)
    dispositivo = escolher_dispositivo(cfg.dispositivo)
    eventos = LogEventos(Path(raiz) / cfg.arquivo_eventos if not Path(cfg.arquivo_eventos).is_absolute() else cfg.arquivo_eventos)

    texto = Path(cfg.corpus).read_text(encoding="utf-8")
    trem_texto, val_texto = dividir_corpus(texto, semente=cfg.semente)

    tok_arquivo = run_dir / "tokens" / "tokenizer.json"
    if retomar:
        if not tok_arquivo.exists():
            raise RuntimeError(f"retomada sem tokenizer salvo em {tok_arquivo}")
        tokenizer = carregar_tokenizer(tok_arquivo.parent)
    else:
        tokenizer = treinar_tokenizer_ptbr([trem_texto], cfg.vocab_bpe, tok_arquivo)

    cfgm = ConfigGPT.de_dict({"vocab": tokenizer.get_vocab_size(), **cfg.modelo})
    x_trem, y_trem = montar_dataset(tokenizer, trem_texto, cfgm.janela_ctx, stride=cfg.stride or None)
    x_val, y_val = montar_dataset(tokenizer, val_texto, cfgm.janela_ctx)
    passos_por_epoch = max(1, x_trem.shape[0] // cfg.lote)
    lotes_val = list(range(min(cfg.iters_avaliacao, x_val.shape[0])))

    modelo = GPT(cfgm)
    modelo.init_pesos(cfg.semente)
    modelo.to(dispositivo)

    decaem = [p for n, p in modelo.named_parameters() if p.dim() >= 2 and "wpe" not in n]
    nao_decaem = [p for n, p in modelo.named_parameters() if not (p.dim() >= 2 and "wpe" not in n)]
    otimizador = torch.optim.AdamW(
        [
            {"params": decaem, "weight_decay": cfg.peso_decay},
            {"params": nao_decaem, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    passo_atual = 0
    base_tempo = 0.0
    alvo = cfg.passos if limite is None else min(limite, cfg.passos)
    if retomar:
        ck_caminho = ultimo_checkpoint(run_dir)
        if ck_caminho is None:
            raise RuntimeError(f"retomada sem checkpoint em {run_dir/'ckpt'}")
        ck = torch.load(ck_caminho, map_location=dispositivo, weights_only=True)
        modelo.load_state_dict(ck["modelo"], strict=True)
        otimizador.load_state_dict(ck["otimizador"])
        passo_atual = int(ck["passo"])
        estado_prev = carregar_estado(run_dir) or {}
        base_tempo = float(estado_prev.get("tempo_decorrido_s", 0.0))
        eventos.registrar("treino_retomado", run=run_dir.name, desde_passo=passo_atual)
    else:
        salvar_estado(run_dir, run_id=run_dir.name, passo=0, passos_totais=cfg.passos, concluido=False)
        eventos.registrar("treino_iniciado", run=run_dir.name, passos=cfg.passos, dispositivo=str(dispositivo))
        # linha de base (passo 0): modelo aleatório + entropia unigram — referência de convergência
        contagens = torch.bincount(y_trem.reshape(-1), minlength=cfgm.vocab)
        p = contagens.float() / max(1, int(contagens.sum()))
        entropia_unigram = float(-(p[p > 0] * p[p > 0].log()).sum())
        registrar_metrica(
            run_dir,
            {
                "passo": 0,
                "loss_trem": None,
                "loss_val": _perda_media(modelo, x_val, y_val, lotes_val, dispositivo),
                "entropia_unigram": entropia_unigram,
                "lr": 0.0,
                "tokens_por_s": 0.0,
                "tempo_s": 0.0,
                "dispositivo": str(dispositivo),
            },
        )

    inicio_segmento = time.time()
    perdas_trem: list[float] = []
    ultimo_reporte = inicio_segmento
    uso_bf16 = dispositivo.type == "cuda"

    for passo in range(passo_atual, alvo):
        lr = fator_lr(cfg, passo)
        for grupo in otimizador.param_groups:
            grupo["lr"] = lr
        xb, yb = lote_trem(x_trem, y_trem, passo, cfg.lote, cfg.semente, passos_por_epoch)
        with torch.autocast(dispositivo.type, dtype=torch.bfloat16, enabled=uso_bf16):
            _, perda = modelo(xb.to(dispositivo), yb.to(dispositivo))
        perda.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
        otimizador.step()
        otimizador.zero_grad(set_to_none=True)
        perdas_trem.append(float(perda.detach()))

        avaliar = ((passo + 1) % cfg.avaliar_a_cada == 0) or (passo + 1 == alvo)
        salvar = ((passo + 1) % cfg.salvar_a_cada == 0) or (passo + 1 == alvo)
        if not (avaliar or salvar):
            continue

        tempo_decorrido = base_tempo + (time.time() - inicio_segmento)
        if avaliar:
            loss_val = _perda_media(modelo, x_val, y_val, lotes_val, dispositivo)
            agora = time.time()
            tokens_avaliados = cfg.lote * cfgm.janela_ctx * len(perdas_trem)
            registrar_metrica(
                run_dir,
                {
                    "passo": passo + 1,
                    "loss_trem": sum(perdas_trem) / len(perdas_trem),
                    "loss_val": loss_val,
                    "lr": lr,
                    "tokens_por_s": tokens_avaliados / max(1e-6, agora - ultimo_reporte),
                    "tempo_s": round(tempo_decorrido, 2),
                    "dispositivo": str(dispositivo),
                },
            )
            perdas_trem = []
            ultimo_reporte = agora

        if salvar:
            salvar_checkpoint(
                run_dir,
                passo + 1,
                {
                    "passo": passo + 1,
                    "modelo": modelo.state_dict(),
                    "otimizador": otimizador.state_dict(),
                    "config_modelo": cfgm.para_dict(),
                },
            )
            concluir = passo + 1 >= cfg.passos
            salvar_estado(
                run_dir,
                passo=passo + 1,
                passos_totais=cfg.passos,
                concluido=concluir,
                tempo_decorrido_s=round(tempo_decorrido, 2),
                dispositivo=str(dispositivo),
            )
            if concluir:
                eventos.registrar("treino_concluido", run=run_dir.name, passo=passo + 1)

    return run_dir
