"""Geração de texto de amostra a partir do último checkpoint de um run."""
from __future__ import annotations

from pathlib import Path

import torch

from ..experiments.runner import ultimo_checkpoint
from ..models.gpt import ConfigGPT, GPT
from .tokenizacao import carregar_tokenizer


def gerar_de_checkpoint(
    run_dir: Path | str,
    prompt: str,
    passos_max: int = 120,
    temperatura: float = 0.8,
    topo_k: int = 50,
    guloso: bool = False,
    semente: int = 1234,
    dispositivo: str = "auto",
) -> str:
    run_dir = Path(run_dir)
    ck_caminho = ultimo_checkpoint(run_dir)
    if ck_caminho is None:
        raise RuntimeError(f"nenhum checkpoint em {run_dir}/ckpt")
    disp = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if dispositivo == "auto"
        else dispositivo
    )
    ck = torch.load(ck_caminho, map_location=disp, weights_only=True)
    tokenizer = carregar_tokenizer(run_dir / "tokens")
    modelo = GPT(ConfigGPT.de_dict(ck["config_modelo"]))
    modelo.load_state_dict(ck["modelo"], strict=True)
    modelo.to(disp).eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    idx = torch.tensor([ids], dtype=torch.long, device=disp)
    gen = torch.Generator(device=disp).manual_seed(semente)
    saida = modelo.gerar(
        idx,
        passos_max=passos_max,
        temperatura=0.0 if guloso else temperatura,
        topo_k=0 if guloso else topo_k,
        gerador=gen,
    )
    return tokenizer.decode(saida[0, len(ids) :].tolist())
