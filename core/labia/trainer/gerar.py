"""Geração de texto: a partir de um run-base (G1) ou de um run de ajuste LoRA/QLoRA (G2)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..experiments.runner import ultimo_checkpoint
from ..models.gpt import ConfigGPT, GPT
from .tokenizacao import carregar_tokenizer


def _dispositivo(preferencia: str = "auto") -> torch.device:
    if preferencia == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferencia)


def _modelo_base(run_dir: Path, disp: torch.device):
    ck_caminho = ultimo_checkpoint(run_dir)
    if ck_caminho is None:
        raise RuntimeError(f"nenhum checkpoint em {run_dir}/ckpt")
    ck = torch.load(ck_caminho, map_location=disp, weights_only=True)
    modelo = GPT(ConfigGPT.de_dict(ck["config_modelo"]))
    modelo.load_state_dict(ck["modelo"], strict=True)
    return modelo, carregar_tokenizer(run_dir / "tokens")


def _preparar_modelo(run_dir: Path, disp: torch.device):
    """Devolve (modelo em eval, tokenizer) para run-base, run de ajuste ou run quantizado."""
    meta_quant = run_dir / "quant_meta.json"
    if meta_quant.exists():
        from .quantiza import _trocar_lineares

        meta = json.loads(meta_quant.read_text(encoding="utf-8"))
        base_dir = Path(meta["base"])
        if not base_dir.exists():
            raise FileNotFoundError(f"run-base da quantização não existe: {base_dir}")
        tokenizer = carregar_tokenizer(base_dir / "tokens")
        ck_caminho = ultimo_checkpoint(run_dir)
        if ck_caminho is None:
            raise RuntimeError(f"nenhum checkpoint em {run_dir}/ckpt")
        ck = torch.load(ck_caminho, map_location=disp, weights_only=True)
        modelo = GPT(ConfigGPT.de_dict(ck["config_modelo"]))
        _trocar_lineares(modelo, meta["modo"])  # estrutura quantizada primeiro, depois carrega buffers
        modelo.load_state_dict(ck["modelo"], strict=True)
        return modelo, tokenizer
    meta_caminho = run_dir / "adaptador" / "meta.json"
    if not meta_caminho.exists():
        modelo, tok = _modelo_base(run_dir, disp)
        return modelo, tok
    from ..models.lora import aplicar_lora, carregar_adaptador

    meta = json.loads(meta_caminho.read_text(encoding="utf-8"))
    base_dir = Path(meta["base"])
    if not base_dir.exists():
        raise FileNotFoundError(f"run-base do adaptador não existe: {base_dir}")
    modelo, tok_base = _modelo_base(base_dir, disp)
    tokenizer = carregar_tokenizer(base_dir / "tokens")
    stats = aplicar_lora(modelo, r=int(meta["r"]), alpha=float(meta["alpha"]), quant=meta.get("quant"))
    meta2 = carregar_adaptador(modelo, run_dir / "adaptador")
    if meta2["r"] != meta["r"] or meta2.get("quant") != meta.get("quant"):
        raise RuntimeError("meta.json do adaptador divergente do estado salvo")
    return modelo, tokenizer


def carregar_para_geracao(run_dir: Path | str, dispositivo: str = "auto"):
    """API pública: (modelo em eval, tokenizer, dispositivo) para qualquer tipo de run."""
    run_dir = Path(run_dir)
    disp = _dispositivo(dispositivo)
    modelo, tokenizer = _preparar_modelo(run_dir, disp)
    return modelo.to(disp).eval(), tokenizer, disp


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
    disp = _dispositivo(dispositivo)
    modelo, tokenizer = _preparar_modelo(run_dir, disp)
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
