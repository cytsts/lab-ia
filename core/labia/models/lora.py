"""LoRA nativo (adaptador de baixo posto) sobre a nossa arquitetura GPT.

Implementação própria de propósito: o laboratório demonstra a técnica
(Hu et al. 2021) em vez de só invocar biblioteca. QLoRA = base quantizada
congelada (ver quant.py) + estes adaptadores treináveis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import dequant_int8, dequant_nf4, quantizar

ALVOS_PADRAO = ("atencao.qkv", "atencao.proj", "mlp.fc1", "mlp.fc2")
_CHAVE_ADAPTADOR = "adaptador"
_CHAVE_BASE_QUANT = "base_quantizada"


class NoLinear(nn.Module):
    """Linear congelada + ramo de baixo posto B·A·(α/r); base opcional quantizada."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, quant: str | None = None):
        super().__init__()
        entrada, saida = base.in_features, base.out_features
        self.entrada, self.saida, self.r, self.alpha = entrada, saida, r, alpha
        self.escala = alpha / r if r else 0.0
        self.quant = quant
        vies = base.bias
        self.vies = nn.Parameter(vies.detach().clone(), requires_grad=False) if vies is not None else None
        if quant is None:
            self.peso = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        else:
            self.peso = None
            buf = quantizar(quant, base.weight.detach())
            chave = "q_peso" if quant == "int8" else "q_codigos"
            self.register_buffer(chave, buf["peso"] if quant == "int8" else buf["codigos"])
            self.register_buffer("q_escala", (buf["escala"] if quant == "int8" else buf["escalas"]).float())
            self.register_buffer("q_forma", buf["forma"])
        if r:
            # os ramos nascem no dispositivo da base: aplicar LoRA em modelo já em CUDA
            # sem `.to()` posterior funcionaria com matmul em dispositivos mistos
            dispositivo_base = base.weight.device
            a = torch.empty(r, entrada, device=dispositivo_base)
            nn.init.normal_(a, mean=0.0, std=1.0 / math.sqrt(entrada))
            self.lora_A = nn.Parameter(a)
            self.lora_B = nn.Parameter(torch.zeros(saida, r, device=dispositivo_base))  # init zerado: parte da base
        else:
            self.lora_A = None  # quantização pura: sem ramo adaptador
            self.lora_B = None

    def peso_base(self) -> torch.Tensor:
        if self.quant is None:
            return self.peso
        forma = tuple(self.q_forma.tolist())
        if self.quant == "int8":
            w = dequant_int8({"peso": self.q_peso, "escala": self.q_escala, "forma": self.q_forma})
        else:
            w = dequant_nf4({"codigos": self.q_codigos, "escalas": self.q_escala.half(), "forma": self.q_forma})
        return w.view(forma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.peso_base(), self.vies)
        if self.lora_A is not None:
            y = y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.escala
        return y


def _substituir_lineares(modelo: nn.Module, r: int, alpha: float, alvos: tuple[str, ...], quant: str | None):
    trocados = []
    for nome, modulo in list(modelo.named_modules()):
        if isinstance(modulo, nn.Linear) and any(nome.endswith(alvo) for alvo in alvos):
            pai_nome, _, filho = nome.rpartition(".")
            pai = modelo.get_submodule(pai_nome) if pai_nome else modelo
            no = NoLinear(modulo, r=r, alpha=alpha, quant=quant)
            setattr(pai, filho, no)
            trocados.append(nome)
    return trocados


def aplicar_lora(
    modelo: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    alvos: tuple[str, ...] = ALVOS_PADRAO,
    quant: str | None = None,
) -> dict:
    """Congela a base e injeta adaptadores nos lineares-alvo. Retorna estatísticas."""
    for p in modelo.parameters():
        p.requires_grad_(False)
    trocados = _substituir_lineares(modelo, r, alpha, alvos, quant)
    for nome, p in modelo.named_parameters():
        if nome.endswith(".lora_A") or nome.endswith(".lora_B"):
            p.requires_grad_(True)
    stats = estatisticas_lora(modelo)
    stats["modulos_alterados"] = trocados
    stats["quant"] = quant
    return stats


def estatisticas_lora(modelo: nn.Module) -> dict:
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return {"treinaveis": treinaveis, "total": total, "proporcao": treinaveis / max(1, total)}


def estado_adaptador(modelo: nn.Module) -> dict:
    """Só o que pertence ao experimento de ajuste: A/B treináveis + buffers quantizados."""
    adaptador: dict[str, dict[str, torch.Tensor]] = {}
    base_quant: dict[str, torch.Tensor] = {}
    for nome, modulo in modelo.named_modules():
        if isinstance(modulo, NoLinear) and modulo.lora_A is not None:
            adaptador[nome] = {"A": modulo.lora_A.detach().clone(), "B": modulo.lora_B.detach().clone()}
            if modulo.quant is not None:
                for bnome, buf in modulo.named_buffers():
                    base_quant[f"{nome}.{bnome}"] = buf.detach().clone()
    return {_CHAVE_ADAPTADOR: adaptador, _CHAVE_BASE_QUANT: base_quant}


def carregar_estado_adaptador(modelo: nn.Module, estado: dict) -> None:
    modulos = dict(modelo.named_modules())
    for nome, par in estado[_CHAVE_ADAPTADOR].items():
        no = modulos.get(nome)
        if not isinstance(no, NoLinear):
            raise RuntimeError(f"modelo sem NoLinear em {nome!r} — aplique o LoRA antes de carregar")
        with torch.no_grad():
            no.lora_A.copy_(par["A"])
            no.lora_B.copy_(par["B"])
    if estado.get(_CHAVE_BASE_QUANT):
        buffers = dict(modelo.named_buffers())
        for nome, t in estado[_CHAVE_BASE_QUANT].items():
            if nome not in buffers:
                raise RuntimeError(f"buffer quantizado {nome!r} ausente no modelo")
            with torch.no_grad():
                buffers[nome].copy_(t)


def salvar_adaptador(modelo: nn.Module, pasta: Path | str, meta_extra: dict | None = None) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    estado = estado_adaptador(modelo)
    if not estado[_CHAVE_ADAPTADOR]:
        raise ValueError(
            "nenhum adaptador LoRA no modelo — nada a salvar "
            "(o modelo passou por mesclar_lora? salve o adaptador antes de fundir)"
        )
    torch.save(estado, pasta / "lora.pt")
    amostra = next(iter(estado[_CHAVE_ADAPTADOR].values()), None)
    no_exemplo = None
    for modulo in modelo.modules():
        if isinstance(modulo, NoLinear) and modulo.lora_A is not None:
            no_exemplo = modulo
            break
    meta = {
        "r": amostra["A"].shape[0],
        "alpha": float(no_exemplo.alpha),
        "quant": no_exemplo.quant,
        "modulos": sorted(estado[_CHAVE_ADAPTADOR]),
        "treinaveis": sum(t.numel() for d in estado[_CHAVE_ADAPTADOR].values() for t in d.values()),
    } if (amostra and (no_exemplo is not None)) else {}
    meta.update(meta_extra or {})
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return pasta


def carregar_adaptador(modelo: nn.Module, pasta: Path | str) -> dict:
    pasta = Path(pasta)
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    estado = torch.load(pasta / "lora.pt", map_location="cpu", weights_only=True)
    carregar_estado_adaptador(modelo, estado)
    return meta


def mesclar_lora(modelo: nn.Module) -> int:
    """Fusão W ← W + (B·A)·escala; volta a ser nn.Linear puro (sem custo no infer)."""
    trocados = 0
    for nome, modulo in list(modelo.named_modules()):
        if isinstance(modulo, NoLinear):
            pai_nome, _, filho = nome.rpartition(".")
            pai = modelo.get_submodule(pai_nome) if pai_nome else modelo
            with torch.no_grad():
                w = modulo.peso_base() + modulo.escala * (modulo.lora_B @ modulo.lora_A)
                linear = nn.Linear(modulo.entrada, modulo.saida, bias=modulo.vies is not None)
                linear.weight = nn.Parameter(w.float(), requires_grad=False)
                if modulo.vies is not None:
                    linear.bias = nn.Parameter(modulo.vies.detach().clone(), requires_grad=False)
            setattr(pai, filho, linear)
            trocados += 1
    return trocados
