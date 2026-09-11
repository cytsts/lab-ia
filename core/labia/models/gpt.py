"""GPT decoder-only minimalista (dimensões pequenas), treinável do zero em CPU/GPU."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfigGPT:
    vocab: int = 4096
    dim: int = 256
    camadas: int = 6
    cabecas: int = 8
    janela_ctx: int = 256
    abandono: float = 0.1

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def de_dict(cls, dados: dict) -> "ConfigGPT":
        nomes = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dados.items() if k in nomes})


class AtencaoMultiCabeca(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        if cfg.dim % cfg.cabecas:
            raise ValueError("dim deve ser divisível por cabecas")
        self.cabecas = cfg.cabecas
        self.cabeca_dim = cfg.dim // cfg.cabecas
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.abandono = nn.Dropout(cfg.abandono)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.cabecas, self.cabeca_dim).transpose(1, 2)
        k = k.view(b, t, self.cabecas, self.cabeca_dim).transpose(1, 2)
        v = v.view(b, t, self.cabecas, self.cabeca_dim).transpose(1, 2)
        saida = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.abandono.p if self.training else 0.0
        )
        saida = saida.transpose(1, 2).reshape(b, t, d)
        return self.abandono(self.proj(saida))


class RedeDensa(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, 4 * cfg.dim)
        self.fc2 = nn.Linear(4 * cfg.dim, cfg.dim)
        self.abandono = nn.Dropout(cfg.abandono)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.abandono(self.fc2(F.gelu(self.fc1(x))))


class BlocoTransformer(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.atencao = AtencaoMultiCabeca(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = RedeDensa(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.atencao(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.dim)
        self.wpe = nn.Embedding(cfg.janela_ctx, cfg.dim)
        self.abandono = nn.Dropout(cfg.abandono)
        self.blocos = nn.ModuleList(BlocoTransformer(cfg) for _ in range(cfg.camadas))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.cabeca = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.cabeca.weight = self.wte.weight  # pesos atados

    def init_pesos(self, semente: int = 0) -> None:
        gerador = torch.Generator().manual_seed(semente)
        for nome, p in self.named_parameters():
            if p.dim() >= 2 or nome.endswith(".weight"):
                std = 0.01 if "wte" in nome or "wpe" in nome else 0.02
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=gerador) * std)
            elif nome.endswith(".bias"):
                nn.init.zeros_(p)
        # escala das projeções de resíduo (GPT-2: sqrt(2*camadas))
        with torch.no_grad():
            escala = 1.0 / math.sqrt(2 * self.cfg.camadas)
            for bloco in self.blocos:
                bloco.atencao.proj.weight.mul_(escala)
                bloco.mlp.fc2.weight.mul_(escala)
            for ln in [self.ln_f, *(b.ln1 for b in self.blocos), *(b.ln2 for b in self.blocos)]:
                ln.weight.fill_(1.0)
                ln.bias.zero_()

    @property
    def dispositivo(self) -> torch.device:
        return self.wte.weight.device

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def emb(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        if t > self.cfg.janela_ctx:
            raise ValueError(f"sequência {t} > janela_ctx {self.cfg.janela_ctx}")
        pos = torch.arange(t, device=idx.device)
        return self.abandono(self.wte(idx) + self.wpe(pos)[None, :, :])

    def forward(self, idx: torch.Tensor, alvos: torch.Tensor | None = None):
        x = self.emb(idx)
        for bloco in self.blocos:
            x = bloco(x)
        logits = self.cabeca(self.ln_f(x))
        if alvos is None:
            return logits, None
        perda = F.cross_entropy(logits.view(-1, self.cfg.vocab), alvos.view(-1))
        return logits, perda

    @torch.no_grad()
    def gerar(
        self,
        idx: torch.Tensor,
        passos_max: int = 120,
        temperatura: float = 0.8,
        topo_k: int = 50,
        gerador: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(passos_max):
            if idx.shape[1] >= self.cfg.janela_ctx:
                idx = idx[:, -self.cfg.janela_ctx :]
            logits = self.forward(idx)[0][:, -1, :]
            if temperatura <= 0:
                prox = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperatura
                if topo_k:
                    limiar, _ = torch.topk(logits, min(topo_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < limiar[:, -1:], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                prox = torch.multinomial(probs, num_samples=1, generator=gerador)
            idx = torch.cat([idx, prox], dim=1)
        return idx


def gerar(
    modelo: GPT,
    texto_prompt: str,
    codificador,
    passos_max: int = 120,
    temperatura: float = 0.8,
    topo_k: int = 50,
    semente: int = 1234,
) -> str:
    """Gera texto a partir de prompt (continuação). codificador: objeto com encode/decode."""
    ids = codificador.encode(texto_prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=modelo.dispositivo)
    gen = torch.Generator(device="cpu").manual_seed(semente)
    if modelo.dispositivo.type == "cuda":
        gen = torch.Generator(device=modelo.dispositivo).manual_seed(semente)
    saida = modelo.gerar(idx, passos_max=passos_max, temperatura=temperatura, topo_k=topo_k, gerador=gen)
    return codificador.decode(saida[0, len(ids) :].tolist())
