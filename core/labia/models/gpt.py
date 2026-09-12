"""GPT decoder-only minimalista (dimensões pequenas), treinável do zero em CPU/GPU."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .componentes import RMSNorm, RedeSwiGLU, aplicar_rope, rope_frequencias


@dataclass
class ConfigGPT:
    vocab: int = 4096
    dim: int = 256
    camadas: int = 6
    cabecas: int = 8
    janela_ctx: int = 256
    abandono: float = 0.1
    n_especialistas: int = 0  # 0 = FFN denso; >1 = camada MoE (spec G4)
    top_k: int = 1  # especialistas ativos por token
    coef_auxiliar: float = 0.01  # peso da perda de balanceamento de carga
    # --- arquitetura (B7): padrões reproduzem o GPT-2 original do laboratório ---
    norm: str = "layernorm"  # layernorm | rmsnorm
    pos: str = "aprendido"  # aprendido | rope
    mlp: str = "gelu"  # gelu | swiglu
    n_cabecas_kv: int = 0  # 0 = igual a cabecas (MHA); menor que cabecas = GQA
    rope_base: float = 10000.0

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def de_dict(cls, dados: dict) -> "ConfigGPT":
        nomes = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dados.items() if k in nomes})


class AtencaoMultiCabeca(nn.Module):
    """Atenção causal; suporta GQA (menos cabeças de K/V que de Q) e RoPE.

    Com n_cabecas_kv = 0 o comportamento é idêntico ao original: uma projeção qkv de
    3·dim e atenção multi-cabeça comum. Com n_cabecas_kv < cabecas, K e V são menores
    e cada cabeça de K/V serve um grupo de cabeças de Q — é o que a Llama usa para
    reduzir o cache de KV sem perder qualidade.
    """

    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        if cfg.dim % cfg.cabecas:
            raise ValueError("dim deve ser divisível por cabecas")
        self.cabecas = cfg.cabecas
        self.cabeca_dim = cfg.dim // cfg.cabecas
        self.cabecas_kv = cfg.n_cabecas_kv or cfg.cabecas
        if cfg.dim % self.cabecas_kv:
            raise ValueError("dim deve ser divisível por n_cabecas_kv")
        if self.cabecas % self.cabecas_kv:
            raise ValueError("cabecas precisa ser múltiplo de n_cabecas_kv (cada K/V serve um grupo)")
        self.kv_dim = self.cabecas_kv * self.cabeca_dim
        self.qkv = nn.Linear(cfg.dim, cfg.dim + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.abandono = nn.Dropout(cfg.abandono)
        self.usar_rope = cfg.pos == "rope"

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split([d, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(b, t, self.cabecas, self.cabeca_dim).transpose(1, 2)
        k = k.view(b, t, self.cabecas_kv, self.cabeca_dim).transpose(1, 2)
        v = v.view(b, t, self.cabecas_kv, self.cabeca_dim).transpose(1, 2)
        if self.usar_rope and rope is not None:
            q = aplicar_rope(q, rope[0], rope[1])
            k = aplicar_rope(k, rope[0], rope[1])
        saida = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.abandono.p if self.training else 0.0,
            enable_gqa=self.cabecas_kv != self.cabecas,
        )
        saida = saida.transpose(1, 2).reshape(b, t, d)
        return self.abandono(self.proj(saida))


class RedeDensa(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, 4 * cfg.dim)
        self.fc2 = nn.Linear(4 * cfg.dim, cfg.dim)
        self.abandono = nn.Dropout(cfg.abandono)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.abandono(self.fc2(F.gelu(self.fc1(x))))
        return y, torch.zeros((), device=y.device)


class Roteador(nn.Module):
    """Porta top-k estilo Switch: softmax → top-k renormalizado; perda aux de carga."""

    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.n = max(1, cfg.n_especialistas)
        self.k = min(max(1, cfg.top_k), self.n)
        self.peso = nn.Linear(cfg.dim, self.n, bias=False)
        self.ultima_fracao: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.peso(x), dim=-1)
        val, idx = probs.topk(self.k, dim=-1)
        val = val / val.sum(-1, keepdim=True)
        destino = torch.zeros_like(probs).scatter_(1, idx, val)
        with torch.no_grad():
            # f_i: fração de tokens roteada para o especialista i — contagem dura, sem gradiente
            fracao = destino.gt(0).float().mean(0)
            self.ultima_fracao = fracao.detach()
        # P̄_i (probabilidade média do roteador) PRECISA carregar gradiente: é por onde a
        # perda auxiliar empurra o roteador de volta ao equilíbrio (Switch Transformer,
        # RF3 da G4). Manter o produto inteiro dentro de no_grad tornava o termo inerte —
        # somado à perda, mas sem nenhum efeito sobre o treino.
        aux = self.n * (fracao * probs.mean(0)).sum()
        return destino, aux


class CamadaMoE(nn.Module):
    """FFN esparsamente ativado: n especialistas, top-k por token (spec G4)."""

    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.roteador = Roteador(cfg)
        self.especialistas = nn.ModuleList(RedeDensa(cfg) for _ in range(self.roteador.n))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        xf = x.reshape(-1, d)
        g, aux = self.roteador(xf)
        y = torch.zeros_like(xf)
        for i, esp in enumerate(self.especialistas):
            m = g[:, i] > 0
            if m.any():
                y[m] = y[m] + g[m, i : i + 1] * esp(xf[m])[0]
        return y.view(b, t, d), aux


def criar_normalizacao(cfg: ConfigGPT) -> nn.Module:
    """LayerNorm (GPT-2) ou RMSNorm (Llama em diante)."""
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.dim)
    if cfg.norm != "layernorm":
        raise ValueError(f"norm desconhecido: {cfg.norm!r} (use layernorm|rmsnorm)")
    return nn.LayerNorm(cfg.dim)


class BlocoTransformer(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.ln1 = criar_normalizacao(cfg)
        self.atencao = AtencaoMultiCabeca(cfg)
        self.ln2 = criar_normalizacao(cfg)
        if cfg.n_especialistas > 1:
            self.mlp = CamadaMoE(cfg)
        elif cfg.mlp == "swiglu":
            self.mlp = RedeSwiGLU(cfg)
        elif cfg.mlp == "gelu":
            self.mlp = RedeDensa(cfg)
        else:
            raise ValueError(f"mlp desconhecido: {cfg.mlp!r} (use gelu|swiglu)")

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.atencao(self.ln1(x), rope)
        saida, aux = self.mlp(self.ln2(x))
        return x + saida, aux


class GPT(nn.Module):
    def __init__(self, cfg: ConfigGPT):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.dim)
        if cfg.pos == "aprendido":
            self.wpe = nn.Embedding(cfg.janela_ctx, cfg.dim)
        elif cfg.pos == "rope":
            # RoPE dispensa embedding de posição: a posição entra girando q e k.
            self.wpe = None
            if (cfg.dim // cfg.cabecas) % 2:
                raise ValueError(
                    f"RoPE precisa de dim_cabeca par (dim {cfg.dim} / cabecas {cfg.cabecas}); "
                    "ajuste --dim ou --cabecas"
                )
            cos, sen = rope_frequencias(
                cfg.janela_ctx, cfg.dim // cfg.cabecas, cfg.rope_base
            )
            # persistent=False: a tabela é derivada da config, não é peso aprendido — não
            # precisa viajar no checkpoint nem mudar o state_dict dos runs antigos.
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sen", sen, persistent=False)
        else:
            raise ValueError(f"pos desconhecido: {cfg.pos!r} (use aprendido|rope)")
        self.abandono = nn.Dropout(cfg.abandono)
        self.blocos = nn.ModuleList(BlocoTransformer(cfg) for _ in range(cfg.camadas))
        self.ln_f = criar_normalizacao(cfg)
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
                mlp = bloco.mlp
                redes = mlp.especialistas if isinstance(mlp, CamadaMoE) else [mlp]
                for red in redes:
                    # a projeção de SAÍDA do bloco é que precisa nascer pequena
                    if isinstance(red, RedeSwiGLU):
                        red.w_down.weight.mul_(escala)
                    else:
                        red.fc2.weight.mul_(escala)
            for ln in [self.ln_f, *(b.ln1 for b in self.blocos), *(b.ln2 for b in self.blocos)]:
                ln.weight.fill_(1.0)
                if getattr(ln, "bias", None) is not None:  # RMSNorm não tem viés
                    ln.bias.zero_()

    @property
    def dispositivo(self) -> torch.device:
        return self.wte.weight.device

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def contar_parametros_ativos(self) -> int:
        """Custo real por token: com MoE top-k, só k de n especialistas por camada."""
        total = self.contar_parametros()
        moe = [b for b in self.blocos if isinstance(b.mlp, CamadaMoE)]
        if not moe:
            return total
        por_especialista = sum(p.numel() for p in moe[0].mlp.especialistas[0].parameters())
        k = moe[0].mlp.roteador.k
        n = moe[0].mlp.roteador.n
        return total - (n - k) * por_especialista * len(moe)

    def mapa_uso_especialistas(self) -> list[float] | None:
        """Fração média de tokens por especialista (por camada MoE), do último forward."""
        moe = [b.mlp for b in self.blocos if isinstance(b.mlp, CamadaMoE)]
        if not moe or any(m.roteador.ultima_fracao is None for m in moe):
            return None
        frac = torch.stack([m.roteador.ultima_fracao for m in moe]).mean(0)
        return [round(float(v), 4) for v in frac]

    @property
    def ultimo_aux(self) -> float | None:
        return getattr(self, "_ultimo_aux", None)

    def emb(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        if t > self.cfg.janela_ctx:
            raise ValueError(f"sequência {t} > janela_ctx {self.cfg.janela_ctx}")
        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(t, device=idx.device)
            x = x + self.wpe(pos)[None, :, :]
        return self.abandono(x)

    def forward(self, idx: torch.Tensor, alvos: torch.Tensor | None = None):
        x = self.emb(idx)
        rope = None
        if self.cfg.pos == "rope":
            tokens = idx.shape[1]
            rope = (self._rope_cos[:tokens], self._rope_sen[:tokens])
        aux_total = x.new_zeros(())
        for bloco in self.blocos:
            x, aux = bloco(x, rope)
            aux_total = aux_total + aux
        logits = self.cabeca(self.ln_f(x))
        n_moe = sum(1 for b in self.blocos if isinstance(b.mlp, CamadaMoE))
        self._ultimo_aux = float(aux_total.detach() / max(1, n_moe)) if n_moe else None
        if alvos is None:
            return logits, None
        perda = F.cross_entropy(logits.view(-1, self.cfg.vocab), alvos.view(-1))
        if n_moe and self.cfg.coef_auxiliar:
            perda = perda + self.cfg.coef_auxiliar * aux_total / n_moe
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
