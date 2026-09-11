"""Quantização de runs inteiros (spec G3): relatórios de tamanho, erro e perda."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..experiments.runner import salvar_estado, ultimo_checkpoint
from ..models.gpt import ConfigGPT, GPT
from ..models.lora import ALVOS_PADRAO, NoLinear
from ..models.quant import tamanho_state_dict
from ..utils.estado import LogEventos, agora_iso
from .dados import montar_dataset
from .tokenizacao import carregar_tokenizer
from .treino import escolher_dispositivo

_BITES_POR_PARAM = {"int8": 8.0, "nf4": 4.25}  # nf4: 4 + 16/64 (escala fp16 por bloco)


def _trocar_lineares(modelo: torch.nn.Module, modo: str) -> dict[str, float]:
    """Substitui lineares-alvo por NoLinear r=0 quantizado; relatório de erro relativo por módulo."""
    relatorio: dict[str, float] = {}
    for nome, modulo in list(modelo.named_modules()):
        if isinstance(modulo, torch.nn.Linear) and any(nome.endswith(a) for a in ALVOS_PADRAO):
            w = modulo.weight.detach().float()
            pai_nome, _, filho = nome.rpartition(".")
            pai = modelo.get_submodule(pai_nome) if pai_nome else modelo
            no = NoLinear(modulo, r=0, alpha=0.0, quant=modo)
            erro = float((w - no.peso_base().float()).norm() / w.norm().clamp_min(1e-8))
            setattr(pai, filho, no)
            relatorio[nome] = erro
    return relatorio


@torch.no_grad()
def _perda_em_janelas(modelo: GPT, x: torch.Tensor, y: torch.Tensor, dispositivo) -> float:
    modelo.eval()
    totais = 0.0
    for i in range(x.shape[0]):
        _, perda = modelo(x[i : i + 1].to(dispositivo), y[i : i + 1].to(dispositivo))
        totais += float(perda)
    return totais / x.shape[0]


def quantizar_run(
    base_dir: Path | str,
    destino_dir: Path | str,
    modo: str,
    corpus_avaliacao: Path | str,
    dispositivo: str = "auto",
    iters: int = 12,
    raiz: Path | str = ".",
    arquivo_eventos: str = ".lab-ia/eventos.jsonl",
) -> dict:
    """Quantiza os lineares dos blocos de um run-base e grava run derivado auditável."""
    if modo not in _BITES_POR_PARAM:
        raise ValueError(f"modo inválido {modo!r} (use {sorted(_BITES_POR_PARAM)})")
    base_dir = Path(base_dir)
    destino_dir = Path(destino_dir)
    (destino_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    disp = escolher_dispositivo(dispositivo)
    eventos = LogEventos(
        Path(arquivo_eventos) if Path(arquivo_eventos).is_absolute() else Path(raiz) / arquivo_eventos
    )

    ck = torch.load(ultimo_checkpoint(base_dir), map_location="cpu", weights_only=True)
    cfgm = ConfigGPT.de_dict(ck["config_modelo"])

    def _carregar_gpt() -> GPT:
        m = GPT(cfgm)
        m.load_state_dict(ck["modelo"], strict=True)
        return m

    fp32 = _carregar_gpt().to(disp).eval()
    quant = _carregar_gpt()
    erro_por_modulo = _trocar_lineares(quant, modo)  # troca lê os pesos fp32 já carregados
    quant = quant.to(disp).eval()

    tokenizer = carregar_tokenizer(base_dir / "tokens")
    texto = Path(corpus_avaliacao).read_text(encoding="utf-8")
    x, y = montar_dataset(tokenizer, texto, cfgm.janela_ctx)
    x, y = x[:iters], y[:iters]

    perda_antes = _perda_em_janelas(fp32, x, y, disp)
    perda_depois = _perda_em_janelas(quant, x, y, disp)

    alvos_bytes_fp32 = sum(
        t.numel() * t.element_size()
        for k, t in ck["modelo"].items()
        if k.endswith(".weight") and any(a in k for a in ALVOS_PADRAO)
    )
    estado_quant = quant.state_dict()
    alvos_bytes_quant = sum(
        t.numel() * t.element_size()
        for k, t in estado_quant.items()
        if ".q_" in k and any(a in k for a in ALVOS_PADRAO)
    )
    n_param_alvos = sum(
        t.numel() for k, t in ck["modelo"].items() if k.endswith(".weight") and any(a in k for a in ALVOS_PADRAO)
    )
    tamanhos = {
        "gerado_em": agora_iso(),
        "modo": modo,
        "base": str(base_dir),
        "base_passo": int(ck["passo"]),
        "bytes_fp32_alvos": int(alvos_bytes_fp32),
        "bytes_quantizados_alvos": int(alvos_bytes_quant),
        "bytes_total_state_fp32": tamanho_state_dict(ck["modelo"]),
        "bytes_total_state_quantizado": tamanho_state_dict(estado_quant),
        "fator_alvos": alvos_bytes_fp32 / max(1, alvos_bytes_quant),
        "bits_efetivos_por_parametro": round(8.0 * alvos_bytes_quant / max(1, n_param_alvos), 3),
        "erro_relativo_por_modulo": erro_por_modulo,
        "perda_val_antes": perda_antes,
        "perda_val_depois": perda_depois,
        "corpus_avaliacao": str(corpus_avaliacao),
        "iters": iters,
        "dispositivo": str(disp),
    }

    torch.save(
        {"passo": int(ck["passo"]), "modelo": {k: v.clone() for k, v in estado_quant.items()}, "config_modelo": cfgm.para_dict()},
        destino_dir / "ckpt" / f"passo-{int(ck['passo'])}.pt",
    )
    (destino_dir / "quant_meta.json").write_text(
        json.dumps({"modo": modo, "base": str(base_dir.resolve()), "alvos": list(ALVOS_PADRAO)}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (destino_dir / "tamanhos.json").write_text(json.dumps(tamanhos, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(destino_dir / "metricas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"passo": 0, "origem": "base-fp32", "loss_val": perda_antes, "tempo": agora_iso()}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"passo": 0, "origem": f"quantizado-{modo}", "loss_val": perda_depois, "tempo": agora_iso()}, ensure_ascii=False) + "\n")
    salvar_estado(destino_dir, tipo="quantizacao", modo=modo, fonte=base_dir.name, concluido=True)
    eventos.registrar("quantizacao_concluida", run=destino_dir.name, modo=modo, base=base_dir.name, fator=round(tamanhos["fator_alvos"], 2))
    return tamanhos
