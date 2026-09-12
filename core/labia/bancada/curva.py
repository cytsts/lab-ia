"""Curvas dos runs: ver tendência em vez de número solto.

Número isolado engana: "val 4,70" não diz se o modelo ainda está caindo, se passou
do ponto ou se oscila. Um gráfico de perda por passo mostra isso em um segundo — e
era o instrumento que faltava para otimizar modelo no laboratório.

Duas saídas, de propósito:

* SVG (padrão) — gerado aqui mesmo, sem biblioteca nenhuma. Funciona offline, abre
  no navegador, no editor e dentro do app Electron.
* PNG — usa matplotlib quando ele está instalado (extra 'viz' do pyproject), porque
  relatório em PNG é o que a maioria espera colar em algum lugar.

A dependência opcional não pode virar dependência obrigatória: o laboratório roda
em máquina sem rede, e uma curva que não sai é pior que uma curva sem antialias.
"""
from __future__ import annotations

from pathlib import Path

from ..experiments.runner import ler_metricas

PALETA = (
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#4d7c0f", "#9333ea", "#b45309",
)

PAINEIS_PADRAO = ("loss_val", "loss_trem", "lr", "tokens_por_s")

ROTULOS = {
    "loss_val": "perda na validação (generalização)",
    "loss_trem": "perda no treino (memorização)",
    "lr": "learning rate",
    "tokens_por_s": "velocidade (tokens/s)",
}


def coletar_series(runs: list[str], raiz: Path | str = ".", metrica: str = "loss_val") -> list[dict]:
    """Lê metricas.jsonl de cada run e devolve as séries da métrica pedida."""
    series = []
    for indice, run_id in enumerate(runs):
        registros = ler_metricas(Path(raiz) / "runs" / run_id)
        pontos = [
            (int(r["passo"]), float(r[metrica]))
            for r in registros
            if r.get(metrica) is not None and r.get("passo") is not None
        ]
        if pontos:
            series.append({"run_id": run_id, "cor": PALETA[indice % len(PALETA)], "pontos": pontos})
    return series


def _paineis_com_dados(runs: list[str], raiz: Path | str, paineis: tuple[str, ...]) -> list[tuple[str, list[dict]]]:
    validos = tuple(p for p in paineis if p in ROTULOS)
    if not validos:
        raise ValueError(f"nenhum painel válido em {paineis}; use {sorted(ROTULOS)}")
    ativos = []
    for metrica in validos:
        series = coletar_series(runs, raiz, metrica)
        if series:
            ativos.append((metrica, series))
    if not ativos:
        raise ValueError(
            f"nenhuma métrica encontrada nos runs {runs} — eles têm metricas.jsonl com valores?"
        )
    return ativos


def desenhar(
    runs: list[str],
    raiz: Path | str = ".",
    saida: Path | str = "curvas.svg",
    paineis: tuple[str, ...] = PAINEIS_PADRAO,
    titulo: str | None = None,
    dpi: int = 140,
) -> Path:
    """Desenha um painel por métrica, com todos os runs sobrepostos. Devolve o caminho.

    A extensão decide o renderizador: .svg sai sem dependência nenhuma; .png usa
    matplotlib (instale com 'pip install matplotlib').
    """
    ativos = _paineis_com_dados(runs, raiz, paineis)
    destino = Path(saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.suffix.lower() == ".svg":
        destino.write_text(_svg(ativos, runs, titulo), encoding="utf-8")
        return destino
    _desenhar_matplotlib(ativos, destino, titulo, dpi)
    return destino


# --- SVG sem dependências ---------------------------------------------------

LARGURA = 900
ALTURA_PAINEL = 190
MARGEM = {"esq": 78, "dir": 150, "topo": 26, "base": 30}


def _svg(ativos: list[tuple[str, list[dict]]], runs: list[str], titulo: str | None) -> str:
    altura = MARGEM["topo"] + ALTURA_PAINEL * len(ativos) + MARGEM["base"]
    partes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{LARGURA}" height="{altura}" '
        f'viewBox="0 0 {LARGURA} {altura}" font-family="Segoe UI, system-ui, sans-serif">',
        f'<rect width="{LARGURA}" height="{altura}" fill="#ffffff"/>',
        f'<text x="{MARGEM["esq"]}" y="17" font-size="13" font-weight="600" fill="#111827">'
        f"{_escapar(titulo or 'Curvas dos runs')}</text>",
    ]
    for indice, (metrica, series) in enumerate(ativos):
        partes.append(_painel(metrica, series, MARGEM["topo"] + indice * ALTURA_PAINEL))
    partes.append(_legenda(runs, altura))
    partes.append("</svg>")
    return "\n".join(partes) + "\n"


def _painel(metrica: str, series: list[dict], topo: int) -> str:
    x0, x1 = MARGEM["esq"], LARGURA - MARGEM["dir"]
    y0, y1 = topo, topo + ALTURA_PAINEL - MARGEM["base"] - 6
    pontos = [p for s in series for p in s["pontos"]]
    xs = [p[0] for p in pontos]
    ys = [p[1] for p in pontos]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max = y_min + 1.0
    span_x = (x_max - x_min) or 1
    span_y = y_max - y_min

    def px(x: float) -> float:
        return x0 + (x - x_min) / span_x * (x1 - x0)

    def py(y: float) -> float:
        return y1 - (y - y_min) / span_y * (y1 - y0)

    partes = [
        f'<text x="{x0}" y="{topo + 10}" font-size="11" font-weight="600" fill="#374151">'
        f"{_escapar(ROTULOS[metrica])}</text>",
        f'<rect x="{x0}" y="{y0 + 12}" width="{x1 - x0}" height="{y1 - y0 - 12}" fill="#fafafa" '
        f'stroke="#e5e7eb" stroke-width="1"/>',
    ]
    for fracao in (0.0, 0.5, 1.0):
        valor = y_min + fracao * span_y
        y = py(valor)
        partes.append(
            f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        partes.append(
            f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="10" fill="#6b7280" text-anchor="end">'
            f"{_numero(valor)}</text>"
        )
    for fracao in (0.0, 0.5, 1.0):
        valor = x_min + fracao * span_x
        x = px(valor)
        partes.append(
            f'<text x="{x:.1f}" y="{y1 + 16:.1f}" font-size="10" fill="#6b7280" text-anchor="middle">'
            f"{int(valor)}</text>"
        )
    for serie in series:
        caminho = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in serie["pontos"])
        partes.append(
            f'<polyline points="{caminho}" fill="none" stroke="{serie["cor"]}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    return "\n".join(partes)


def _legenda(runs: list[str], altura: int) -> str:
    partes = []
    for indice, run_id in enumerate(runs):
        y = MARGEM["topo"] + 14 + indice * 18
        cor = PALETA[indice % len(PALETA)]
        partes.append(f'<line x1="{LARGURA - MARGEM["dir"] + 10}" x2="{LARGURA - MARGEM["dir"] + 34}" '
                      f'y1="{y}" y2="{y}" stroke="{cor}" stroke-width="3"/>')
        partes.append(
            f'<text x="{LARGURA - MARGEM["dir"] + 40}" y="{y + 4}" font-size="11" fill="#374151">'
            f"{_escapar(run_id)}</text>"
        )
    return "\n".join(partes)


def _numero(valor: float) -> str:
    if abs(valor) >= 10_000:
        return f"{valor / 1000:.0f}k"
    if abs(valor) >= 100:
        return f"{valor:.0f}"
    if abs(valor) >= 1:
        return f"{valor:.2f}".rstrip("0").rstrip(".")
    return f"{valor:.4f}"


def _escapar(texto: str) -> str:
    return (
        str(texto)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --- PNG com matplotlib (opcional) ------------------------------------------

def _desenhar_matplotlib(ativos, destino: Path, titulo: str | None, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "matplotlib não está instalado (necessário só para PNG). "
            "Instale com '.venv\\Scripts\\pip install matplotlib' "
            "ou peça saída .svg, que não precisa de nada."
        ) from e

    figura, eixos = plt.subplots(len(ativos), 1, figsize=(9, 2.6 * len(ativos)), sharex=True, squeeze=False)
    figura.suptitle(titulo or "Curvas dos runs", fontsize=12, ha="left", x=0.01)
    for eixo, (metrica, series) in zip(eixos[:, 0], ativos):
        for serie in series:
            eixo.plot(
                [p[0] for p in serie["pontos"]],
                [p[1] for p in serie["pontos"]],
                label=serie["run_id"],
                color=serie["cor"],
                linewidth=1.8,
            )
        eixo.set_ylabel(ROTULOS[metrica], fontsize=9)
        eixo.grid(alpha=0.25, linewidth=0.6)
        eixo.tick_params(labelsize=8)
        if metrica == "loss_val":
            eixo.legend(fontsize=8, ncols=2, loc="upper right")
    eixos[-1, 0].set_xlabel("passo", fontsize=9)
    figura.tight_layout(rect=(0, 0, 1, 0.97))
    figura.savefig(destino, dpi=dpi)
    plt.close(figura)


def dados_para_ui(runs: list[str], raiz: Path | str = ".", metricas: tuple[str, ...] = PAINEIS_PADRAO) -> dict:
    """Séries prontas para a interface (não depende de matplotlib nem de SVG)."""
    return {
        "runs": list(runs),
        "paineis": [
            {"metrica": m, "rotulo": ROTULOS[m], "series": coletar_series(runs, raiz, m)}
            for m in metricas
            if m in ROTULOS
        ],
    }
