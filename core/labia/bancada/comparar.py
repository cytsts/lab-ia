"""Comparação de runs: medir o que aconteceu e dizer o que fazer a seguir.

Sem esta camada, otimizar modelo no laboratório significava abrir metricas.jsonl na
mão e comparar de cabeça. Aqui a leitura vira diagnóstico:

  * melhor validação e em que passo ela aconteceu;
  * deriva (drift) = validação final - melhor validação → sinal de overfit;
  * ganho sobre a entropia unigram → o modelo aprendeu estrutura ou só frequência;
  * validação no orçamento comum → comparação justa entre runs de tamanhos diferentes;
  * estabilidade (saltos e valores não finitos).

Cada número sai de métricas já gravadas pelo treino; nada é recalculado por fora.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..experiments.runner import carregar_estado, ler_metricas

# Famílias de run: perdas de famílias diferentes NÃO são comparáveis entre si.
# treino = do zero; lora = ajuste sobre uma base; quantizacao = run derivado, não treina.
FAMILIAS = {"treino": "treino do zero", "lora": "ajuste LoRA/QLoRA", "quantizacao": "quantização"}

LIMIAR_DRIFT_OVERFIT = 0.2  # mesma barra da CA1b da G1 (specs/PLANO.md)
LIMIAR_ESTAGNACAO = 0.01  # ganho por avaliação abaixo disso = platô
LIMIAR_SALTO = 1.0  # queda brusca de validação entre avaliações = instabilidade


@dataclass
class ResumoRun:
    """Retrato de um run a partir de estado.json + metricas.jsonl."""

    run_id: str
    existe: bool = False
    concluido: bool = False
    familia: str = "treino"
    tipo: str | None = None
    base: str | None = None
    passos: int = 0
    passos_totais: int = 0
    avaliacoes: int = 0
    melhor_val: float | None = None
    passo_melhor_val: int | None = None
    val_final: float | None = None
    trem_final: float | None = None
    drift: float | None = None
    entropia_unigram: float | None = None
    ganho_vs_unigram: float | None = None
    tokens_por_s: float | None = None
    tempo_s: float | None = None
    dispositivo: str | None = None
    origem_dados: str | None = None
    parametros: int | None = None
    parametros_ativos: int | None = None
    tokens_vistos: int | None = None
    epocas: float | None = None
    lote: int | None = None
    janela: int | None = None
    curva: list[dict] = field(default_factory=list)
    melhor_val_orcamento_comum: float | None = None
    diagnostico: str = "sem_dados"
    comentario: str = ""
    instavel: bool = False

    def para_dict(self) -> dict:
        dados = {k: v for k, v in self.__dict__.items() if k != "curva"}
        return dados


def listar_runs(raiz: Path | str = ".", com_metricas: bool = True) -> list[str]:
    """Run-ids presentes em runs/, ordenados; por padrão só os que têm métricas."""
    pasta = Path(raiz) / "runs"
    if not pasta.is_dir():
        return []
    achados = []
    for caminho in sorted(pasta.iterdir()):
        if not caminho.is_dir() or not (caminho / "estado.json").exists():
            continue
        if com_metricas and not (caminho / "metricas.jsonl").exists():
            continue
        achados.append(caminho.name)
    return achados


def resumir_run(run_id: str, raiz: Path | str = ".", janela_estagnacao: int = 3) -> ResumoRun:
    """Lê um run do disco e devolve o retrato com diagnóstico textual."""
    pasta = Path(raiz) / "runs" / run_id
    resumo = ResumoRun(run_id=run_id, existe=pasta.is_dir())
    if not resumo.existe:
        resumo.diagnostico = "ausente"
        resumo.comentario = f"run {run_id!r} não existe em runs/"
        return resumo

    estado = carregar_estado(pasta) or {}
    resumo.concluido = bool(estado.get("concluido"))
    resumo.tipo = estado.get("tipo")
    resumo.familia = resumo.tipo or "treino"
    resumo.base = estado.get("base")
    resumo.passos = int(estado.get("passo") or 0)
    resumo.passos_totais = int(estado.get("passos_totais") or 0)
    resumo.tempo_s = estado.get("tempo_decorrido_s")
    resumo.dispositivo = estado.get("dispositivo")
    resumo.origem_dados = estado.get("origem_dados")

    cfg_modelo = estado.get("config_modelo") or {}
    cfg_treino = estado.get("config_treino") or {}
    resumo.lote = cfg_treino.get("lote")
    resumo.janela = cfg_modelo.get("janela_ctx")
    if cfg_modelo:
        from .presets import contar_parametros  # import tardio: presets importa dados

        params = contar_parametros(
            cfg_modelo.get("vocab", 0),
            cfg_modelo.get("dim", 0),
            cfg_modelo.get("camadas", 0),
            cfg_modelo.get("janela_ctx", 0),
            cfg_modelo.get("n_especialistas", 0),
            cfg_modelo.get("top_k", 1),
            norm=cfg_modelo.get("norm", "layernorm"),
            pos=cfg_modelo.get("pos", "aprendido"),
            mlp=cfg_modelo.get("mlp", "gelu"),
            cabecas=cfg_modelo.get("cabecas", 0),
            n_cabecas_kv=cfg_modelo.get("n_cabecas_kv", 0),
        )
        resumo.parametros = params["total"]
        resumo.parametros_ativos = params["ativos"]
    if resumo.lote and resumo.janela and resumo.passos:
        resumo.tokens_vistos = resumo.lote * resumo.janela * resumo.passos

    registros = ler_metricas(pasta)
    if not registros:
        resumo.diagnostico = "sem_metricas"
        resumo.comentario = "run sem metricas.jsonl (ou vazio) — não há o que comparar"
        return resumo
    # Run derivado (quantização): as perdas continuam úteis na tabela, mas curva de
    # treino não diz nada sobre aprendizado — ele não treina.
    eh_quantizacao = resumo.familia == "quantizacao"

    resumo.avaliacoes = len([r for r in registros if r.get("loss_val") is not None])
    resumo.entropia_unigram = next(
        (float(r["entropia_unigram"]) for r in registros if r.get("entropia_unigram") is not None), None
    )
    resumo.curva = [
        {
            "passo": r.get("passo"),
            "loss_trem": r.get("loss_trem"),
            "loss_val": r.get("loss_val"),
            "lr": r.get("lr"),
            "tokens_por_s": r.get("tokens_por_s"),
        }
        for r in registros
    ]

    com_val = [r for r in registros if r.get("loss_val") is not None]
    finitos = [r for r in com_val if math.isfinite(float(r["loss_val"]))]
    nao_finitos = len(com_val) - len(finitos)
    resumo.instavel = bool(nao_finitos) or _tem_salto(com_val)
    # Divergência é o colapso (metade ou mais das avaliações inválidas), não um pico isolado:
    # um NaN solto no meio de um treino saudável é instabilidade, não fim de linha.
    metade = max(1, math.ceil(len(com_val) / 2))
    if not eh_quantizacao and (not finitos or nao_finitos >= metade):
        resumo.diagnostico = "divergiu"
        resumo.comentario = (
            f"{nao_finitos} de {len(com_val)} avaliações de validação não são finitas — o treino divergiu. "
            "Reduza --lr, aumente --warmup e verifique se o corpus tem lixo (linhas gigantes, repetição)."
        )
        return resumo

    if not finitos:
        resumo.diagnostico = "quantizacao" if eh_quantizacao else "sem_metricas"
        resumo.comentario = (
            _comentario_quantizacao(pasta, estado)
            if eh_quantizacao
            else "run sem nenhum valor de validação finito — não há o que comparar"
        )
        return resumo

    melhor = min(finitos, key=lambda r: float(r["loss_val"]))
    resumo.melhor_val = round(float(melhor["loss_val"]), 4)
    resumo.passo_melhor_val = int(melhor["passo"])
    resumo.val_final = round(float(finitos[-1]["loss_val"]), 4)
    if finitos[-1].get("loss_trem") is not None:
        resumo.trem_final = round(float(finitos[-1]["loss_trem"]), 4)
    resumo.drift = round(resumo.val_final - resumo.melhor_val, 4)
    if resumo.entropia_unigram is not None:
        resumo.ganho_vs_unigram = round(resumo.entropia_unigram - resumo.melhor_val, 4)
    taxas = [float(r["tokens_por_s"]) for r in registros if r.get("tokens_por_s")]
    if taxas:
        resumo.tokens_por_s = round(sum(taxas) / len(taxas))
    if resumo.tokens_vistos and resumo.tempo_s:
        passos_por_epoca = None

    if eh_quantizacao:
        resumo.diagnostico = "quantizacao"
        resumo.comentario = _comentario_quantizacao(pasta, estado)
        return resumo
    resumo.diagnostico, resumo.comentario = _diagnosticar(resumo, finitos, janela_estagnacao)
    return resumo


def _comentario_quantizacao(pasta: Path, estado: dict) -> str:
    """Quantização se avalia por tamanho e perda, não por curva."""
    arquivo = pasta / "tamanhos.json"
    modo = estado.get("modo", "?")
    if not arquivo.exists():
        return f"run de quantização ({modo}): sem tamanhos.json — só o estado foi gravado."
    dados = json.loads(arquivo.read_text(encoding="utf-8"))
    return (
        f"run de quantização {modo} sobre {estado.get('fonte', '?')}: "
        f"fator {dados.get('fator_alvos', float('nan')):.2f}x nos alvos, "
        f"bits efetivos {dados.get('bits_efetivos_por_parametro', '?')}, "
        f"perda {dados.get('perda_val_antes', float('nan')):.3f} -> {dados.get('perda_val_depois', float('nan')):.3f}. "
        "Não treina: compare tamanho e perda, não curva."
    )


def _tem_salto(registros: list[dict]) -> bool:
    """Salto grande entre avaliações consecutivas indica lr alta, dado ruim ou divergência.

    O passo 0 é o modelo ALEATÓRIO: a queda dele para a primeira avaliação treinada é
    sempre grande (foi 2,0 nats no run livros-rapido) e não significa instabilidade.
    Por isso a checagem começa depois da linha de base.
    """
    valores = [
        float(r["loss_val"])
        for r in registros
        if math.isfinite(float(r["loss_val"])) and int(r.get("passo") or 0) > 0
    ]
    return any(abs(b - a) > LIMIAR_SALTO for a, b in zip(valores, valores[1:]))


def _diagnosticar(resumo: ResumoRun, finitos: list[dict], janela_estagnacao: int) -> tuple[str, str]:
    if resumo.drift is not None and resumo.drift > LIMIAR_DRIFT_OVERFIT:
        return (
            "overfit",
            (
                f"a validação MELHOROU até o passo {resumo.passo_melhor_val} ({resumo.melhor_val}) "
                f"e depois PIOROU até {resumo.val_final} (deriva +{resumo.drift} > {LIMIAR_DRIFT_OVERFIT}). "
                "O modelo passou a decorar o corpus: pare no passo do mínimo, aumente o corpus, "
                "suba --abandono ou reduza --passos."
            ),
        )
    ultimas = finitos[-(janela_estagnacao + 1) :]
    if len(ultimas) > janela_estagnacao:
        ganho = float(ultimas[0]["loss_val"]) - float(ultimas[-1]["loss_val"])
        if ganho < -LIMIAR_ESTAGNACAO:
            return (
                "deteriorando",
                (
                    f"a validação subiu {abs(ganho):.4f} nas últimas {janela_estagnacao} avaliações — "
                    "já passou do ponto. Retome do checkpoint do mínimo, use menos passos ou mais dados."
                ),
            )
        if ganho < LIMIAR_ESTAGNACAO:
            return (
                "estagnado",
                (
                    f"ganho de apenas {ganho:.4f} nas últimas {janela_estagnacao} avaliações — platô. "
                    "Parte disso é o decaimento cosseno do --lr chegando ao piso; se não for o caso, "
                    "mude algo estrutural (--dim, --camadas, --lr, dados) em vez de somar passos."
                ),
            )
    if resumo.instavel:
        return (
            "instavel",
            "houve salto grande ou valor não finito entre avaliações — reduza --lr, aumente --warmup ou revise os dados.",
        )
    if resumo.passo_melhor_val is not None and resumo.passo_melhor_val >= resumo.passos:
        return (
            "ainda_caindo",
            (
                f"o melhor ponto é a ÚLTIMA avaliação (passo {resumo.passo_melhor_val}, val {resumo.melhor_val}): "
                "o treino não convergiu — mais passos devem melhorar."
            ),
        )
    return (
        "estavel",
        (
            f"mínimo em {resumo.melhor_val} no passo {resumo.passo_melhor_val}, "
            f"fechou em {resumo.val_final} (deriva {resumo.drift:+}). Dentro da barra de {LIMIAR_DRIFT_OVERFIT}."
        ),
    )


def _melhor_val_ate(resumo: ResumoRun, orcamento: int) -> float | None:
    """Melhor validação dentro de um orçamento de passos — comparação justa entre runs."""
    valores = [
        float(p["loss_val"])
        for p in resumo.curva
        if p.get("loss_val") is not None and p.get("passo") is not None and int(p["passo"]) <= orcamento
    ]
    return round(min(valores), 4) if valores else None


def comparar(runs: list[str], raiz: Path | str = ".") -> dict:
    """Compara N runs e devolve resumos, orçamento comum, ranking e veredito."""
    resumos = [resumir_run(run_id, raiz) for run_id in runs]
    # só runs treinados entram no ranking: quantização é derivada e não aprende nada
    treinados = [r for r in resumos if r.melhor_val is not None and r.familia != "quantizacao"]
    orcamento = min((r.passos for r in treinados if r.passos > 0), default=None)
    for resumo in treinados:
        if orcamento is not None and resumo.passos > 0:
            resumo.melhor_val_orcamento_comum = _melhor_val_ate(resumo, orcamento)
    ranking = sorted(treinados, key=lambda r: r.melhor_val) if treinados else []
    familias = sorted({r.familia for r in resumos if r.melhor_val is not None})
    return {
        "runs": [r.para_dict() for r in resumos],
        "orcamento_comum": orcamento,
        "familias": familias,
        "ranking": [r.run_id for r in ranking],
        "melhor": ranking[0].run_id if ranking else None,
        "veredito": _veredito(ranking, orcamento, familias),
        "limiares": {
            "drift_overfit": LIMIAR_DRIFT_OVERFIT,
            "estagnacao": LIMIAR_ESTAGNACAO,
            "salto_instavel": LIMIAR_SALTO,
        },
    }


def _veredito(ranking: list[ResumoRun], orcamento: int | None, familias: list[str] | None = None) -> str:
    if not ranking:
        return "nenhum run treinado com métricas de validação: nada a comparar."
    aviso_familias = ""
    if familias and len(familias) > 1:
        aviso_familias = (
            f" ATENÇÃO: há famílias diferentes na lista ({', '.join(familias)}) — "
            "perda de ajuste LoRA não é comparável com perda de treino do zero; compare dentro da mesma família."
        )
    if len(ranking) == 1:
        unico = ranking[0]
        return f"um único run treinado ({unico.run_id}): {unico.comentario}.{aviso_familias}"
    melhor = ranking[0]
    segundo = ranking[1]
    diferenca = round(segundo.melhor_val - melhor.melhor_val, 4)
    linha = (
        f"{melhor.run_id} generaliza melhor: val {melhor.melhor_val} contra {segundo.melhor_val} "
        f"de {segundo.run_id} (diferença de {diferenca})."
    )
    if orcamento is not None:
        comum = [
            (r.melhor_val_orcamento_comum, r.run_id)
            for r in ranking
            if r.melhor_val_orcamento_comum is not None
        ]
        if comum:
            comum_ordenado = sorted(comum)
            if comum_ordenado[0][1] != melhor.run_id:
                linha += (
                    f" ATENÇÃO: no orçamento comum de {orcamento} passos quem lidera é "
                    f"{comum_ordenado[0][1]} ({comum_ordenado[0][0]}) — a vantagem do primeiro vem de treinar mais tempo."
                )
    return linha + aviso_familias


def tabela_texto(resultado: dict) -> str:
    """Tabela alinhada; colunas com dado ausente saem como '?' em vez de sumir."""
    cabecalho = (
        f"{'run':<20} {'família':<10} {'passos':>7} {'params':>10} {'melhor val':>10} {'passo':>7} "
        f"{'final':>8} {'deriva':>8} {'val@comum':>10} {'tok/s':>9} {'veredito':<12}"
    )
    linhas = [cabecalho, "-" * len(cabecalho)]
    for r in resultado["runs"]:
        linhas.append(
            f"{r['run_id']:<20} "
            f"{r.get('familia', 'treino'):<10} "
            f"{r['passos'] if r['passos'] else '?':>7} "
            f"{_fmt_int(r.get('parametros')):>10} "
            f"{_fmt(r.get('melhor_val')):>10} "
            f"{r.get('passo_melhor_val') if r.get('passo_melhor_val') is not None else '?':>7} "
            f"{_fmt(r.get('val_final')):>8} "
            f"{_fmt_sinal(r.get('drift')):>8} "
            f"{_fmt(r.get('melhor_val_orcamento_comum')):>10} "
            f"{_fmt_int(r.get('tokens_por_s')):>9} "
            f"{r['diagnostico']:<12}"
        )
    linhas.append("")
    for r in resultado["runs"]:
        if r.get("comentario"):
            linhas.append(f"  {r['run_id']}: {r['comentario']}")
    linhas.append("")
    linhas.append(f"veredito: {resultado['veredito']}")
    return "\n".join(linhas)


def _fmt(valor) -> str:
    return "?" if valor is None else f"{valor:.4f}"


def _fmt_sinal(valor) -> str:
    return "?" if valor is None else f"{valor:+.4f}"


def _fmt_int(valor) -> str:
    return "?" if valor is None else f"{valor:,}".replace(",", ".")


def relatorio_markdown(resultado: dict) -> str:
    """Relatório em markdown para guardar junto do experimento."""
    linhas = [
        "# Comparação de runs",
        "",
        "| run | família | passos | params | melhor val | passo | final | deriva | val@comum | veredito |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resultado["runs"]:
        linhas.append(
            f"| {r['run_id']} | {r.get('familia', 'treino')} | {r['passos'] or '?'} | {_fmt_int(r.get('parametros'))} | "
            f"{_fmt(r.get('melhor_val'))} | {r.get('passo_melhor_val') or '?'} | {_fmt(r.get('val_final'))} | "
            f"{_fmt_sinal(r.get('drift'))} | {_fmt(r.get('melhor_val_orcamento_comum'))} | {r['diagnostico']} |"
        )
    linhas.append("")
    if resultado.get("orcamento_comum"):
        linhas.append(
            f"Orçamento comum: {resultado['orcamento_comum']} passos "
            "(coluna val@comum — comparação justa entre runs de durações diferentes)."
        )
        linhas.append("")
    linhas.append(f"**Veredito:** {resultado['veredito']}")
    linhas.append("")
    linhas.append("## Diagnóstico por run")
    for r in resultado["runs"]:
        if r.get("comentario"):
            linhas.append(f"- **{r['run_id']}** ({r['diagnostico']}): {r['comentario']}")
    limiares = resultado["limiares"]
    linhas += [
        "",
        "## Critérios usados",
        f"- deriva acima de {limiares['drift_overfit']} entre a melhor validação e a final = overfit;",
        f"- ganho abaixo de {limiares['estagnacao']} nas últimas avaliações = platô;",
        f"- salto acima de {limiares['salto_instavel']} entre avaliações consecutivas = instabilidade.",
    ]
    return "\n".join(linhas) + "\n"


def salvar_relatorio(resultado: dict, destino: Path | str) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(relatorio_markdown(resultado), encoding="utf-8")
    return destino


def json_serializavel(resultado: dict) -> str:
    return json.dumps(resultado, ensure_ascii=False, indent=1, default=str)
