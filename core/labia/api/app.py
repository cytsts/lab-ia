"""API interna (FastAPI) para a camada visual TS: runs, métricas, eventos, execuções.

A partir da bancada (B1/B2) a API também opera o ciclo de experimento: preparar
dados próprios, gerar config, comparar runs e desenhar curvas. É esse contrato que
a janela Electron consome — sem ele, o laboratório portátil só funcionaria por
linha de comando.

Notas de projeto:
  * /curvas.svg devolve SVG gerado no próprio núcleo, sem matplotlib: o pacote
    portátil não depende de biblioteca gráfica nenhuma para mostrar curva;
  * nada aqui aceita caminho arbitrário de configuração (só nomes dentro de
    configs/), e ids passam por validação de run-id/dataset;
  * /saude existe para a janela saber se o núcleo subiu, com o comando de subida
    na resposta quando não subiu.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ID_OK = re.compile(r"[A-Za-z0-9._-]+")
ACOES = {
    "train": lambda b, raiz: ["train", "--config", _config_segura(b.config, raiz), "--raiz", str(raiz)],
    "ajustar": lambda b, raiz: ["ajustar", "--config", _config_segura(b.config, raiz), "--raiz", str(raiz)],
}


def _validar_run_id(run_id: str) -> str:
    if not _ID_OK.fullmatch(run_id) or run_id in (".", ".."):
        raise HTTPException(status_code=400, detail="run-id inválido")
    return run_id


def _config_segura(nome: str, raiz: Path) -> str:
    if not nome or "/" in nome or "\\" in nome or ".." in nome:
        raise HTTPException(status_code=400, detail="config deve ser um nome de arquivo em configs/")
    caminho = (raiz / "configs" / nome).resolve()
    if not str(caminho).startswith(str((raiz / "configs").resolve())) or not caminho.suffix == ".yaml":
        raise HTTPException(status_code=400, detail="config fora de configs/ ou extensão inválida")
    if not caminho.exists():
        raise HTTPException(status_code=404, detail="config não encontrada")
    return str(caminho)


class ExecucaoCorpo(BaseModel):
    acao: str
    config: str


class DadosCorpo(BaseModel):
    """Corpo de POST /dados — preparo de corpus próprio (bancada B1)."""

    fontes: list[str] = Field(min_length=1)
    id: str
    frac_val: float = 0.05
    min_chars: int = 20
    quase_duplicata: bool = True
    limiar_quase: float = 0.8
    campo: str | None = None
    limite_mb: float = 256.0
    forcar: bool = False
    destino: str = "data"


class VarreduraCorpo(BaseModel):
    """Corpo de POST /varrer e /varrer/seco — varredura de hiperparâmetros (bancada B5)."""

    base: str
    grades: list[str] = Field(min_length=1)
    modo: str = "grade"
    n: int | None = None
    passos: int | None = None
    prefixo: str | None = None
    semente: int = 42


class NovoCorpo(BaseModel):
    """Corpo de POST /novo — geração de config comentada (bancada B1)."""

    nome: str
    dados: str
    preset: str = "equilibrado"
    sobrescritas: dict = Field(default_factory=dict)
    tokens_por_s: float | None = None
    forcar: bool = False
    saida: str | None = None


def criar_app(raiz: Path | str = ".") -> FastAPI:
    from ..bancada import comparar as bancada_comparar
    from ..bancada import curva as bancada_curva
    from ..bancada import dados as bancada_dados
    from ..bancada import presets as bancada_presets
    from ..bancada import varrer as bancada_varrer

    raiz = Path(raiz)
    app = FastAPI(title="Lab-IA API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    execucoes: dict[str, subprocess.Popen] = {}

    # --- núcleo e estado ----------------------------------------------------

    @app.get("/saude")
    def saude() -> dict:
        """Estado do núcleo: a janela usa isto para saber se o serviço está no ar."""
        import torch

        return {
            "ok": True,
            "versao_api": app.version,
            "raiz": str(raiz.resolve()),
            "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runs": len(bancada_comparar.listar_runs(raiz)),
            "datasets": len(bancada_dados.listar(raiz)),
            "configs": len(list((raiz / "configs").glob("*.yaml"))) if (raiz / "configs").is_dir() else 0,
            "guia": [p.name for p in sorted((raiz / "docs").glob("*.md"))] if (raiz / "docs").is_dir() else [],
        }

    @app.get("/guia/{nome}")
    def guia(nome: str) -> Response:
        """Devolve um guia em markdown de docs/ (o laboratório portátil leva o material junto)."""
        if not _ID_OK.fullmatch(nome.replace(".md", "")):
            raise HTTPException(status_code=400, detail="nome de guia inválido")
        caminho = (raiz / "docs" / f"{nome.removesuffix('.md')}.md").resolve()
        if not str(caminho).startswith(str((raiz / "docs").resolve())) or not caminho.exists():
            raise HTTPException(status_code=404, detail="guia não encontrado")
        return Response(content=caminho.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    # --- runs ---------------------------------------------------------------

    @app.get("/corre")
    def listar_runs() -> list[dict]:
        pasta = raiz / "runs"
        if not pasta.exists():
            return []
        saida = []
        for d in sorted(pasta.iterdir()):
            estado = d / "estado.json"
            if d.is_dir() and estado.exists():
                registro = json.loads(estado.read_text(encoding="utf-8"))
                registro["run_id"] = d.name
                saida.append(registro)
        return saida

    @app.get("/corre/{run_id}/metricas")
    def metricas(run_id: str) -> list[dict]:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "metricas.jsonl"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run ou métricas não encontrados")
        return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]

    @app.get("/corre/{run_id}/estado")
    def estado(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "estado.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run não encontrado")
        return json.loads(caminho.read_text(encoding="utf-8"))

    @app.get("/corre/{run_id}/tamanhos")
    def tamanhos(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "tamanhos.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run sem relatório de tamanhos")
        return json.loads(caminho.read_text(encoding="utf-8"))

    @app.get("/corre/{run_id}/comparativo")
    def comparativo(run_id: str) -> dict:
        caminho = raiz / "runs" / _validar_run_id(run_id) / "comparativo.json"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="run sem comparativo de estratégias")
        return json.loads(caminho.read_text(encoding="utf-8"))

    # --- bancada: dados e config -------------------------------------------

    @app.get("/datasets")
    def listar_datasets() -> list[dict]:
        return bancada_dados.listar(raiz)

    @app.post("/dados")
    def preparar_dados(corpo: DadosCorpo) -> dict:
        try:
            cfg = bancada_dados.ConfigPreparo(
                fontes=corpo.fontes,
                id=corpo.id,
                destino=corpo.destino,
                frac_val=corpo.frac_val,
                min_chars_paragrafo=corpo.min_chars,
                quase_duplicata=corpo.quase_duplicata,
                limiar_quase=corpo.limiar_quase,
                campo_jsonl=corpo.campo,
                limite_mb=corpo.limite_mb,
                forcar=corpo.forcar,
                raiz=str(raiz),
            )
            return bancada_dados.preparar(cfg)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/presets")
    def listar_presets() -> list[dict]:
        return [
            {
                "nome": nome,
                "descricao": valor["descricao"],
                "modelo": valor["modelo"],
                "passos": valor["passos"],
                "lote": valor["lote"],
            }
            for nome, valor in bancada_presets.PRESETS.items()
        ]

    @app.post("/novo")
    def gerar_config(corpo: NovoCorpo) -> dict:
        try:
            resultado = bancada_presets.gerar(
                nome=corpo.nome,
                dados=corpo.dados,
                preset=corpo.preset,
                destino=corpo.saida,
                raiz=raiz,
                tokens_por_s=corpo.tokens_por_s,
                forcar=corpo.forcar,
                sobrescritas=corpo.sobrescritas,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "caminho": resultado["caminho"],
            "caminho_relativo": resultado["caminho_relativo"],
            "arquivo_config": Path(resultado["caminho"]).name,
            "resumo": bancada_presets.resumo_texto(resultado),
            "config": resultado["config"],
            "parametros": resultado["parametros"],
            "desempenho": resultado["desempenho"],
            "vram": resultado["vram"],
            "epocas": resultado["epocas"],
            "yaml": resultado["texto"],
        }

    # --- bancada: comparação e curvas --------------------------------------

    def _runs_pedidos(runs: str | None) -> list[str]:
        if runs:
            pedidos = [r.strip() for r in runs.split(",") if r.strip()]
            if not pedidos:
                raise HTTPException(status_code=400, detail="lista de runs vazia")
            return pedidos
        achados = bancada_comparar.listar_runs(raiz)
        if not achados:
            raise HTTPException(status_code=404, detail="nenhum run com métricas em runs/")
        return achados

    @app.get("/comparar")
    def comparar(runs: str | None = Query(default=None, description="run-ids separados por vírgula")) -> dict:
        return bancada_comparar.comparar(_runs_pedidos(runs), raiz)

    @app.get("/comparar/relatorio")
    def relatorio_comparacao(runs: str | None = None) -> Response:
        resultado = bancada_comparar.comparar(_runs_pedidos(runs), raiz)
        return Response(
            content=bancada_comparar.relatorio_markdown(resultado),
            media_type="text/markdown; charset=utf-8",
        )

    @app.get("/comparar/curvas.svg")
    def curvas_svg(
        runs: str | None = None,
        metricas: str = "loss_val,loss_trem",
    ) -> Response:
        """Curvas em SVG geradas no núcleo — sem matplotlib, sem dependência gráfica."""
        pedidos = _runs_pedidos(runs)
        paineis = tuple(m.strip() for m in metricas.split(",") if m.strip())
        try:
            ativos = bancada_curva._paineis_com_dados(pedidos, raiz, paineis)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        svg = bancada_curva._svg(ativos, pedidos, "Curvas dos runs")
        return Response(content=svg, media_type="image/svg+xml")


    # --- bancada: varredura de hiperparâmetros (B5) -------------------------

    def _config_de_varredura(nome: str) -> str:
        return _config_segura(nome, raiz)

    def _validar_grades(brutos: list[str]) -> list[str]:
        """Aceita só 'chave=v1,v2' com caracteres seguros — o comando vira subprocesso."""
        limpos = []
        for bruto in brutos:
            texto = bruto.strip()
            if not texto or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.eE+,-]+", texto):
                raise HTTPException(status_code=400, detail=f"grade inválida: {bruto!r} (use chave=v1,v2)")
            limpos.append(texto)
        if not limpos:
            raise HTTPException(status_code=400, detail="informe pelo menos uma --grade chave=v1,v2")
        return limpos

    def _especificacao(corpo: "VarreduraCorpo"):
        from ..trainer.treino import ConfigTreino

        caminho = _config_de_varredura(corpo.base)
        try:
            cfg_base = ConfigTreino.de_arquivo(caminho)
            grades = bancada_varrer.interpretar_grades(_validar_grades(corpo.grades), cfg_base)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        prefixo = corpo.prefixo or f"varrer-{Path(corpo.base).stem}"
        if not _ID_OK.fullmatch(prefixo):
            raise HTTPException(status_code=400, detail="prefixo inválido (use letras, dígitos, ponto, hífen, _)")
        return caminho, grades, prefixo

    @app.post("/varrer/seco")
    def varrer_seco(corpo: "VarreduraCorpo") -> dict:
        """Confere a grade sem treinar nada: quais combinações saem e quais são impossíveis."""
        caminho, grades, prefixo = _especificacao(corpo)
        relatorio = bancada_varrer.executar(
            bancada_varrer.ConfigVarredura(
                base=caminho,
                grades=grades,
                prefixo=prefixo,
                modo=corpo.modo,
                n=corpo.n,
                passos=corpo.passos,
                semente=corpo.semente,
                raiz=str(raiz),
                seco=True,
            )
        )
        return {
            "prefixo": relatorio["prefixo"],
            "passos_por_variante": relatorio["passos_por_variante"],
            "variantes": relatorio["variantes"],
            "falhas": relatorio["falhas"],
            "tabela": bancada_varrer.tabela_texto(relatorio),
        }

    @app.post("/varrer")
    def varrer(corpo: "VarreduraCorpo") -> dict:
        """Dispara a varredura como processo próprio: minutos de treino não cabem numa resposta HTTP."""
        caminho, grades, prefixo = _especificacao(corpo)
        args = [sys.executable, "-m", "labia.cli", "varrer", "--base", caminho, "--prefixo", prefixo, "--raiz", str(raiz)]
        for grade in corpo.grades:
            args += ["--grade", _validar_grades([grade])[0]]
        if corpo.modo:
            args += ["--modo", corpo.modo]
        if corpo.n:
            args += ["--n", str(corpo.n)]
        if corpo.passos:
            args += ["--passos", str(corpo.passos)]
        args += ["--semente", str(corpo.semente)]
        log_dir = raiz / ".lab-ia" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chave = f"varrer-{prefixo}"
        log = open(log_dir / f"{chave}.log", "a", encoding="utf-8")
        processo = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(raiz))
        execucoes[chave] = processo
        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "varredura_iniciada", prefixo=prefixo, base=corpo.base, grades=corpo.grades, pid=processo.pid
        )
        return {"chave": chave, "pid": processo.pid, "prefixo": prefixo, "comando": args}

    @app.get("/varreduras")
    def listar_varreduras() -> list[dict]:
        pasta = raiz / "runs" / "_varredura"
        if not pasta.is_dir():
            return []
        saida = []
        for caminho in sorted(pasta.iterdir()):
            arquivo = caminho / "varredura.json"
            if arquivo.exists():
                dados = json.loads(arquivo.read_text(encoding="utf-8"))
                saida.append(
                    {
                        "prefixo": dados.get("prefixo"),
                        "base": dados.get("base"),
                        "criado_em": dados.get("criado_em"),
                        "melhor": dados.get("melhor"),
                        "variantes": len(dados.get("variantes", [])),
                        "falhas": dados.get("falhas", []),
                    }
                )
        return saida

    @app.get("/varreduras/{prefixo}")
    def ler_varredura(prefixo: str, markdown: bool = False):
        if not _ID_OK.fullmatch(prefixo):
            raise HTTPException(status_code=400, detail="prefixo inválido")
        arquivo = raiz / "runs" / "_varredura" / prefixo / "varredura.json"
        if not arquivo.exists():
            raise HTTPException(status_code=404, detail="varredura não encontrada")
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        if markdown:
            return Response(
                content=bancada_varrer.relatorio_markdown(dados),
                media_type="text/markdown; charset=utf-8",
            )
        dados["tabela"] = bancada_varrer.tabela_texto(dados)
        return dados

# --- trilha de estudo (B6) ---------------------------------------------

    @app.get("/trilha")
    def trilha_indice() -> dict:
        """Índice das lições — o laboratório portátil leva o material de estudo junto."""
        from .. import trilha as estudo

        indice = estudo.indice(raiz)
        problemas = estudo.conferir_referencias(raiz)
        return {**indice, "citacoes_quebradas": problemas}

    @app.get("/trilha/{numero}")
    def trilha_licao(numero: int) -> Response:
        from .. import trilha as estudo

        try:
            licao = estudo.achar_licao(numero, raiz)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return Response(content=licao.arquivo.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    @app.post("/trilha/{numero}/rodar")
    def trilha_rodar(numero: int) -> dict:
        """Roda o experimento da lição em processo próprio (minutos não cabem numa resposta HTTP)."""
        from .. import trilha as estudo

        try:
            licao = estudo.achar_licao(numero, raiz)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if licao.experimento is None or not licao.experimento.exists():
            raise HTTPException(status_code=404, detail=f"lição {numero} não tem experimento")
        log_dir = raiz / ".lab-ia" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chave = f"trilha-{numero:02d}"
        log = open(log_dir / f"{chave}.log", "a", encoding="utf-8")
        processo = subprocess.Popen(
            [sys.executable, str(licao.experimento)],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(raiz),
        )
        execucoes[chave] = processo
        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "trilha_experimento_iniciado", licao=numero, experimento=licao.experimento.name, pid=processo.pid
        )
        return {"chave": chave, "pid": processo.pid, "licao": numero, "experimento": licao.experimento.name}

    # --- execução -----------------------------------------------------------


    @app.get("/configs")
    def listar_configs() -> list[str]:
        pasta = raiz / "configs"
        if not pasta.exists():
            return []
        return sorted(p.name for p in pasta.glob("*.yaml"))

    @app.get("/eventos")
    def eventos(desde: int = 0) -> list[dict]:
        caminho = raiz / ".lab-ia" / "eventos.jsonl"
        if not caminho.exists():
            return []
        linhas = caminho.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in linhas[desde:] if l.strip()]

    @app.post("/execucao")
    def iniciar_execucao(corpo: ExecucaoCorpo) -> dict:
        if corpo.acao not in ACOES:
            raise HTTPException(status_code=400, detail=f"ação inválida (use {sorted(ACOES)})")
        args = [sys.executable, "-m", "labia.cli"] + ACOES[corpo.acao](corpo, raiz)
        log_dir = raiz / ".lab-ia" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chave = f"{corpo.acao}-{corpo.config}"
        log = open(log_dir / f"{chave}.log", "a", encoding="utf-8")
        processo = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(raiz))
        execucoes[chave] = processo
        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "execucao_iniciada", acao=corpo.acao, config=corpo.config, pid=processo.pid
        )
        return {"chave": chave, "pid": processo.pid, "comando": args}

    @app.get("/execucoes")
    def listar_execucoes() -> list[dict]:
        saida = []
        for chave, proc in execucoes.items():
            status = proc.poll()
            saida.append({"chave": chave, "pid": proc.pid, "vivo": status is None, "codigo_saida": status})
        return saida

    @app.get("/execucoes/{chave}/log")
    def log_execucao(chave: str, linhas: int = 60) -> dict:
        """Cauda do log: a janela mostra o treino acontecendo em vez de só um PID."""
        if not _ID_OK.fullmatch(chave.replace("-", "").replace(".", "")):
            raise HTTPException(status_code=400, detail="chave inválida")
        caminho = raiz / ".lab-ia" / "logs" / f"{chave}.log"
        if not caminho.exists():
            return {"chave": chave, "linhas": [], "existe": False}
        conteudo = caminho.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"chave": chave, "linhas": conteudo[-max(1, linhas) :], "existe": True, "total": len(conteudo)}

    return app
