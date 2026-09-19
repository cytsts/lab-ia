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


class TestarCorpo(BaseModel):
    """Corpo de POST /testar — inferência sob demanda e exploração de CoT (B13 RF5/RF7)."""

    run: str
    enunciado: str
    estrategia: str = "cot"
    temperatura: float = 0.7
    guloso: bool = True
    max_tokens: int = 256
    prefixo: str | None = None
    semente: int = 1234


class BenchmarkCorpo(BaseModel):
    """Corpo de POST /benchmark — disparo de medição como execução controlada (B13 RF6)."""

    run: str
    estrategia: str = "cot"
    benchmark: str = "data/logica-pq/benchmark.jsonl"
    split: str = "teste"
    limite: int | None = 60
    semente: int = 1234


def _ler_cauda_linhas(caminho: Path, n_linhas: int = 60, bloco_bytes: int = 8192) -> list[str]:
    """Lê eficientemente as últimas n_linhas do arquivo a partir do fim (CA12)."""
    if not caminho.exists():
        return []
    tamanho = caminho.stat().st_size
    if tamanho == 0:
        return []
    if tamanho < bloco_bytes * 2:
        return caminho.read_text(encoding="utf-8", errors="replace").splitlines()[-n_linhas:]

    with open(caminho, "rb") as f:
        linhas = []
        pos = tamanho
        sobra = b""
        while pos > 0 and len(linhas) <= n_linhas:
            ler = min(pos, bloco_bytes)
            pos -= ler
            f.seek(pos)
            pedaco = f.read(ler) + sobra
            pedacos = pedaco.split(b"\n")
            sobra = pedacos[0]
            for linha in reversed(pedacos[1:]):
                linhas.append(linha.decode("utf-8", errors="replace"))
                if len(linhas) >= n_linhas:
                    break
        if sobra and len(linhas) < n_linhas:
            linhas.append(sobra.decode("utf-8", errors="replace"))
        return list(reversed(linhas[:n_linhas]))


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

    @app.get("/corre/{run_id}/progresso")
    def progresso_run(run_id: str) -> dict:
        _validar_run_id(run_id)
        pasta = raiz / "runs" / run_id
        if not pasta.exists():
            raise HTTPException(status_code=404, detail="run não encontrado")

        estado_arquivo = pasta / "estado.json"
        estado = {}
        if estado_arquivo.exists():
            try:
                estado = json.loads(estado_arquivo.read_text(encoding="utf-8"))
            except Exception:
                pass

        passo_atual = estado.get("passo", 0)
        passos_totais = estado.get("passos_totais") or estado.get("passos", 0)
        concluido = bool(estado.get("concluido", False))

        metricas_arquivo = pasta / "metricas.jsonl"
        ultimas_linhas = _ler_cauda_linhas(metricas_arquivo, n_linhas=40)

        sparkline = []
        ultima_loss_trem = None
        ultima_loss_val = None
        ultimo_lr = None
        ultimo_tokens_s = None
        tempo_s = None

        for linha in ultimas_linhas:
            try:
                m = json.loads(linha)
            except Exception:
                continue
            if m.get("tipo") == "treino" or "loss_trem" in m or "loss_val" in m:
                if m.get("passo") is not None:
                    sparkline.append({
                        "passo": m["passo"],
                        "loss_trem": m.get("loss_trem"),
                        "loss_val": m.get("loss_val"),
                    })
                if m.get("loss_trem") is not None:
                    ultima_loss_trem = m["loss_trem"]
                if m.get("loss_val") is not None:
                    ultima_loss_val = m["loss_val"]
                if m.get("lr") is not None:
                    ultimo_lr = m["lr"]
                if m.get("tokens_por_s") is not None:
                    ultimo_tokens_s = m["tokens_por_s"]
                if m.get("tempo_s") is not None:
                    tempo_s = m["tempo_s"]
                if m.get("passo") is not None:
                    passo_atual = m["passo"]

        tempo_restante_s = None
        if passos_totais and passo_atual and tempo_s and passo_atual < passos_totais:
            segundos_por_passo = tempo_s / max(1, passo_atual)
            tempo_restante_s = round(segundos_por_passo * (passos_totais - passo_atual), 1)

        bm_prog_arquivo = pasta / "benchmark_progresso.json"
        benchmark_progresso = None
        if bm_prog_arquivo.exists():
            try:
                benchmark_progresso = json.loads(bm_prog_arquivo.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "run_id": run_id,
            "passo": passo_atual,
            "passos_totais": passos_totais,
            "concluido": concluido,
            "loss_trem": ultima_loss_trem,
            "loss_val": ultima_loss_val,
            "lr": ultimo_lr,
            "tokens_por_s": ultimo_tokens_s,
            "tempo_s": tempo_s,
            "tempo_restante_s": tempo_restante_s,
            "sparkline": sparkline[-20:],
            "benchmark_progresso": benchmark_progresso,
        }

    @app.get("/corre/{run_id}/benchmarks")
    def benchmarks_do_run(run_id: str) -> list[dict]:
        _validar_run_id(run_id)
        pasta = raiz / "runs" / run_id
        if not pasta.exists():
            raise HTTPException(status_code=404, detail="run não encontrado")
        relatorios = []
        for arq in sorted(pasta.glob("benchmark-*.json")):
            try:
                conteudo = json.loads(arq.read_text(encoding="utf-8"))
                conteudo["arquivo"] = arq.name
                relatorios.append(conteudo)
            except Exception:
                continue
        return relatorios

    # --- bancada: dados e config -------------------------------------------

    @app.get("/datasets")
    def listar_datasets() -> list[dict]:
        return bancada_dados.listar(raiz)

    @app.get("/datasets/{dataset_id}")
    def obter_dataset(dataset_id: str) -> dict:
        _validar_run_id(dataset_id)
        pasta = raiz / "data" / dataset_id
        arquivo_manifesto = pasta / "manifesto.json"
        if not pasta.is_dir() or not arquivo_manifesto.exists():
            raise HTTPException(status_code=404, detail="dataset ou manifesto não encontrado")
        manifesto = json.loads(arquivo_manifesto.read_text(encoding="utf-8"))
        familias = manifesto.get("parametros", {}).get("familias") or list(
            manifesto.get("distribuicao", {}).get("por_familia", {}).keys()
        )
        splits = manifesto.get("distribuicao", {}).get("por_split") or {}
        trem_info = manifesto.get("saidas", {}).get("trem", {})
        benchmark_info = manifesto.get("saidas", {}).get("benchmark", {})
        tem_benchmark = (pasta / "benchmark.jsonl").exists()
        return {
            "id": dataset_id,
            "manifesto": manifesto,
            "familias": familias,
            "splits": splits,
            "itens_treino": trem_info.get("itens"),
            "itens_benchmark": benchmark_info.get("itens"),
            "sha256": trem_info.get("sha256"),
            "idioma": manifesto.get("estatisticas", {}).get("trem", {}).get("idioma_provavel", "pt"),
            "tokens_aprox_trem": manifesto.get("estimativas", {}).get("tokens_aprox_trem"),
            "tem_benchmark": tem_benchmark,
        }

    @app.get("/datasets/{dataset_id}/itens")
    def listar_itens_dataset(
        dataset_id: str,
        desde: int = 0,
        limite: int = 50,
        split: str | None = None,
        familia: str | None = None,
        busca: str | None = None,
    ) -> dict:
        _validar_run_id(dataset_id)
        pasta = raiz / "data" / dataset_id
        if not pasta.is_dir():
            raise HTTPException(status_code=404, detail="dataset não encontrado")

        benchmark_arquivo = pasta / "benchmark.jsonl"
        if benchmark_arquivo.exists():
            total = 0
            filtrados = 0
            itens_selecionados = []
            busca_termo = busca.strip().lower() if busca else None

            with open(benchmark_arquivo, "r", encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if not linha:
                        continue
                    total += 1
                    item = json.loads(linha)
                    if split and item.get("split") != split:
                        continue
                    if familia and item.get("familia") != familia:
                        continue
                    if busca_termo:
                        enunciado = str(item.get("enunciado", "")).lower()
                        cot = str(item.get("cot", "")).lower()
                        resp = str(item.get("resposta", "")).lower()
                        if busca_termo not in enunciado and busca_termo not in cot and busca_termo not in resp:
                            continue

                    if filtrados >= desde and len(itens_selecionados) < limite:
                        itens_selecionados.append(item)
                    filtrados += 1

            return {
                "id": dataset_id,
                "total": total,
                "filtrados": filtrados,
                "desde": desde,
                "limite": limite,
                "itens": itens_selecionados,
                "tem_benchmark": True,
            }

        trem_arquivo = pasta / "trem.txt"
        if not trem_arquivo.exists():
            raise HTTPException(status_code=404, detail="arquivos de texto do dataset não encontrados")

        paragrafos = [p.strip() for p in trem_arquivo.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        total = len(paragrafos)
        busca_termo = busca.strip().lower() if busca else None
        if busca_termo:
            paragrafos = [p for p in paragrafos if busca_termo in p.lower()]
        filtrados = len(paragrafos)
        selecionados = paragrafos[desde : desde + limite]
        itens_formatados = [
            {
                "id": f"{dataset_id}-{desde + i}",
                "enunciado": p,
                "cot": "",
                "resposta": "",
                "familia": "texto",
                "split": "treino",
            }
            for i, p in enumerate(selecionados)
        ]
        return {
            "id": dataset_id,
            "total": total,
            "filtrados": filtrados,
            "desde": desde,
            "limite": limite,
            "itens": itens_formatados,
            "tem_benchmark": False,
        }


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

    @app.post("/execucoes/{chave}/parar")
    def parar_execucao(chave: str) -> dict:
        if not _ID_OK.fullmatch(chave.replace("-", "").replace(".", "")):
            raise HTTPException(status_code=400, detail="chave inválida")
        proc = execucoes.get(chave)
        if proc is None:
            raise HTTPException(status_code=404, detail="execução não encontrada")

        pid = proc.pid
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

        # RF3.7 e CA8: garantir que runs em andamento fiquem com concluido=false no estado.json
        for p in (raiz / "runs").glob("*"):
            if p.is_dir():
                estado_file = p / "estado.json"
                if estado_file.exists():
                    try:
                        dados_est = json.loads(estado_file.read_text(encoding="utf-8"))
                        if not dados_est.get("concluido", False):
                            dados_est["concluido"] = False
                            estado_file.write_text(json.dumps(dados_est, ensure_ascii=False, indent=1), encoding="utf-8")
                    except Exception:
                        pass

        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "execucao_parada", chave=chave, pid=pid
        )
        return {"chave": chave, "pid": pid, "parado": True}

    @app.get("/execucoes/{chave}/log")
    def log_execucao(chave: str, linhas: int = 60) -> dict:
        """Cauda do log: leitura leve e rápida mesmo para arquivos de vários MBs (CA12)."""
        if not _ID_OK.fullmatch(chave.replace("-", "").replace(".", "")):
            raise HTTPException(status_code=400, detail="chave inválida")
        caminho = raiz / ".lab-ia" / "logs" / f"{chave}.log"
        if not caminho.exists():
            return {"chave": chave, "linhas": [], "existe": False, "total": 0}
        linhas_cauda = _ler_cauda_linhas(caminho, n_linhas=max(1, linhas))
        # total = número real de linhas no arquivo (contrato B3); contagem via bytes é eficiente
        # mesmo para logs grandes (não carrega o conteúdo inteiro na memória como string)
        with open(caminho, "rb") as _f:
            total_arquivo = _f.read().count(b"\n")
        return {"chave": chave, "linhas": linhas_cauda, "existe": True, "total": total_arquivo}

    # --- inferência e exploração de raciocínio (B13 RF5 / RF7) -------------

    @app.post("/testar")
    def testar_modelo(corpo: TestarCorpo) -> dict:
        import time
        import torch
        from ..bancada.dados import agora_iso
        from ..reasoning.estrategias import (
            COT,
            _RESPOSTA_COMPLETA,
            _gerar_tokens,
            parse_resposta,
            responder_cot,
            responder_direta,
        )
        from ..trainer.gerar import carregar_para_geracao

        _validar_run_id(corpo.run)
        pasta = raiz / "runs" / corpo.run
        if not pasta.exists():
            raise HTTPException(status_code=404, detail=f"run '{corpo.run}' não encontrado")

        inicio = time.perf_counter()
        modelo, tok, disp = carregar_para_geracao(pasta)

        if corpo.estrategia == "direta":
            resp, texto = responder_direta(modelo, tok, disp, corpo.enunciado, semente=corpo.semente)
        else:
            if corpo.prefixo:
                # RF7: Exploração do CoT (Teacher Forcing)
                prompt_base = f"{corpo.enunciado}\n{COT}\n{corpo.prefixo}"
                ids = tok.encode(prompt_base, add_special_tokens=False).ids

                def resposta_completa(novos: list[int]) -> bool:
                    return _RESPOSTA_COMPLETA.search(tok.decode(novos)) is not None

                novos, _ = _gerar_tokens(
                    modelo,
                    ids,
                    disp,
                    n_max=corpo.max_tokens,
                    temperatura=0.0 if corpo.guloso else corpo.temperatura,
                    guloso=corpo.guloso,
                    gerador=torch.Generator(device=disp).manual_seed(corpo.semente),
                    parada=resposta_completa,
                )
                resto_gerado = tok.decode(novos)
                texto = corpo.prefixo + resto_gerado
                if "Resposta:" in texto:
                    texto = texto.split("Resposta:")[0] + "Resposta:" + texto.split("Resposta:")[1].split("\n")[0]
                resp = parse_resposta(texto)

                # RF7.5: Gravar exclusivamente em .lab-ia/exploracao.jsonl
                exploracao_dir = raiz / ".lab-ia"
                exploracao_dir.mkdir(parents=True, exist_ok=True)
                with open(exploracao_dir / "exploracao.jsonl", "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "data": agora_iso(),
                                "run": corpo.run,
                                "enunciado": corpo.enunciado,
                                "prefixo": corpo.prefixo,
                                "texto_gerado": texto,
                                "resposta_extraida": resp,
                                "estrategia": corpo.estrategia,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                resp, texto = responder_cot(
                    modelo,
                    tok,
                    disp,
                    corpo.enunciado,
                    semente=corpo.semente,
                    guloso=corpo.guloso,
                    n_max=corpo.max_tokens,
                )

        duracao = round(time.perf_counter() - inicio, 3)
        return {
            "run": corpo.run,
            "enunciado": corpo.enunciado,
            "estrategia": corpo.estrategia,
            "texto_gerado": texto,
            "resposta_extraida": resp,
            "exploracao": bool(corpo.prefixo),
            "tempo_s": duracao,
        }

    # --- medição e benchmark de modelos (B13 RF6) -------------------------

    @app.post("/benchmark")
    def iniciar_benchmark(corpo: BenchmarkCorpo) -> dict:
        _validar_run_id(corpo.run)
        pasta_run = raiz / "runs" / corpo.run
        if not pasta_run.exists():
            raise HTTPException(status_code=404, detail=f"run '{corpo.run}' não encontrado")

        caminho_bench = (raiz / corpo.benchmark).resolve()
        if not str(caminho_bench).startswith(str((raiz / "data").resolve())) or not caminho_bench.exists():
            raise HTTPException(status_code=400, detail="benchmark fora de data/ ou inexistente")

        args = [
            sys.executable,
            "-m",
            "labia.cli",
            "raciocinio",
            "--run",
            corpo.run,
            "--estrategia",
            corpo.estrategia,
            "--benchmark",
            str(caminho_bench),
            "--split",
            corpo.split,
            "--semente",
            str(corpo.semente),
            "--raiz",
            str(raiz),
        ]
        if corpo.limite:
            args += ["--limite", str(corpo.limite)]

        log_dir = raiz / ".lab-ia" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        chave = f"benchmark-{corpo.run}-{corpo.estrategia}-{corpo.split}"
        log = open(log_dir / f"{chave}.log", "a", encoding="utf-8")
        processo = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(raiz))
        execucoes[chave] = processo

        from ..utils.estado import LogEventos

        LogEventos(raiz / ".lab-ia" / "eventos.jsonl").registrar(
            "benchmark_iniciado", run=corpo.run, estrategia=corpo.estrategia, split=corpo.split, pid=processo.pid
        )
        return {"chave": chave, "pid": processo.pid, "comando": args}

    # --- catálogo de modelos e bases (B13 RF2.2 / RF2.3) ------------------

    @app.get("/modelos")
    def listar_modelos() -> dict:
        runs_locais = []
        pasta_runs = raiz / "runs"
        if pasta_runs.exists():
            for d in sorted(pasta_runs.iterdir()):
                if not d.is_dir() or d.name.startswith(("_", ".")):
                    continue
                estado_file = d / "estado.json"
                tamanhos_file = d / "tamanhos.json"
                params = None
                disp = "cpu"
                corpus = None
                cfg_mod = None
                if tamanhos_file.exists():
                    try:
                        tam = json.loads(tamanhos_file.read_text(encoding="utf-8"))
                        params = tam.get("total")
                    except Exception:
                        pass
                if estado_file.exists():
                    try:
                        est = json.loads(estado_file.read_text(encoding="utf-8"))
                        disp = est.get("dispositivo", disp)
                        corpus = (est.get("config_treino") or {}).get("corpus")
                        cfg_mod = est.get("config_modelo")
                    except Exception:
                        pass

                # Runs de treino não gravam tamanhos.json (só a quantização grava);
                # reconstrói a contagem a partir do config_modelo com a mesma aritmética
                # verificada de bancada/presets (tests/b7 compara com GPT.contar_parametros).
                if params is None and isinstance(cfg_mod, dict):
                    try:
                        from labia.bancada.presets import contar_parametros

                        params = contar_parametros(
                            vocab=int(cfg_mod.get("vocab", 0)),
                            dim=int(cfg_mod["dim"]),
                            camadas=int(cfg_mod["camadas"]),
                            janela=int(cfg_mod["janela_ctx"]),
                            n_especialistas=int(cfg_mod.get("n_especialistas", 0)),
                            top_k=int(cfg_mod.get("top_k", 1)),
                            norm=str(cfg_mod.get("norm", "layernorm")),
                            pos=str(cfg_mod.get("pos", "aprendido")),
                            mlp=str(cfg_mod.get("mlp", "gelu")),
                            cabecas=int(cfg_mod.get("cabecas", 0)),
                            n_cabecas_kv=int(cfg_mod.get("n_cabecas_kv", 0)),
                        ).get("total")
                    except Exception:
                        params = None

                benchmarks_salvos = []
                for b_file in sorted(d.glob("benchmark-*.json")):
                    try:
                        b_data = json.loads(b_file.read_text(encoding="utf-8"))
                        benchmarks_salvos.append({
                            "estrategia": b_data.get("estrategia"),
                            "split": b_data.get("split"),
                            "acuracia_global": b_data.get("acuracia_global"),
                            "itens": b_data.get("itens"),
                        })
                    except Exception:
                        pass

                # Runs de treino não gravam config.yaml: o dataset de origem vem do
                # config_treino.corpus do estado.json (ex.: data\logica-pq\trem.txt).
                dados_id = None
                config_file = d / "config.yaml"
                if config_file.exists():
                    try:
                        import yaml

                        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                        dados_id = cfg.get("dados")
                    except Exception:
                        pass
                if not dados_id and corpus:
                    partes = Path(str(corpus).replace("\\", "/")).parts
                    if "data" in partes and len(partes) > partes.index("data") + 1:
                        dados_id = partes[partes.index("data") + 1]
                treinado_em_raciocinio = bool(
                    dados_id
                    and (raiz / "data" / str(dados_id) / "benchmark.jsonl").exists()
                )

                runs_locais.append({
                    "id": d.name,
                    "origem": "run",
                    "parametros": params,
                    "dispositivo": disp,
                    "medido": len(benchmarks_salvos) > 0,
                    "treinado_em_raciocinio": treinado_em_raciocinio,
                    "benchmarks": benchmarks_salvos,
                })

        modelos_gguf = []
        pasta_modelos = raiz / "modelos"
        if pasta_modelos.exists():
            for f in sorted(pasta_modelos.glob("*.gguf")):
                tam_mb = round(f.stat().st_size / (1024 * 1024), 1)
                nome = f.name
                quant = "desconhecido"
                for q in ("q4_k_m", "q4_k_s", "q5_k_m", "q8_0", "q4_0", "q4_1", "f16"):
                    if q in nome.lower():
                        quant = q.upper()
                        break
                termos_reasoning = ("r1", "reasoning", "cot", "logic", "qwen2.5")
                declarado = any(t in nome.lower() for t in termos_reasoning)
                modelos_gguf.append({
                    "arquivo": nome,
                    "origem": "gguf",
                    "tamanho_mb": tam_mb,
                    "quantizacao": quant,
                    "reasoning_declarado": declarado,
                    "rotulo_honestidade": "reasoning: declarado pelo publicador — não medido aqui",
                    "medido": False,
                })

        llama_instalado = False
        try:
            import llama_cpp  # type: ignore

            llama_instalado = True
        except ImportError:
            pass

        return {
            "runs": runs_locais,
            "gguf": modelos_gguf,
            "llama_cpp_instalado": llama_instalado,
            "comando_instalacao": "pip install llama-cpp-python",
        }

    return app
