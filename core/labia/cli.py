"""CLI `lab-ia`: operações headless do laboratório (treinar, gerar, servir)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bancada import dados as bancada_dados
from .bancada import presets as bancada_presets
from .trainer.gerar import gerar_de_checkpoint
from .trainer.treino import ConfigTreino, executar_treino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab-ia", description="Lab-IA — laboratório de IA em pt-BR")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("train", help="treina um modelo do zero (spec G1)")
    p.add_argument("--config", required=True, help="YAML de configuração (ex.: configs/g1_treino_zero.yaml)")
    p.add_argument("--run-id", default=None, help="identificador do run (padrão: campo 'nome' da config)")
    p.add_argument("--retomar", action="store_true", help="retoma do último checkpoint")
    p.add_argument("--raiz", default=".", help="diretório raiz do lab (padrão: atual)")

    p = sub.add_parser("gerar", help="gera texto de amostra do último checkpoint de um run")
    p.add_argument("--run", required=True, help="run-id")
    p.add_argument("--prompt", required=True)
    p.add_argument("--passos-max", type=int, default=120)
    p.add_argument("--temperatura", type=float, default=0.8)
    p.add_argument("--guloso", action="store_true", help="decodificação gulosa (determinística)")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("ajustar", help="fine-tuning LoRA/QLoRA sobre um run-base (spec G2)")
    p.add_argument("--config", required=True, help="YAML de ajuste (ex.: configs/g2_ajuste_estilo.yaml)")
    p.add_argument("--base", default=None, help="run-id da base (sobrescreve o campo 'base' da config)")
    p.add_argument("--run-id", default=None, help="identificador do run de ajuste (padrão: campo 'nome')")
    p.add_argument("--retomar", action="store_true")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("quantizar", help="quantiza os lineares de um run (spec G3: int8|nf4)")
    p.add_argument("--run", required=True, help="run-id da base")
    p.add_argument("--saida", required=True, help="run-id do run quantizado derivado")
    p.add_argument("--modo", required=True, choices=["int8", "nf4"])
    p.add_argument("--corpus", default="data/tarefa_ciencia.txt", help="corpus de avaliação da perda")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("raciocinio", help="benchmark de raciocínio: direta|cot|tot ou --comparar (spec G5)")
    p.add_argument("--run", required=True, help="run-id (base ou com adaptador)")
    p.add_argument("--estrategia", choices=["direta", "cot", "tot"], default=None)
    p.add_argument("--comparar", action="store_true", help="roda as três e grava comparativo.json")
    p.add_argument("--benchmark", default="data/benchmark_matematica.jsonl")
    p.add_argument("--split", default="teste")
    p.add_argument("--limite", type=int, default=None)
    p.add_argument("--semente", type=int, default=1234)
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("agentes", help="lista agentes e skills expostas (spec G9)")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("agente", help="executa uma skill de um agente")
    p.add_argument("nome", help="treinador | avaliador | arquiteto")
    p.add_argument("skill", help="nome da skill")
    p.add_argument("--json", default="{}", help="argumentos em JSON")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("dados", help="bancada B1: prepara corpus próprio (limpeza, dedup, split, manifesto)")
    p.add_argument("--de", dest="fontes", action="append", default=[], help="arquivo, pasta ou curinga (pode repetir)")
    p.add_argument("--id", default=None, help="id do dataset (vira data/<id>/)")
    p.add_argument("--destino", default="data", help="pasta base dos datasets (padrão: data)")
    p.add_argument("--frac-val", type=float, default=0.05, help="fração de validação (padrão 0.05)")
    p.add_argument("--min-chars", type=int, default=20, help="descarta parágrafos menores que isso (0 = mantém tudo)")
    p.add_argument("--sem-quase-duplicata", action="store_true", help="só dedup exata (mais rápido)")
    p.add_argument("--limiar-quase", type=float, default=0.8, help="Jaccard mínimo para quase-duplicata")
    p.add_argument("--campo", default=None, help="campo de texto em fontes .jsonl (padrão: texto/text)")
    p.add_argument("--limite-mb", type=float, default=256.0, help="teto de leitura por segurança")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--forcar", action="store_true", help="ignora o teto de MB")
    p.add_argument("--listar", action="store_true", help="lista os datasets já preparados")
    p.add_argument("--json", action="store_true", help="imprime o manifesto em JSON")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("novo", help="bancada B1: gera config de treino comentada, com estimativas")
    p.add_argument("--nome", default=None, help="nome do run (vira runs/<nome>/)")
    p.add_argument("--dados", default=None, help="id de dataset preparado (data/<id>) ou arquivo de corpus")
    p.add_argument("--preset", default="equilibrado", choices=sorted(bancada_presets.PRESETS))
    p.add_argument("--saida", default=None, help="caminho do YAML (padrão: configs/<nome>.yaml)")
    p.add_argument("--tokens-por-s", type=float, default=None, help="use sua própria medição de velocidade")
    p.add_argument("--forcar", action="store_true", help="sobrescreve YAML existente")
    p.add_argument("--listar-presets", action="store_true", help="mostra os presets e sai")
    p.add_argument("--raiz", default=".")
    for apelido, tipo, ajuda in (
        ("--dim", int, "largura do modelo"),
        ("--camadas", int, "profundidade"),
        ("--cabecas", int, "cabeças de atenção"),
        ("--janela", int, "contexto em tokens"),
        ("--abandono", float, "dropout"),
        ("--especialistas", int, "nº de especialistas MoE (0 = denso)"),
        ("--top-k", int, "especialistas ativos por token"),
        ("--passos", int, "passos de otimização"),
        ("--lote", int, "sequências por passo"),
        ("--stride", int, "passo entre janelas (0 = janela inteira)"),
        ("--lr", float, "learning rate de pico"),
        ("--minimo-lr", float, "piso do decaimento"),
        ("--warmup", int, "passos de aquecimento"),
        ("--vocab-bpe", int, "tamanho do vocabulário BPE"),
        ("--semente", int, "semente"),
        ("--dispositivo", str, "auto | cuda | cpu"),
    ):
        p.add_argument(apelido, dest=apelido.lstrip("-").replace("-", "_"), type=tipo, default=None, help=ajuda)

    p = sub.add_parser("comparar", help="bancada B2: compara runs lado a lado e diz o que fazer")
    p.add_argument("runs", nargs="*", help="run-ids (padrão: todos os runs com métricas)")
    p.add_argument("--json", action="store_true", help="imprime o resultado em JSON")
    p.add_argument("--relatorio", default=None, help="grava um relatório markdown no caminho dado")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("curva", help="bancada B2: desenha as curvas dos runs (PNG/SVG)")
    p.add_argument("runs", nargs="*", help="run-ids (padrão: todos os runs com métricas)")
    p.add_argument("--saida", default="curvas.svg", help="arquivo de saída (.svg sem dependências, .png com matplotlib)")
    p.add_argument(
        "--metricas",
        default="loss_val,loss_trem,lr,tokens_por_s",
        help="paineis separados por vírgula",
    )
    p.add_argument("--titulo", default=None)
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("varrer", help="bancada B5: varredura de hiperparâmetros com orçamento igual")
    p.add_argument("--base", required=True, help="config base (ex.: configs/livros-rapido.yaml)")
    p.add_argument("--grade", action="append", default=[], help="chave=v1,v2 (pode repetir para várias chaves)")
    p.add_argument("--modo", choices=["grade", "aleatorio"], default="grade")
    p.add_argument("--n", type=int, default=None, help="quantas combinações sortear no modo aleatório")
    p.add_argument("--passos", type=int, default=None, help="orçamento de passos por variante (padrão: o da base)")
    p.add_argument("--prefixo", default=None, help="prefixo dos runs (padrão: varrer-<base>)")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--seco", action="store_true", help="só lista as combinações, sem treinar")
    p.add_argument("--json", action="store_true", help="imprime o relatório em JSON")
    p.add_argument("--relatorio", default=None, help="grava também um relatório markdown")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("trilha", help="trilha de estudo: lições em pt-BR + experimentos executáveis")
    p.add_argument("--licao", type=int, default=None, help="número da lição (sem isso, lista todas)")
    p.add_argument("--rodar", action="store_true", help="roda o experimento da lição escolhida")
    p.add_argument("--rodar-tudo", action="store_true", help="roda todos os experimentos em ordem")
    p.add_argument("--conferir", action="store_true", help="confere se as lições ainda citam código existente")
    p.add_argument("--json", action="store_true")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser(
        "logica-proposicional",
        help="gera dataset de lógica proposicional (avaliação, tautologia, satisfatível, equivalência, implicação)",
    )
    p.add_argument("--id", required=True, help="id do dataset (vira data/<id>/)")
    p.add_argument("--familias", default="avaliacao,tautologia,satisfativel,equivalencia,implicacao")
    p.add_argument("--treino", type=int, default=3000, help="itens de treino")
    p.add_argument("--teste", type=int, default=300, help="itens de teste (mesma dificuldade do treino)")
    p.add_argument("--dificil", type=int, default=200, help="itens do split difícil (generalização de comprimento)")
    p.add_argument("--variaveis", type=int, default=2, help="quantas variáveis (p, q, r...)")
    p.add_argument("--operadores", type=int, default=3, help="operadores por fórmula no treino/teste")
    p.add_argument("--operadores-dificeis", type=int, default=5, help="operadores no split difícil")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--mostrar", type=int, default=0, help="imprime N exemplos com a resposta")
    p.add_argument("--destino", default="data")
    p.add_argument("--json", action="store_true")
    p.add_argument("--raiz", default=".")

    p = sub.add_parser("servir", help="expõe a API interna para a camada visual")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--porta", type=int, default=8765)
    p.add_argument("--raiz", default=".")

    args = parser.parse_args(argv)

    if args.comando == "train":
        cfg = ConfigTreino.de_arquivo(args.config)
        run_id = args.run_id or cfg.nome
        run_dir = Path(args.raiz) / "runs" / run_id
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (run_dir / "tokens").mkdir(parents=True, exist_ok=True)
        print(f"[lab-ia] treinando run '{run_id}' (dispositivo={cfg.dispositivo}, passos={cfg.passos})")
        executar_treino(cfg, run_dir, retomar=args.retomar, raiz=args.raiz)
        print(f"[lab-ia] concluído: métricas em {run_dir / 'metricas.jsonl'}")
        return 0

    if args.comando == "gerar":
        run_dir = Path(args.raiz) / "runs" / args.run
        texto = gerar_de_checkpoint(
            run_dir, args.prompt, passos_max=args.passos_max, temperatura=args.temperatura, guloso=args.guloso
        )
        print(texto)
        return 0

    if args.comando == "ajustar":
        from .trainer.ajuste import ConfigAjuste, executar_ajuste

        cfg = ConfigAjuste.de_arquivo(args.config)
        if args.base:
            base = Path(args.base)
            cfg.base = str(base if base.is_dir() else Path(args.raiz) / "runs" / args.base)
        run_id = args.run_id or cfg.nome
        run_dir = Path(args.raiz) / "runs" / run_id
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        print(f"[lab-ia] ajustando ({cfg.tipo}, r={cfg.r}, base={cfg.base}) -> run '{run_id}'")
        executar_ajuste(cfg, run_dir, retomar=args.retomar, raiz=args.raiz)
        print(f"[lab-ia] ajuste concluído: adaptador em {run_dir / 'adaptador'}")
        return 0

    if args.comando == "quantizar":
        from .trainer.quantiza import quantizar_run

        base_dir = Path(args.raiz) / "runs" / args.run
        destino_dir = Path(args.raiz) / "runs" / args.saida
        destino_dir.mkdir(parents=True, exist_ok=True)
        tam = quantizar_run(base_dir, destino_dir, args.modo, args.corpus, raiz=args.raiz)
        print(
            f"[lab-ia] quantizado ({args.modo}): fator {tam['fator_alvos']:.2f}x nos alvos, "
            f"bits efetivos {tam['bits_efetivos_por_parametro']}, "
            f"perda {tam['perda_val_antes']:.3f} -> {tam['perda_val_depois']:.3f}"
        )
        return 0

    if args.comando == "raciocinio":
        from .reasoning.runner import comparar_estrategias, executar_benchmark

        run_dir = Path(args.raiz) / "runs" / args.run
        if not args.estrategia and not args.comparar:
            parser.error("especifique --estrategia ou --comparar")
        if args.comparar:
            comp = comparar_estrategias(
                run_dir,
                benchmark=args.benchmark,
                split=args.split,
                semente=args.semente,
                limite=args.limite,
                raiz=args.raiz,
            )
            print(
                f"[lab-ia] comparativo: direta={comp['estrategias']['direta']['acuracia_global']:.3f} "
                f"cot={comp['estrategias']['cot']['acuracia_global']:.3f} "
                f"tot={comp['estrategias']['tot']['acuracia_global']:.3f} "
                f"(cot-direta {comp['cot_menos_direta_pp']:+.1f} p.p.)"
            )
        else:
            rel = executar_benchmark(
                run_dir,
                args.estrategia,
                benchmark=args.benchmark,
                split=args.split,
                semente=args.semente,
                limite=args.limite,
                raiz=args.raiz,
            )
            print(
                f"[lab-ia] {args.estrategia}: acurácia {rel['acuracia_global']:.3f} "
                f"valida {rel['taxa_resposta_valida']:.3f} por família {rel['acuracia_por_familia']}"
            )
        return 0

    if args.comando == "agentes":
        import json as _json

        from .agents import criar_agentes

        saida = {
            nome: {"papel": ag.papel, "escopo": ag.escopo, "skills": ag.expor_skills()}
            for nome, ag in criar_agentes(args.raiz).items()
        }
        print(_json.dumps(saida, ensure_ascii=False, indent=1))
        return 0

    if args.comando == "agente":
        import json as _json

        from .agents import REGISTRO

        if args.nome not in REGISTRO:
            parser.error(f"agente {args.nome!r} desconhecido (use {sorted(REGISTRO)})")
        ag = REGISTRO[args.nome](raiz=args.raiz)
        argumentos = _json.loads(args.json)
        resultado = ag.executar(args.skill, **argumentos)
        print(_json.dumps(resultado, ensure_ascii=False, indent=1, default=str))
        return 0

    if args.comando == "dados":
        import json as _json

        if args.listar:
            achados = bancada_dados.listar(args.raiz)
            if args.json:
                print(_json.dumps(achados, ensure_ascii=False, indent=1))
            elif not achados:
                print("[lab-ia] nenhum dataset preparado em data/ — use: lab-ia dados --de <arquivo> --id <nome>")
            else:
                print(f"{'id':<22} {'chars trem':>12} {'chars val':>11} {'idioma':<10} fontes")
                for d in achados:
                    print(
                        f"{d['id']:<22} {d['chars_trem']:>12} {d['chars_val']:>11} {d['idioma']:<10} "
                        + ", ".join(d["fontes"])
                    )
            return 0
        if not args.fontes:
            parser.error("informe --de <arquivo|pasta|curinga> (ou use --listar)")
        if not args.id:
            parser.error("informe --id <nome-do-dataset>")
        try:
            cfg_preparo = bancada_dados.ConfigPreparo(
                fontes=args.fontes,
                id=args.id,
                destino=args.destino,
                frac_val=args.frac_val,
                semente=args.semente,
                min_chars_paragrafo=args.min_chars,
                quase_duplicata=not args.sem_quase_duplicata,
                limiar_quase=args.limiar_quase,
                campo_jsonl=args.campo,
                limite_mb=args.limite_mb,
                forcar=args.forcar,
                raiz=args.raiz,
            )
            manifesto = bancada_dados.preparar(cfg_preparo)
        except (ValueError, FileNotFoundError) as e:
            print(f"[lab-ia] erro no preparo: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(_json.dumps(manifesto, ensure_ascii=False, indent=1))
        else:
            print(_relatorio_dados(manifesto))
        return 0

    if args.comando == "novo":
        if args.listar_presets:
            print(f"{'preset':<14} {'modelo':<44} descrição")
            for nome_preset, valor in bancada_presets.PRESETS.items():
                m = valor["modelo"]
                descricao_modelo = (
                    f"dim={m['dim']} camadas={m['camadas']} cabecas={m['cabecas']} "
                    f"janela={m['janela_ctx']} passos={valor['passos']}"
                )
                print(f"{nome_preset:<14} {descricao_modelo:<44} {valor['descricao']}")
            return 0
        if not args.nome or not args.dados:
            parser.error("informe --nome <run> e --dados <dataset|arquivo>")
        campos = (
            "dim", "camadas", "cabecas", "janela", "abandono", "especialistas", "top_k",
            "passos", "lote", "stride", "lr", "minimo_lr", "warmup", "vocab_bpe", "semente", "dispositivo",
        )
        sobrescritas = {c: getattr(args, c) for c in campos if getattr(args, c, None) is not None}
        try:
            resultado_novo = bancada_presets.gerar(
                nome=args.nome,
                dados=args.dados,
                preset=args.preset,
                destino=args.saida,
                raiz=args.raiz,
                tokens_por_s=args.tokens_por_s,
                forcar=args.forcar,
                sobrescritas=sobrescritas,
            )
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"[lab-ia] erro ao gerar config: {e}", file=sys.stderr)
            return 2
        print(bancada_presets.resumo_texto(resultado_novo))
        return 0

    if args.comando == "comparar":
        import json as _json

        from .bancada import comparar as bancada_comparar

        runs = args.runs or bancada_comparar.listar_runs(args.raiz)
        if not runs:
            print("[lab-ia] nenhum run com métricas em runs/ — treine algo primeiro", file=sys.stderr)
            return 2
        resultado = bancada_comparar.comparar(runs, args.raiz)
        if args.relatorio:
            destino = bancada_comparar.salvar_relatorio(resultado, args.relatorio)
            print(f"[lab-ia] relatório: {destino}")
        if args.json:
            print(bancada_comparar.json_serializavel(resultado))
        else:
            print(bancada_comparar.tabela_texto(resultado))
        return 0

    if args.comando == "curva":
        from .bancada import comparar as bancada_comparar
        from .bancada import curva as bancada_curva

        runs = args.runs or bancada_comparar.listar_runs(args.raiz)
        if not runs:
            print("[lab-ia] nenhum run com métricas em runs/ — treine algo primeiro", file=sys.stderr)
            return 2
        paineis = tuple(m.strip() for m in args.metricas.split(",") if m.strip())
        try:
            destino = bancada_curva.desenhar(
                runs, raiz=args.raiz, saida=args.saida, paineis=paineis, titulo=args.titulo
            )
        except (ValueError, RuntimeError) as e:
            print(f"[lab-ia] erro ao desenhar: {e}", file=sys.stderr)
            return 2
        print(f"[lab-ia] curvas de {', '.join(runs)} em {destino}")
        return 0

    if args.comando == "varrer":
        import json as _json

        from .bancada import varrer as bancada_varrer

        prefixo = args.prefixo or f"varrer-{Path(args.base).stem}"
        try:
            cfg_varrer = bancada_varrer.ConfigVarredura(
                base=args.base,
                grades=bancada_varrer.interpretar_grades(
                    args.grade,
                    ConfigTreino.de_arquivo(
                        Path(args.raiz) / args.base if not Path(args.base).is_absolute() else Path(args.base)
                    ),
                ),
                prefixo=prefixo,
                modo=args.modo,
                n=args.n,
                passos=args.passos,
                semente=args.semente,
                raiz=args.raiz,
                seco=args.seco,
            )
            relatorio = bancada_varrer.executar(cfg_varrer)
        except (ValueError, FileNotFoundError, FileExistsError) as e:
            print(f"[lab-ia] erro na varredura: {e}", file=sys.stderr)
            return 2
        if args.relatorio:
            destino = bancada_varrer.salvar_relatorio(relatorio, args.relatorio)
            print(f"[lab-ia] relatório: {destino}")
        if args.json:
            print(_json.dumps(relatorio, ensure_ascii=False, indent=1, default=str))
        else:
            print(bancada_varrer.tabela_texto(relatorio))
        return 0

    if args.comando == "trilha":
        import json as _json

        from . import trilha as estudo

        if args.conferir:
            problemas = estudo.conferir_referencias(args.raiz)
            if args.json:
                print(_json.dumps(problemas, ensure_ascii=False, indent=1))
            elif not problemas:
                print("[lab-ia] todas as lições citam arquivos e símbolos que existem")
            else:
                print(f"[lab-ia] {len(problemas)} citação(ões) quebrada(s):")
                for problema in problemas:
                    simbolo = f" → {problema.get('simbolo')}" if problema.get("simbolo") else ""
                    print(f"  lição {problema['licao']}: {problema['arquivo']}{simbolo} — {problema['motivo']}")
            return 1 if problemas else 0

        if args.rodar_tudo:
            resultados = estudo.rodar_todos(args.raiz)
            falhas = []
            for resultado in resultados:
                print(f"=== lição {resultado['licao']} — {resultado['titulo']} ({resultado['experimento']}) ===")
                print(resultado["saida"].rstrip())
                if resultado["codigo"] != 0:
                    falhas.append(resultado["licao"])
                    print(resultado["erro"].rstrip(), file=sys.stderr)
                print()
            print(f"[lab-ia] {len(resultados) - len(falhas)}/{len(resultados)} experimentos rodaram")
            return 1 if falhas else 0

        if args.licao is None:
            indice = estudo.indice(args.raiz)
            if args.json:
                print(_json.dumps(indice, ensure_ascii=False, indent=1))
                return 0
            print("Trilha de estudo — para entender o que os números dizem\n")
            for licao in indice["licoes"]:
                marca = Path(licao["experimento"]).name if licao["tem_experimento"] else "—"
                print(f"  {licao['numero']}. {licao['titulo']}")
                print(f"     {licao['resumo']}")
                print(f"     experimento: {marca}")
            print("\nleia uma:  lab-ia trilha --licao 3")
            print("rode:      lab-ia trilha --licao 3 --rodar")
            print("tudo:      lab-ia trilha --rodar-tudo   (~1 min)")
            return 0

        try:
            caminho = estudo.achar_licao(args.licao, args.raiz).arquivo
        except ValueError as e:
            print(f"[lab-ia] {e}", file=sys.stderr)
            return 2
        print(caminho.read_text(encoding="utf-8"))
        if args.rodar:
            print("\n=== rodando o experimento ===")
            processo = estudo.rodar_experimento(args.licao, args.raiz)
            print(processo.stdout.rstrip())
            if processo.returncode != 0:
                print(processo.stderr.rstrip(), file=sys.stderr)
                return 1
        return 0

    if args.comando == "logica-proposicional":
        import json as _json

        from .logica import gerador_proposicional as logica

        try:
            manifesto = logica.gerar(
                logica.ConfigLogica(
                    id=args.id,
                    familias=tuple(f.strip() for f in args.familias.split(",") if f.strip()),
                    n_variaveis=args.variaveis,
                    operadores=args.operadores,
                    operadores_dificeis=args.operadores_dificeis,
                    n_treino=args.treino,
                    n_teste=args.teste,
                    n_dificil=args.dificil,
                    semente=args.semente,
                    destino=args.destino,
                    raiz=args.raiz,
                )
            )
        except ValueError as e:
            print(f"[lab-ia] erro ao gerar logica: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(_json.dumps(manifesto, ensure_ascii=False, indent=1))
            return 0
        d = manifesto["distribuicao"]
        print(f"dataset : {manifesto['id']}  ({manifesto['pasta']})")
        print(f"treino  : {d['por_split']['treino']} itens · {d['por_familia']}")
        print(f"teste   : {d['por_split']['teste']} itens (mesma dificuldade)")
        print(f"dificil : {d['por_split']['dificil']} itens (ate {args.operadores_dificeis} operadores vs {args.operadores})")
        print(f"tamanho : {manifesto['saidas']['trem']['bytes'] / 1024:.0f} KB de corpus · sha {manifesto['saidas']['trem']['sha256']}")
        print(f"benchmark: {manifesto['saidas']['benchmark']['arquivo']}")
        for item in manifesto["amostras"][: args.mostrar]:
            print()
            print(f"--- {item['id']} ({item['familia']}) ---")
            print(item["enunciado"])
            print(item["cot"])
        print()
        print(f"proximo : lab-ia novo --nome logica-1 --dados {manifesto['id']} --preset equilibrado")
        print(f"          lab-ia raciocinio --run logica-1 --comparar --benchmark {manifesto['saidas']['benchmark']['arquivo']}")
        return 0

    if args.comando == "servir":
        import uvicorn

        from .api.app import criar_app

        uvicorn.run(criar_app(args.raiz), host=args.host, port=args.porta)
        return 0

    return 2


def _relatorio_dados(manifesto: dict) -> str:
    """Relatório legível do preparo (o manifesto completo fica em data/<id>/manifesto.json)."""
    limpeza = manifesto["limpeza"]
    trem = manifesto["saidas"]["trem"]
    val = manifesto["saidas"]["val"]
    est = manifesto["estatisticas"]["trem"]
    linhas = [
        f"dataset : {manifesto['id']}  ({manifesto['pasta']})",
        f"fontes  : {len(manifesto['fontes'])} arquivo(s)",
    ]
    for fonte in manifesto["fontes"]:
        linhas.append(
            f"          {fonte['caminho']}  {fonte['bytes'] / 1024:.0f} KB  "
            f"{fonte['codificacao']}  sha {fonte['sha256']}"
        )
    linhas += [
        f"limpeza : {limpeza['paragrafos_brutos']} parágrafos → {limpeza['paragrafos_finais']} "
        f"(exatas {limpeza['removidas_exatas']}, quase {limpeza['removidas_quase']}, "
        f"curtos {limpeza['descartados_por_tamanho']})",
        f"treino  : {trem['chars']} chars / {trem['paragrafos']} parágrafos  ({trem['arquivo']})  sha {trem['sha256']}",
        f"validação: {val['chars']} chars / {val['paragrafos']} parágrafos  ({val['arquivo']})  sha {val['sha256']}",
        f"idioma  : {est['idioma_provavel']} (pistas {est['pistas_idioma']}) · "
        f"entropia {est['entropia_caractere_bits']} bits/char · ascii {est['fracao_ascii']:.0%}",
        f"vocabulário: {est['palavras_unicas']} palavras únicas · {est['chars_por_palavra']} chars/palavra",
        f"vazamento val→trem: {manifesto['vazamento_val_para_trem']['palavras_val_no_trem']:.1%} do vocabulário de validação",
        f"tokens  : ~{manifesto['estimativas']['tokens_aprox_trem']} no treino / "
        f"~{manifesto['estimativas']['tokens_aprox_val']} na validação (estimativa)",
        "top palavras: " + ", ".join(p for p, _ in est["top_palavras"][:10]),
        "",
        f"próximo : lab-ia novo --nome meu-run --dados {manifesto['id']} --preset equilibrado",
    ]
    return "\n".join(linhas)


if __name__ == "__main__":
    sys.exit(main())
