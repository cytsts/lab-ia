"""CLI `lab-ia`: operações headless do laboratório (treinar, gerar, servir)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    if args.comando == "servir":
        import uvicorn

        from .api.app import criar_app

        uvicorn.run(criar_app(args.raiz), host=args.host, port=args.porta)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
