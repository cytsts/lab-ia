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

    if args.comando == "servir":
        import uvicorn

        from .api.app import criar_app

        uvicorn.run(criar_app(args.raiz), host=args.host, port=args.porta)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
