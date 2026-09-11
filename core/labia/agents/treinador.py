"""Agente Treinador: executa treino, ajuste e quantização como skills (spec G9)."""
from __future__ import annotations

from pathlib import Path

from ..trainer.ajuste import ConfigAjuste, executar_ajuste
from ..trainer.quantiza import quantizar_run
from ..trainer.treino import ConfigTreino, executar_treino
from .base import AgenteBase, Skill


class AgenteTreinador(AgenteBase):
    nome = "treinador"
    papel = "conduz experimentos de treinamento, fine-tuning e quantização"
    escopo = "executa runners do núcleo; nunca edita configs do usuário"

    def definir_skills(self) -> list[Skill]:
        return [
            Skill(
                "train_model",
                "treina um modelo do zero a partir de uma config YAML",
                "train_model(config: str, run_id: str | None = None, retomar: bool = False) -> dict",
                self.train_model,
            ),
            Skill(
                "fine_tune",
                "aplica LoRA/QLoRA sobre um run-base a partir de config YAML",
                "fine_tune(config: str, run_id: str | None = None, retomar: bool = False) -> dict",
                self.fine_tune,
            ),
            Skill(
                "quantize",
                "quantiza os lineares de um run (int8|nf4) com relatório",
                "quantize(run: str, saida: str, modo: str = 'int8', corpus: str = 'data/tarefa_ciencia.txt') -> dict",
                self.quantize,
            ),
        ]

    def _dir(self, run_id: str) -> Path:
        d = self.raiz / "runs" / run_id
        (d / "ckpt").mkdir(parents=True, exist_ok=True)
        (d / "tokens").mkdir(parents=True, exist_ok=True)
        return d

    def train_model(self, config: str, run_id: str | None = None, retomar: bool = False) -> dict:
        cfg = ConfigTreino.de_arquivo(config)
        rid = run_id or cfg.nome
        run_dir = self._dir(rid)
        executar_treino(cfg, run_dir, retomar=retomar, raiz=self.raiz)
        return {"run_id": rid, "estado": "concluido", "pasta": str(run_dir)}

    def fine_tune(self, config: str, run_id: str | None = None, retomar: bool = False) -> dict:
        cfg = ConfigAjuste.de_arquivo(config)
        rid = run_id or cfg.nome
        run_dir = self._dir(rid)
        executar_ajuste(cfg, run_dir, retomar=retomar, raiz=self.raiz)
        return {"run_id": rid, "adaptador": str(run_dir / "adaptador")}

    def quantize(self, run: str, saida: str, modo: str = "int8", corpus: str = "data/tarefa_ciencia.txt") -> dict:
        destino = self._dir(saida)
        tam = quantizar_run(
            self.raiz / "runs" / run, destino, modo,
            (self.raiz / corpus) if not Path(corpus).is_absolute() else corpus,
            raiz=self.raiz,
        )
        return {
            "run_id": saida, "modo": modo,
            "fator_alvos": round(tam["fator_alvos"], 2),
            "perda_antes": round(tam["perda_val_antes"], 3),
            "perda_depois": round(tam["perda_val_depois"], 3),
        }
