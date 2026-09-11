"""Base dos agentes autônomos do lab (spec G9): papel, skills expostas, log de ações."""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..utils.estado import LogEventos


@dataclass
class Skill:
    nome: str
    descricao: str
    assinatura: str
    fn: Callable[..., dict] = field(repr=False, default=None)  # type: ignore[assignment]


class AgenteBase:
    nome: str = "base"
    papel: str = ""
    escopo: str = ""

    def __init__(self, raiz: Path | str = ".", arquivo_eventos: str = ".lab-ia/eventos.jsonl"):
        self.raiz = Path(raiz)
        self.eventos = LogEventos(
            Path(arquivo_eventos) if Path(arquivo_eventos).is_absolute() else self.raiz / arquivo_eventos
        )
        self._skills: dict[str, Skill] = {}
        for skill in self.definir_skills():
            self._skills[skill.nome] = skill

    def definir_skills(self) -> list[Skill]:  # pragma: no cover - sobrescrito
        raise NotImplementedError

    def expor_skills(self) -> list[dict]:
        return [
            {"nome": s.nome, "descricao": s.descricao, "assinatura": s.assinatura}
            for s in self._skills.values()
        ]

    def executar(self, skill: str, **argumentos) -> dict:
        if skill not in self._skills:
            self.eventos.registrar("agente_acao", agente=self.nome, skill=skill, ok=False, erro="skill desconhecida")
            raise ValueError(f"{self.nome} não conhece a skill {skill!r} (disponíveis: {sorted(self._skills)})")
        inicio = time.time()
        try:
            resultado = self._skills[skill].fn(**argumentos)
        except Exception as exc:
            self.eventos.registrar(
                "agente_acao", agente=self.nome, skill=skill, ok=False,
                erro=f"{type(exc).__name__}: {exc}", duracao_s=round(time.time() - inicio, 2),
            )
            raise
        self.eventos.registrar(
            "agente_acao", agente=self.nome, skill=skill, ok=True,
            duracao_s=round(time.time() - inicio, 2),
            resumo={k: v for k, v in resultado.items() if isinstance(v, (int, float, str, bool, type(None)))},
        )
        return resultado

    def __repr__(self) -> str:
        return f"<Agente {self.nome}: {len(self._skills)} skills>"


def serializar(args: dict) -> dict:
    """Converte dataclasses em dict para o transporte JSON da CLI."""
    return {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v for k, v in args.items()}
