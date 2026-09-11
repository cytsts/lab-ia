from .base import AgenteBase, Skill
from .treinador import AgenteTreinador
from .avaliador import AgenteAvaliador
from .arquiteto import AgenteArquiteto

REGISTRO = {
    "treinador": AgenteTreinador,
    "avaliador": AgenteAvaliador,
    "arquiteto": AgenteArquiteto,
}


def criar_agentes(raiz="."):
    return {nome: cls(raiz=raiz) for nome, cls in REGISTRO.items()}


__all__ = ["AgenteBase", "Skill", "AgenteTreinador", "AgenteAvaliador", "AgenteArquiteto", "REGISTRO", "criar_agentes"]
