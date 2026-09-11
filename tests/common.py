""" Helpers compartilhados dos testes (importáveis como módulo comum)."""
from __future__ import annotations

import random
from pathlib import Path

FRASES = [
    "A noite estava fria e chuvosa quando ele decidiu partir.",
    "Todos os caminhos levavam à velha praça do colégio.",
    "Ela sorriu sem dizer nada e guardou a carta na gaveta.",
    "O mar batia com força contra as pedras do quebra-mar.",
    "Ninguém sabia ao certo quando a casa fora abandonada.",
    "Disseram que ele voltaria na primavera, mas nunca voltou.",
    "A janela dava para um quintal cheio de laranjeiras.",
    "Ele escreveu o nome dela na areia e esperou a maré apagar.",
    "O sino da igreja tocava ao meio-da-manhã sem pressa nenhuma.",
    "Havia livros empoeirados em cada canto daquela biblioteca.",
]


def gerar_corpus(n_paragrafos: int = 160, semente: int = 7) -> str:
    rnd = random.Random(semente)
    blocos = []
    for _ in range(n_paragrafos):
        frases = [rnd.choice(FRASES) for _ in range(rnd.randint(4, 12))]
        blocos.append(" ".join(frases))
    return "\n\n".join(blocos)


FRASES_TAREFA = [
    "O sensor mediu a temperatura ambiente com precisao de zero coma um grau.",
    "A amostra foi analisada por espectrometria de massa em modo positivo.",
    "O protocolo experimental exige tres repeticoes independentes por condition.",
    "Os dados brutos foram registrados no laboratorio de computacao cientifica.",
    "A calibracao do equipamento segue a norma internacional vigente.",
    "O modelo estatistico assumiu distribuicao normal dos residuos.",
    "A experiencia produziu um aumento significativo de rendimento.",
    "Os reagentes foram armazenados a menos vinte graus Celsius.",
    "O circuito integrado dissipa calor proporcional a corrente eletrica.",
    "A medicao do campo magnetico foi feita com sonda hall calibrada.",
]


def gerar_tarefa(n_paragrafos: int = 120, semente: int = 21) -> str:
    rnd = random.Random(semente)
    return "\n\n".join(
        " ".join(rnd.choices(FRASES_TAREFA, k=rnd.randint(4, 10))) for _ in range(n_paragrafos)
    )


def config_micro(corpus: Path, run: Path | None, **extras) -> dict:
    base = {
        "nome": "micro",
        "corpus": str(corpus),
        "vocab_bpe": 256,
        "modelo": {"dim": 64, "camadas": 2, "cabecas": 2, "janela_ctx": 64, "abandono": 0.0},
        "passos": 20,
        "lote": 2,
        "avaliar_a_cada": 10,
        "salvar_a_cada": 10,
        "iters_avaliacao": 5,
        "lr": 1e-3,
        "minimo_lr": 1e-4,
        "warmup": 4,
        "peso_decay": 0.0,
        "grad_clip": 1.0,
        "semente": 42,
        "dispositivo": "cpu",
        "arquivo_eventos": str(Path(run) / "eventos.jsonl") if run else ".lab-ia/eventos.jsonl",
    }
    base.update(extras)
    return base


def criar_dir_run(run: Path) -> Path:
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    (run / "tokens").mkdir(parents=True, exist_ok=True)
    return run
