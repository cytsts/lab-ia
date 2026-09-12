r"""Lição 4 — o bloco transformer, o resíduo e por que a escala da inicialização importa.

Experimento: mede como a variância da ativação se comporta ao atravessar as camadas,
com e sem a escala sqrt(2·camadas) que GPT.init_pesos aplica.

Rode:  .venv\Scripts\python trilha\experimentos\e04_bloco.py
"""
from __future__ import annotations

import math

import torch

from labia.models.gpt import GPT, ConfigGPT


def modelo(camadas: int = 6, dim: int = 64, semente: int = 0) -> GPT:
    cfg = ConfigGPT(vocab=128, dim=dim, camadas=camadas, cabecas=4, janela_ctx=32, abandono=0.0)
    modelo_ = GPT(cfg)
    modelo_.init_pesos(semente)
    modelo_.eval()
    return modelo_


def desfazer_escala(modelo_: GPT) -> None:
    """Multiplica de volta as projeções de resíduo: é o 'sem escala' do experimento."""
    escala = math.sqrt(2 * modelo_.cfg.camadas)
    with torch.no_grad():
        for bloco in modelo_.blocos:
            bloco.atencao.proj.weight.mul_(escala)
            bloco.mlp.fc2.weight.mul_(escala)


def perfil_de_ativacao(modelo_: GPT, entrada: torch.Tensor) -> list[float]:
    """Desvio-padrão da ativação depois de cada bloco."""
    with torch.no_grad():
        x = modelo_.emb(entrada)
        perfil = [round(float(x.std()), 4)]
        for bloco in modelo_.blocos:
            x, _ = bloco(x)
            perfil.append(round(float(x.std()), 4))
    return perfil


def main() -> dict:
    torch.manual_seed(0)
    entrada = torch.randint(0, 128, (2, 32))
    com_escala = modelo()
    sem_escala = modelo()
    desfazer_escala(sem_escala)

    perfil_com = perfil_de_ativacao(com_escala, entrada)
    perfil_sem = perfil_de_ativacao(sem_escala, entrada)
    crescimento_com = perfil_com[-1] / max(1e-9, perfil_com[0])
    crescimento_sem = perfil_sem[-1] / max(1e-9, perfil_sem[0])

    print("desvio-padrão da ativação depois de cada bloco")
    print(f"  com escala sqrt(2·camadas): {perfil_com}")
    print(f"  sem escala                : {perfil_sem}")
    print()
    print(f"crescimento da entrada até a última camada:")
    print(f"  com escala: {crescimento_com:.2f}x")
    print(f"  sem escala: {crescimento_sem:.2f}x")
    print()
    print("Leia assim: cada bloco SOMA na entrada (resíduo). Se a contribuição do bloco")
    print("não for pequena na saída, a magnitude explode ao longo da profundidade, e com")
    print("ela o gradiente. GPT.init_pesos multiplica as projeções de saída por")
    print("1/sqrt(2·camadas) justamente por isso — a regra que o GPT-2 usa.")
    return {
        "perfil_com_escala": perfil_com,
        "perfil_sem_escala": perfil_sem,
        "crescimento_com_escala": round(crescimento_com, 4),
        "crescimento_sem_escala": round(crescimento_sem, 4),
    }


if __name__ == "__main__":
    main()
