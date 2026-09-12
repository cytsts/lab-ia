r"""Lição 5 — o laço de treino: lr, warmup, clip e o que a perda está dizendo.

Experimento: imprime a curva de learning rate que o treino realmente usa e mede
o efeito de exagerar no lr — a diferença entre "não aprendeu" e "divergiu".

Rode:  .venv\Scripts\python trilha\experimentos\e05_treino.py
"""
from __future__ import annotations

import torch

from labia.trainer.treino import ConfigTreino, fator_lr


def curva_de_lr(passos: int = 120, warmup: int = 20, lr: float = 3e-3, minimo: float = 3e-4) -> list[dict]:
    """Formato do schedule: sobe linear no warmup, desce em cosseno até o piso."""
    cfg = ConfigTreino(passos=passos, warmup=warmup, lr=lr, minimo_lr=minimo)
    pontos = [0, warmup // 2, warmup, passos // 2, int(passos * 0.75), passos - 1]
    return [{"passo": p, "lr": round(fator_lr(cfg, p), 8)} for p in sorted(set(pontos))]


def treino_curto(lr: float, passos: int = 150, semente: int = 0) -> dict:
    """Treino minúsculo em CPU para comparar um lr sadio com um exagerado."""
    from labia.trainer.dados import dividir_corpus, lote_trem, montar_dataset
    from labia.trainer.tokenizacao import treinar_tokenizer_ptbr
    from pathlib import Path
    import tempfile

    from labia.models.gpt import GPT, ConfigGPT

    texto = (
        "A noite estava fria e chuvosa quando ele decidiu partir.\n\n"
        "O mar batia com força contra as pedras do quebra-mar.\n\n"
        "Ela sorriu sem dizer nada e guardou a carta na gaveta.\n\n"
    ) * 30
    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta) / "tok"
        treinar_tokenizer_ptbr([texto], 256, destino / "tokenizer.json")
        from labia.trainer.tokenizacao import carregar_tokenizer

        tokenizer = carregar_tokenizer(destino)
        cfgm = ConfigGPT(vocab=tokenizer.get_vocab_size(), dim=64, camadas=2, cabecas=4, janela_ctx=32, abandono=0.0)
        trem, val = dividir_corpus(texto, semente=semente)
        x, y = montar_dataset(tokenizer, trem, cfgm.janela_ctx)
        xv, yv = montar_dataset(tokenizer, val, cfgm.janela_ctx)

        torch.manual_seed(semente)
        modelo = GPT(cfgm)
        modelo.init_pesos(semente)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, betas=(0.9, 0.98))
        cfg = ConfigTreino(passos=passos, warmup=10, lr=lr, minimo_lr=lr / 10)
        passos_por_epoch = max(1, x.shape[0] // 8)
        perdas = []
        for passo in range(passos):
            for grupo in otimizador.param_groups:
                grupo["lr"] = fator_lr(cfg, passo)
            xb, yb = lote_trem(x, y, passo, 8, semente, passos_por_epoch)
            _, perda = modelo(xb, yb)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            otimizador.zero_grad(set_to_none=True)
            perdas.append(float(perda.detach()))
        modelo.eval()
        with torch.no_grad():
            _, perda_val = modelo(xv[:4], yv[:4])
    return {
        "lr": lr,
        "perda_inicial": round(sum(perdas[:10]) / 10, 4),
        "perda_final": round(sum(perdas[-10:]) / 10, 4),
        "perda_val": round(float(perda_val), 4),
        "finito": all(p == p and abs(p) != float("inf") for p in perdas),
        "maior_perda": round(max(perdas), 4),
    }


def main(lr_bom: float = 3e-3, lr_alto: float = 1.0, passos: int = 150) -> dict:
    curva = curva_de_lr()
    print("1) a curva de learning rate do treino (warmup + cosseno)")
    print(f"   {'passo':>6} {'lr':>12}")
    for ponto in curva:
        print(f"   {ponto['passo']:>6} {ponto['lr']:>12.8f}")
    print()
    resultado = {"curva_lr": curva, "treinos": []}
    print("2) dois treinos curtos em CPU (mesmo dado, mesma semente, só o lr muda)")
    for lr in (lr_bom, lr_alto):
        medido = treino_curto(lr, passos=passos)
        resultado["treinos"].append(medido)
        print(
            f"   lr={medido['lr']:<6} inicial {medido['perda_inicial']:.3f} → final {medido['perda_final']:.3f} "
            f"· validação {medido['perda_val']:.3f} · maior perda no meio {medido['maior_perda']:.3f}"
        )
    print()
    print("Leia assim: lr pequeno demais desce devagar (e às vezes nem sai do lugar);")
    print("lr grande demais faz a perda oscilar ou explodir — o grad_clip segura parte,")
    print("mas não conserta. O warmup existe para os primeiros passos, quando o")
    print("gradiente de um modelo recém-inicializado é errático.")
    return resultado


if __name__ == "__main__":
    main()
