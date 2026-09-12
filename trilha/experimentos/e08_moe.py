r"""Lição 8 — MoE: mais parâmetros sem mais conta (e o preço do roteador).

Experimento: mede o que a esparsidade compra, o que a perda auxiliar vale em cada
situação de roteamento, e — o ponto principal — para onde ela empurra os pesos do
roteador (medido pelo gradiente, não por fé).

Rode:  .venv\Scripts\python trilha\experimentos\e08_moe.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from labia.models.gpt import GPT, CamadaMoE, ConfigGPT, Roteador


def parametros_do_moe(especialistas: int = 4, top_k: int = 1, dim: int = 256, camadas: int = 6) -> dict:
    cfg = ConfigGPT(vocab=4096, dim=dim, camadas=camadas, cabecas=8, janela_ctx=256,
                    n_especialistas=especialistas, top_k=top_k)
    modelo = GPT(cfg)
    total = modelo.contar_parametros()
    ativos = modelo.contar_parametros_ativos()
    return {
        "especialistas": especialistas,
        "top_k": top_k,
        "total": total,
        "ativos": ativos,
        "proporcao_ativa": round(ativos / total, 4),
    }


def perda_auxiliar(fracao: list[float], probabilidade: list[float]) -> float:
    """A fórmula do roteador: n · Σ (fração de tokens · probabilidade média)."""
    n = len(fracao)
    return round(n * sum(f * p for f, p in zip(fracao, probabilidade)), 4)


def extremos_da_perda_auxiliar() -> dict:
    uniforme = [1 / 4] * 4
    colapsado = [1.0, 0.0, 0.0, 0.0]
    meio = [0.7, 0.1, 0.1, 0.1]
    return {
        "uniforme": perda_auxiliar(uniforme, uniforme),
        "colapsado": perda_auxiliar(colapsado, colapsado),
        "meio": perda_auxiliar(meio, meio),
        "explicacao": "piso = 1,0 quando o roteador distribui igualmente; teto = n com um especialista só",
    }


def roteador_enviesado(entradas: int = 64, dim: int = 16, semente: int = 0) -> tuple[Roteador, torch.Tensor, torch.Tensor]:
    """Roteador forçado a mandar tudo para o especialista 0 (é o colapso que se quer medir).

    Somar uma constante à linha do peso NÃO enviesa: a entrada é normalizada (média ~0)
    e o logit é x·w, então c·Σx ≈ 0. O viés tem de ser na DIREÇÃO média das entradas.
    """
    gerador = torch.Generator().manual_seed(semente)
    x = torch.randn(entradas, dim, generator=gerador)
    direcao = x.mean(0)
    cfg = ConfigGPT(vocab=8, dim=dim, camadas=1, cabecas=2, janela_ctx=8, abandono=0.0,
                    n_especialistas=4, top_k=1, coef_auxiliar=0.0)
    roteador = Roteador(cfg)
    with torch.no_grad():
        roteador.peso.weight[0] = 8.0 * direcao / direcao.norm().clamp_min(1e-6)
        roteador.peso.weight[1:] = 0.0
    return roteador, x, direcao


def para_onde_a_auxiliar_empurra() -> dict:
    """O gradiente da perda auxiliar, projetado em cada especialista.

    Descida de gradiente faz w ← w − lr·g. Se (g · direção) > 0, o logit daquele
    especialista DIMINUI; se < 0, aumenta. É assim que a auxiliar desfaz o colapso.
    """
    roteador, x, direcao = roteador_enviesado()
    with torch.no_grad():
        destino, _ = roteador(x)
        fracao = destino.gt(0).float().mean(0)
    roteador.zero_grad()
    _, aux = roteador(x)
    aux.backward()
    gradiente = roteador.peso.weight.grad
    projecoes = [round(float((gradiente[i] * direcao).sum()), 4) for i in range(4)]
    return {
        "aux_com_roteador_colapsado": round(float(aux), 4),
        "fracao_por_especialista": [round(float(f), 4) for f in fracao],
        "projecao_do_gradiente": projecoes,
        "efeito": [
            "aumenta o logit (o gradiente empurra para usar mais)"
            if p < 0
            else "diminui o logit (o gradiente empurra para usar menos)"
            for p in projecoes
        ],
    }


def treino_curto(coef_auxiliar: float, passos: int = 120, semente: int = 0) -> dict:
    """Treino minúsculo de MoE, começando colapsado, para ver o roteador se abrir."""
    from labia.trainer.dados import dividir_corpus, lote_trem, montar_dataset
    from labia.trainer.tokenizacao import carregar_tokenizer, treinar_tokenizer_ptbr
    from labia.trainer.treino import ConfigTreino, fator_lr

    texto = (
        "A noite estava fria e chuvosa quando ele decidiu partir.\n\n"
        "O mar batia com força contra as pedras do quebra-mar.\n\n"
        "Ela sorriu sem dizer nada e guardou a carta na gaveta.\n\n"
    ) * 30
    with tempfile.TemporaryDirectory() as pasta:
        destino = Path(pasta) / "tok"
        treinar_tokenizer_ptbr([texto], 256, destino / "tokenizer.json")
        tokenizer = carregar_tokenizer(destino)
        cfgm = ConfigGPT(vocab=tokenizer.get_vocab_size(), dim=64, camadas=2, cabecas=4,
                         janela_ctx=32, abandono=0.0, n_especialistas=4, top_k=1,
                         coef_auxiliar=coef_auxiliar)
        trem, val = dividir_corpus(texto, semente=semente)
        x, y = montar_dataset(tokenizer, trem, cfgm.janela_ctx)
        torch.manual_seed(semente)
        modelo = GPT(cfgm)
        modelo.init_pesos(semente)
        x_fixo, y_fixo = x[:4], y[:4]
        _forcar_para_um_especialista(modelo, x_fixo, y_fixo)
        uso_inicial = _uso_em_lote_fixo(modelo, x_fixo, y_fixo)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
        cfg = ConfigTreino(passos=passos, warmup=10, lr=3e-3, minimo_lr=3e-4)
        por_epoch = max(1, x.shape[0] // 8)
        for passo in range(passos):
            for grupo in otimizador.param_groups:
                grupo["lr"] = fator_lr(cfg, passo)
            xb, yb = lote_trem(x, y, passo, 8, semente, por_epoch)
            _, perda = modelo(xb, yb)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            otimizador.zero_grad(set_to_none=True)
        uso_final = _uso_em_lote_fixo(modelo, x_fixo, y_fixo)
        aux_final = round(float(modelo.ultimo_aux or 0.0), 4)
    return {
        "coef_auxiliar": coef_auxiliar,
        "uso_inicial": uso_inicial,
        "uso_final": uso_final,
        "concentracao_inicial": round(max(uso_inicial), 4) if uso_inicial else None,
        "concentracao_final": round(max(uso_final), 4) if uso_final else None,
        "aux_router": aux_final,
    }


def _uso_em_lote_fixo(modelo: GPT, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    with torch.no_grad():
        modelo(x, y)
    return modelo.mapa_uso_especialistas() or []


def _forcar_para_um_especialista(modelo: GPT, x: torch.Tensor, y: torch.Tensor, vantagem: float = 8.0) -> None:
    direcoes: dict[int, torch.Tensor] = {}
    ganchos = []
    for indice, bloco in enumerate(modelo.blocos):
        if isinstance(bloco.mlp, CamadaMoE):
            ganchos.append(
                bloco.mlp.roteador.register_forward_hook(
                    lambda _m, entrada, _s, indice=indice: direcoes.__setitem__(indice, entrada[0].detach().mean(0))
                )
            )
    with torch.no_grad():
        modelo(x, y)
    for gancho in ganchos:
        gancho.remove()
    with torch.no_grad():
        for indice, bloco in enumerate(modelo.blocos):
            if indice in direcoes and isinstance(bloco.mlp, CamadaMoE):
                direcao = direcoes[indice]
                peso = bloco.mlp.roteador.peso.weight
                peso[0] = vantagem * direcao / direcao.norm().clamp_min(1e-6)
                peso[1:] = 0.0


def main(passos_treino: int = 120) -> dict:
    resultado = {
        "parametros": {"top1": parametros_do_moe(4, 1), "top2": parametros_do_moe(4, 2)},
        "auxiliar": extremos_da_perda_auxiliar(),
        "gradiente": para_onde_a_auxiliar_empurra(),
        "sem_auxiliar": treino_curto(0.0, passos_treino),
        "com_auxiliar": treino_curto(0.5, passos_treino),
    }
    print("1) parâmetros totais x ativos (4 especialistas, dim 256, 6 camadas)")
    for nome, dados in resultado["parametros"].items():
        print(
            f"   {nome}: {dados['total']:,} totais · {dados['ativos']:,} ativos por token "
            f"({dados['proporcao_ativa']:.1%})".replace(",", ".")
        )
    print()
    a = resultado["auxiliar"]
    print("2) o valor da perda auxiliar (n=4)")
    print(f"   roteamento uniforme  → {a['uniforme']}  (piso)")
    print(f"   roteamento parcial   → {a['meio']}")
    print(f"   roteamento colapsado → {a['colapsado']}  (teto = n)")
    print()
    g = resultado["gradiente"]
    print("3) para onde a auxiliar empurra (gradiente projetado na direção das entradas)")
    print(f"   roteador colapsado: uso {g['fracao_por_especialista']} · aux {g['aux_com_roteador_colapsado']}")
    for i, (projecao, efeito) in enumerate(zip(g["projecao_do_gradiente"], g["efeito"])):
        print(f"   especialista {i}: projeção {projecao:+.4f} → {efeito}")
    print()
    print(f"4) treino curto partindo do MESMO colapso ({passos_treino} passos em CPU)")
    for rotulo, chave in (("sem auxiliar", "sem_auxiliar"), ("com auxiliar (0,5)", "com_auxiliar")):
        dados = resultado[chave]
        print(f"   {rotulo:<20} {dados['uso_inicial']} → {dados['uso_final']}")
        print(
            f"   {'':<20} dominante {dados['concentracao_inicial']:.1%} → {dados['concentracao_final']:.1%} "
            f"· aux {dados['aux_router']}"
        )
    print()
    print("Leia assim: o MoE guarda n especialistas mas calcula top-k por token — é isso que")
    print("separa 'parâmetros totais' de 'custo por token'. O risco é o roteador escolher")
    print("sempre o mesmo (colapso), e a perda auxiliar existe para empurrar o uso de volta")
    print("ao uniforme: o item 3 mostra o gradiente fazendo exatamente isso — diminuindo o")
    print("logit do especialista dominante e aumentando o dos outros.")
    return resultado


if __name__ == "__main__":
    main()
