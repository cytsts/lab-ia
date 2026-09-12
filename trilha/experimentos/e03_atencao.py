r"""Lição 3 — atenção e a máscara causal.

Experimento: prova, medindo, que a posição t só enxerga até t — e mostra o
tamanho do vazamento quando a máscara causal é esquecida.

Rode:  .venv\Scripts\python trilha\experimentos\e03_atencao.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from labia.models.gpt import AtencaoMultiCabeca, ConfigGPT


def config_pequena() -> ConfigGPT:
    return ConfigGPT(vocab=64, dim=32, camadas=1, cabecas=4, janela_ctx=8, abandono=0.0)


def causalidade_na_atencao_do_lab() -> dict:
    """Muda o ÚLTIMO token e confere que as saídas anteriores ficam idênticas."""
    torch.manual_seed(0)
    atencao = AtencaoMultiCabeca(config_pequena())
    atencao.eval()  # sem dropout: queremos o efeito da máscara, não do ruído
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        base = atencao(x)
        alterado = x.clone()
        alterado[0, -1] = alterado[0, -1] + 5.0  # bagunça só a última posição
        novo = atencao(alterado)
    diferenca_antes = float((base[0, :-1] - novo[0, :-1]).abs().max())
    diferenca_ultima = float((base[0, -1] - novo[0, -1]).abs().max())
    return {
        "diferenca_nas_posicoes_anteriores": diferenca_antes,
        "diferenca_na_ultima_posicao": diferenca_ultima,
        "causal": diferenca_antes == 0.0,
    }


def vazamento_sem_mascara() -> dict:
    """A mesma conta SEM is_causal: quanto do futuro entra na posição 0."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 6, 8)
    k = torch.randn(1, 4, 6, 8)
    v = torch.randn(1, 4, 6, 8)
    causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    livre = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    # bagunça o último token e vê o que acontece na posição 0
    v2 = v.clone()
    v2[0, :, -1] = v2[0, :, -1] + 5.0
    causal2 = F.scaled_dot_product_attention(q, k, v2, is_causal=True)
    livre2 = F.scaled_dot_product_attention(q, k, v2, is_causal=False)
    return {
        "posicao_0_muda_com_causal": float((causal[0, :, 0] - causal2[0, :, 0]).abs().max()),
        "posicao_0_muda_sem_causal": float((livre[0, :, 0] - livre2[0, :, 0]).abs().max()),
    }


def main() -> dict:
    resultado = {"causalidade": causalidade_na_atencao_do_lab(), "vazamento": vazamento_sem_mascara()}
    c, v = resultado["causalidade"], resultado["vazamento"]
    print("1) causalidade na AtencaoMultiCabeca do núcleo (core/labia/models/gpt.py)")
    print(f"   mudança vista nas posições ANTERIORES: {c['diferenca_nas_posicoes_anteriores']:.3e}")
    print(f"   mudança vista na própria última     : {c['diferenca_na_ultima_posicao']:.3f}")
    print(f"   causal? {c['causal']}")
    print()
    print("2) o que aconteceria sem a máscara (scaled_dot_product_attention)")
    print(f"   posição 0 com is_causal=True : {v['posicao_0_muda_com_causal']:.3e}")
    print(f"   posição 0 com is_causal=False: {v['posicao_0_muda_sem_causal']:.3f}")
    print()
    print("Leia assim: sem a máscara, o token 0 já 'vê' o último token. Treinar assim")
    print("dá uma perda baixíssima e um modelo inútil: ele aprende a copiar o alvo que")
    print("está olhando. É o erro silencioso mais caro de um GPT feito à mão.")
    return resultado


if __name__ == "__main__":
    main()
