r"""Lição 6 — LoRA: adaptar sem reescrever o modelo.

Experimento: mede o que o adaptador de baixo posto realmente faz.
  1. quantos parâmetros ficam treináveis (e quantos congelam);
  2. com B iniciado em zero, o adaptador é um no-op EXATO na primeira passada;
  3. o posto da matriz aprendida é limitado por r;
  4. mesclar devolve um Linear comum com a mesma saída (sem custo na inferência).

Rode:  .venv\Scripts\python trilha\experimentos\e06_lora.py
"""
from __future__ import annotations

import torch

from labia.models.gpt import GPT, ConfigGPT
from labia.models.lora import NoLinear, aplicar_lora, estatisticas_lora, mesclar_lora


def config_pequena() -> ConfigGPT:
    return ConfigGPT(vocab=128, dim=64, camadas=2, cabecas=4, janela_ctx=32, abandono=0.0)


def adaptador_zera_a_saida(r: int = 8) -> dict:
    """B=0 ⇒ o ramo LoRA contribui zero: o modelo adaptado começa idêntico à base."""
    torch.manual_seed(0)
    base = GPT(config_pequena())
    base.init_pesos(0)
    base.eval()
    adaptado = GPT(config_pequena())
    adaptado.load_state_dict(base.state_dict())
    adaptado.eval()
    aplicar_lora(adaptado, r=r, alpha=2 * r)
    entrada = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        saida_base = base(entrada)[0]
        saida_adaptada = adaptado(entrada)[0]
    return {
        "diferenca_maxima": float((saida_base - saida_adaptada).abs().max()),
        "identico": bool(torch.equal(saida_base, saida_adaptada)),
    }


def proporcao_treinavel(r: int = 8) -> dict:
    torch.manual_seed(0)
    modelo = GPT(config_pequena())
    modelo.init_pesos(0)
    antes = estatisticas_lora(modelo)
    stats = aplicar_lora(modelo, r=r, alpha=2 * r)
    return {
        "r": r,
        "total": stats["total"],
        "treinaveis": stats["treinaveis"],
        "proporcao": round(stats["proporcao"], 5),
        "modulos_alterados": len(stats["modulos_alterados"]),
        "antes_total": antes["total"],
        "antes_treinaveis": antes["treinaveis"],
        "adaptador_adicionado": stats["total"] - antes["total"],
    }


def posto_do_aprendizado(r: int = 4) -> dict:
    """B·A tem posto no máximo r — é a hipótese do método, medida e não assumida."""
    torch.manual_seed(1)
    base = torch.nn.Linear(32, 32)
    no = NoLinear(base, r=r, alpha=2 * r)
    with torch.no_grad():
        no.lora_B.normal_(0, 0.5)  # simula um adaptador JÁ treinado (B deixou de ser zero)
        no.lora_A.normal_(0, 0.2)
    delta = (no.lora_B @ no.lora_A) * no.escala
    return {
        "r": r,
        "posto_da_atualizacao": int(torch.linalg.matrix_rank(delta)),
        "parametros_do_ramo": int(no.lora_A.numel() + no.lora_B.numel()),
        "parametros_da_base": int(base.weight.numel()),
    }


def mesclar_preserva_a_saida(r: int = 4) -> dict:
    torch.manual_seed(2)
    modelo = GPT(config_pequena())
    modelo.init_pesos(0)
    modelo.eval()
    aplicar_lora(modelo, r=r, alpha=2 * r)
    for modulo in modelo.modules():
        if isinstance(modulo, NoLinear):
            with torch.no_grad():
                modulo.lora_B.normal_(0, 0.1)
    entrada = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        antes = modelo(entrada)[0]
    trocados = mesclar_lora(modelo)
    with torch.no_grad():
        depois = modelo(entrada)[0]
    tipos = {type(m).__name__ for m in modelo.modules() if isinstance(m, torch.nn.Linear)}
    return {
        "modulos_mesclados": trocados,
        "diferenca_maxima": float((antes - depois).abs().max()),
        "ainda_tem_nolinear": any(isinstance(m, NoLinear) for m in modelo.modules()),
        "tipos_de_linear": sorted(tipos),
    }


def main() -> dict:
    resultado = {
        "no_op": adaptador_zera_a_saida(),
        "proporcao": proporcao_treinavel(),
        "posto": posto_do_aprendizado(),
        "mesclagem": mesclar_preserva_a_saida(),
    }
    n, p, po, m = resultado["no_op"], resultado["proporcao"], resultado["posto"], resultado["mesclagem"]
    print("1) o adaptador começa como no-op (B iniciado em zero)")
    print(f"   diferença máxima na saída vs modelo base: {n['diferenca_maxima']:.3e}  (idêntico: {n['identico']})")
    print()
    print("2) quantos parâmetros ficam treináveis")
    print(f"   modelo antes do LoRA : {p['antes_total']:,} parâmetros, TODOS treináveis".replace(",", "."))
    print(f"   modelo com adaptador : {p['total']:,} (+{p['adaptador_adicionado']:,} do ramo A·B)".replace(",", "."))
    print(f"   treináveis com r={p['r']}     : {p['treinaveis']:,} em {p['modulos_alterados']} módulos".replace(",", "."))
    print(f"   proporção      : {p['proporcao']:.2%}")
    print()
    print("3) o posto da atualização é limitado por r")
    print(f"   r = {po['r']} → posto medido da matriz B·A = {po['posto_da_atualizacao']}")
    print(f"   ramo treinável: {po['parametros_do_ramo']} parâmetros contra {po['parametros_da_base']} da base")
    print()
    print("4) mesclar devolve um Linear comum com a mesma saída")
    print(f"   módulos mesclados: {m['modulos_mesclados']} · diferença máxima: {m['diferenca_maxima']:.3e}")
    print(f"   ainda há NoLinear no modelo? {m['ainda_tem_nolinear']} · tipos de Linear: {m['tipos_de_linear']}")
    print()
    print("Leia assim: LoRA não 'reduz' o modelo — ele congela a base e aprende um ajuste")
    print("de posto baixo. O ganho é de custo: treinar 1–3% dos parâmetros e guardar um")
    print("adaptador de poucos MB em vez de um checkpoint inteiro. Mesclar devolve o")
    print("formato original, quando você quiser servir sem overhead.")
    return resultado


if __name__ == "__main__":
    main()
