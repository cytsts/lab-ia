"""Presets de modelo, estimativas e gerador de YAML comentado.

O laboratório tinha cinco arquivos .yaml escritos à mão, copiados e editados. Aqui
o mesmo resultado sai de um comando só, com números medidos/estimados antes de
gastar GPU: parâmetros, VRAM aproximada, épocas sobre o corpus e tempo previsto.

A estimativa de tempo é calibrada em uma medição real registrada no PLANO
(G1: dim=256, 6 camadas, janela=256, lote=32 → 281k tokens/s na RTX 3070 em
bf16). O modelo de custo é FLOPs/token = 6*N_ativos + 12*camadas*dim*janela —
o primeiro termo é o custo denso clássico, o segundo é a atenção (cresce com a
janela). É estimativa de ordem de grandeza, não benchmark: a nota do YAML diz isso.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- calibração -------------------------------------------------------------
# Medições reais usadas para ajustar o custo por passo. Cada linha é um treino que
# aconteceu de verdade nesta máquina (RTX 3070, bf16+autocast); o ajuste é refeito
# por regressão em tempo de import, então acrescentar medição recalibra o modelo.
MEDICOES = [
    {
        "nome": "g1-treino-zero",
        "vocab": 4096, "dim": 256, "camadas": 6, "janela": 256, "lote": 32,
        "passos": 2500, "segundos": 72.0, "tokens_por_s": 281_000.0,
        "fonte": "specs/PLANO.md (meta G1)",
    },
    {
        "nome": "livros-rapido",
        "vocab": 4096, "dim": 192, "camadas": 4, "janela": 192, "lote": 24,
        "passos": 1000, "segundos": 14.48, "tokens_por_s": 318_000.0,
        "fonte": "runs/livros-rapido/estado.json (bancada B1)",
    },
    {
        "nome": "valida-custo",
        "vocab": 2048, "dim": 128, "camadas": 3, "janela": 256, "lote": 32,
        "passos": 600, "segundos": 7.29, "tokens_por_s": 674_000.0,
        "fonte": "runs/valida-custo/estado.json (validação independente do modelo de custo)",
    },
    {
        # mesma arquitetura da G1, corpus diferente: mede a variação entre execuções
        "nome": "livros-equilibrado",
        "vocab": 4096, "dim": 256, "camadas": 6, "janela": 256, "lote": 32,
        "passos": 2500, "segundos": 69.45, "tokens_por_s": 295_000.0,
        "fonte": "runs/livros-equilibrado/estado.json (bancada B1, corpus próprio)",
    },
]

# Compatibilidade: mantém o nome antigo apontando para a medição de referência.
REFERENCIA_MEDIDA = MEDICOES[0]

PRESETS: dict[str, dict] = {
    "micro": {
        "descricao": "teste de fumaça: minutos em GPU, horas em CPU — use para validar a config",
        "vocab_bpe": 2048,
        "modelo": {"dim": 128, "camadas": 4, "cabecas": 4, "janela_ctx": 128, "abandono": 0.0},
        "passos": 400,
        "lote": 16,
        "stride": 0,
        "lr": 6e-4,
        "minimo_lr": 6e-5,
        "warmup": 40,
        "avaliar_a_cada": 50,
        "salvar_a_cada": 100,
    },
    "rapido": {
        "descricao": "primeiro treino de verdade: resultados visíveis em poucos minutos",
        "vocab_bpe": 4096,
        "modelo": {"dim": 192, "camadas": 4, "cabecas": 6, "janela_ctx": 192, "abandono": 0.05},
        "passos": 1000,
        "lote": 24,
        "stride": 96,
        "lr": 3e-4,
        "minimo_lr": 3e-5,
        "warmup": 60,
        "avaliar_a_cada": 100,
        "salvar_a_cada": 100,
    },
    "equilibrado": {
        "descricao": "mesma receita do run g1-treino-zero (linha de base do laboratório)",
        "vocab_bpe": 4096,
        "modelo": {"dim": 256, "camadas": 6, "cabecas": 8, "janela_ctx": 256, "abandono": 0.1},
        "passos": 2500,
        "lote": 32,
        "stride": 128,
        "lr": 3e-4,
        "minimo_lr": 3e-5,
        "warmup": 100,
        "avaliar_a_cada": 250,
        "salvar_a_cada": 250,
    },
    "longo": {
        "descricao": "modelo maior e mais passos: exige corpus e paciência maiores",
        "vocab_bpe": 8192,
        "modelo": {"dim": 384, "camadas": 6, "cabecas": 8, "janela_ctx": 320, "abandono": 0.1},
        "passos": 6000,
        "lote": 32,
        "stride": 160,
        "lr": 2.5e-4,
        "minimo_lr": 2.5e-5,
        "warmup": 200,
        "avaliar_a_cada": 250,
        "salvar_a_cada": 250,
    },
    "moe": {
        "descricao": "FFN esparsamente ativado (4 especialistas, top-1) — achado da G4: em corpus pequeno empata com o denso",
        "vocab_bpe": 4096,
        "modelo": {
            "dim": 256,
            "camadas": 6,
            "cabecas": 8,
            "janela_ctx": 256,
            "abandono": 0.1,
            "n_especialistas": 4,
            "top_k": 1,
            "coef_auxiliar": 0.01,
        },
        "passos": 2500,
        "lote": 32,
        "stride": 128,
        "lr": 3e-4,
        "minimo_lr": 3e-5,
        "warmup": 100,
        "avaliar_a_cada": 250,
        "salvar_a_cada": 250,
    },
}

# Chaves aceitas como sobrescrita de linha de comando (apelido -> campo).
SOBRESCRITAS_MODELO = {
    "dim": "dim",
    "camadas": "camadas",
    "cabecas": "cabecas",
    "janela": "janela_ctx",
    "abandono": "abandono",
    "especialistas": "n_especialistas",
    "top_k": "top_k",
    "coef_auxiliar": "coef_auxiliar",
}
SOBRESCRITAS_TREINO = {
    "passos": "passos",
    "lote": "lote",
    "stride": "stride",
    "lr": "lr",
    "minimo_lr": "minimo_lr",
    "warmup": "warmup",
    "vocab_bpe": "vocab_bpe",
    "semente": "semente",
    "dispositivo": "dispositivo",
    "peso_decay": "peso_decay",
    "grad_clip": "grad_clip",
    "avaliar_a_cada": "avaliar_a_cada",
    "salvar_a_cada": "salvar_a_cada",
    "iters_avaliacao": "iters_avaliacao",
}

COMENTARIOS_TREINO = {
    "nome": "identificador do run; vira a pasta runs/<nome>/",
    "corpus": "arquivo de treino (texto puro, UTF-8)",
    "corpus_val": "arquivo de validação; se vazio, o treino divide o corpus (95/5)",
    "vocab_bpe": "tamanho do vocabulário BPE; maior = menos tokens por palavra, embedding maior",
    "passos": "passos de otimização (um lote por passo)",
    "lote": "sequências por passo",
    "stride": "passo entre janelas; menor que janela_ctx = janelas deslizantes (aproveita corpus pequeno)",
    "avaliar_a_cada": "intervalo de avaliação na validação (mede generalização)",
    "salvar_a_cada": "intervalo de checkpoint atômico (é o que permite retomar após queda)",
    "iters_avaliacao": "quantos lotes de validação por avaliação",
    "lr": "learning rate de pico",
    "minimo_lr": "piso do decaimento cosseno",
    "warmup": "passos de aquecimento linear (evita divergir no começo)",
    "peso_decay": "weight decay do AdamW (só em pesos 2D; embeddings de posição ficam fora)",
    "grad_clip": "norma máxima do gradiente (0 desliga)",
    "semente": "semente; com o mesmo corpus e semente o treino é reproduzível",
    "dispositivo": "auto | cuda | cpu",
}

COMENTARIOS_MODELO = {
    "dim": "largura (dimensão do resíduo)",
    "camadas": "profundidade (blocos transformer)",
    "cabecas": "cabeças de atenção; dim precisa ser divisível por cabecas",
    "janela_ctx": "contexto máximo em tokens",
    "abandono": "dropout (0 desliga; corpus pequeno costuma pedir mais)",
    "n_especialistas": "0 = FFN denso; >1 = MoE com n especialistas",
    "top_k": "especialistas ativos por token no MoE",
    "coef_auxiliar": "peso da perda de balanceamento de carga do roteador",
}


def contar_parametros(
    vocab: int, dim: int, camadas: int, janela: int, n_especialistas: int = 0, top_k: int = 1
) -> dict:
    """Conta parâmetros pela mesma aritmética do modelo em models/gpt.py.

    Verificado contra GPT.contar_parametros() no teste tests/b1/test_presets.py —
    se a arquitetura mudar, o teste quebra aqui primeiro.
    """
    embedding = vocab * dim + janela * dim
    atencao = 4 * dim * dim  # qkv (3*dim*dim) + proj (dim*dim), sem bias
    normalizacoes = 4 * dim  # ln1 + ln2
    densa = 8 * dim * dim + 5 * dim  # fc1(4d*d + 4d) + fc2(d*4d + d)
    if n_especialistas > 1:
        mlp = dim * n_especialistas + n_especialistas * densa  # roteador sem bias
        ativos_mlp = dim * n_especialistas + max(1, min(top_k, n_especialistas)) * densa
    else:
        mlp = densa
        ativos_mlp = densa
    total = embedding + camadas * (atencao + normalizacoes + mlp) + 2 * dim
    ativos = embedding + camadas * (atencao + normalizacoes + ativos_mlp) + 2 * dim
    return {"total": int(total), "ativos": int(ativos), "por_camada_mlp": int(mlp)}


def _flops_por_token(ativos: int, camadas: int, dim: int, janela: int) -> float:
    return 6.0 * ativos + 12.0 * camadas * dim * janela


def ajustar_modelo_de_custo(medicoes: list[dict] | None = None) -> dict:
    """Ajusta segundos_por_passo = custo_fixo + flops_por_passo / taxa_peak.

    Por que dois termos: com 2 a 6 milhões de parâmetros a GPU NÃO é o gargalo — o
    custo fixo por passo (Python, otimizador, lançamento de kernel) domina. Um modelo
    só de FLOPs previu 7,2 s para o run livros-rapido e o treino real levou 14,5 s.
    Ajuste por mínimos quadrados sobre as medições registradas em MEDICOES.
    """
    medicoes = medicoes or MEDICOES
    pontos = []
    for m in medicoes:
        params = contar_parametros(m["vocab"], m["dim"], m["camadas"], m["janela"])
        flops = _flops_por_token(params["ativos"], m["camadas"], m["dim"], m["janela"]) * m["lote"] * m["janela"]
        pontos.append((flops, m["segundos"] / m["passos"]))
    n = len(pontos)
    media_x = sum(x for x, _ in pontos) / n
    media_y = sum(y for _, y in pontos) / n
    numerador = sum((x - media_x) * (y - media_y) for x, y in pontos)
    denominador = sum((x - media_x) ** 2 for x, _ in pontos)
    if denominador <= 0 or numerador <= 0:
        return {"custo_fixo_s": 0.0, "flops_por_s": 1.0, "medicoes": len(pontos), "origem": "sem ajuste"}
    inclinacao = numerador / denominador
    return {
        "custo_fixo_s": round(media_y - inclinacao * media_x, 6),
        "flops_por_s": 1.0 / inclinacao,
        "medicoes": n,
        "origem": f"ajuste de 2 termos sobre {n} medições reais (custo fixo + FLOPs)",
    }


MODELO_DE_CUSTO = ajustar_modelo_de_custo()


def validar_modelo_de_custo(medicoes: list[dict] | None = None) -> dict:
    """Erro do modelo com validação cruzada deixando-um-de-fora.

    Ajustar e medir sobre as mesmas medições é circular. Aqui cada medição é
    prevista por um modelo ajustado SEM ela — o número que sobra é erro de
    generalização de verdade, e é o que o teste automatizado cobra.
    """
    medicoes = medicoes or MEDICOES
    erros: dict[str, float] = {}
    for alvo in medicoes:
        restantes = [m for m in medicoes if m is not alvo]
        if len(restantes) < 2:
            continue
        modelo = ajustar_modelo_de_custo(restantes)
        params = contar_parametros(alvo["vocab"], alvo["dim"], alvo["camadas"], alvo["janela"])
        flops = _flops_por_token(params["ativos"], alvo["camadas"], alvo["dim"], alvo["janela"]) * alvo["lote"] * alvo["janela"]
        previsto = (modelo["custo_fixo_s"] + flops / modelo["flops_por_s"]) * alvo["passos"]
        erros[alvo["nome"]] = round(previsto / alvo["segundos"] - 1.0, 4)
    valores = [abs(v) for v in erros.values()]
    return {
        "erros_relativos": erros,
        "erro_maximo": round(max(valores), 4) if valores else None,
        "erro_medio": round(sum(valores) / len(valores), 4) if valores else None,
        "metodo": "validação cruzada deixando-um-de-fora sobre MEDICOES",
    }


def estimar_desempenho(
    params: dict, camadas: int, dim: int, janela: int, lote: int, passos: int, tokens_por_s: float | None = None
) -> dict:
    """Segundos por passo, tempo total e taxa previstos, com o modelo de dois termos."""
    flops_passo = _flops_por_token(params["ativos"], camadas, dim, janela) * lote * janela
    tokens_por_passo = lote * janela
    if tokens_por_s:
        segundos_passo = tokens_por_passo / float(tokens_por_s)
        origem = "medida informada pelo usuário (--tokens-por-s)"
        custo_fixo = None
    else:
        segundos_passo = MODELO_DE_CUSTO["custo_fixo_s"] + flops_passo / MODELO_DE_CUSTO["flops_por_s"]
        origem = MODELO_DE_CUSTO["origem"]
        custo_fixo = MODELO_DE_CUSTO["custo_fixo_s"]
    segundos = segundos_passo * passos
    return {
        "segundos_por_passo": round(segundos_passo, 5),
        "tokens_por_passo": tokens_por_passo,
        "tokens_por_s_previsto": round(tokens_por_passo / max(1e-9, segundos_passo)),
        "tokens_totais": tokens_por_passo * passos,
        "segundos_previstos": round(segundos, 1),
        "horas_previstas": round(segundos / 3600, 2),
        "origem_taxa": origem,
        "custo_fixo_s": custo_fixo,
        "flops_por_passo": round(flops_passo),
        "flops_por_token": round(_flops_por_token(params["ativos"], camadas, dim, janela) / 1e6, 2),
    }


def estimar_vram(params: dict, camadas: int, dim: int, janela: int, lote: int, cabecas: int, vocab: int) -> dict:
    """Ordem de grandeza da VRAM: estado do otimizador + ativações + logits.

    Estados: 16 bytes por parâmetro (peso fp32, gradiente, m e v do AdamW).
    Ativações: ~16 tensores bf16 por camada do tamanho lote*janela*dim, mais a
    matriz de atenção lote*cabecas*janela^2. Logits: lote*janela*vocab em bf16.
    """
    estados = params["total"] * 16
    ativacoes = camadas * lote * janela * dim * 16 * 2
    atencao = lote * cabecas * janela * janela * 2
    logits = lote * janela * vocab * 2
    total = estados + ativacoes + atencao + logits
    return {
        "mb_total": round(total / 1024**2, 1),
        "mb_estados_otimizador": round(estados / 1024**2, 1),
        "mb_ativacoes": round((ativacoes + atencao) / 1024**2, 1),
        "mb_logits": round(logits / 1024**2, 1),
        "nota": "ordem de grandeza (+/-): o pico real sai de torch.cuda.max_memory_allocated",
    }


def montar_config(
    nome: str,
    corpus: str,
    corpus_val: str | None = None,
    preset: str = "equilibrado",
    sobrescritas: dict | None = None,
) -> dict:
    """Combina preset + sobrescritas e valida o que o modelo exige."""
    if preset not in PRESETS:
        raise ValueError(f"preset desconhecido: {preset!r} (use {sorted(PRESETS)})")
    base = PRESETS[preset]
    modelo = dict(base["modelo"])
    cfg = {
        "nome": nome,
        "corpus": corpus,
        "vocab_bpe": base["vocab_bpe"],
        "modelo": modelo,
        "passos": base["passos"],
        "lote": base["lote"],
        "stride": base["stride"],
        "avaliar_a_cada": base["avaliar_a_cada"],
        "salvar_a_cada": base["salvar_a_cada"],
        "iters_avaliacao": base.get("iters_avaliacao", 40),
        "lr": base["lr"],
        "minimo_lr": base["minimo_lr"],
        "warmup": base["warmup"],
        "peso_decay": base.get("peso_decay", 0.1),
        "grad_clip": base.get("grad_clip", 1.0),
        "semente": 42,
        "dispositivo": "auto",
    }
    for chave, valor in (sobrescritas or {}).items():
        if valor is None:
            continue
        if chave in SOBRESCRITAS_MODELO:
            modelo[SOBRESCRITAS_MODELO[chave]] = valor
        elif chave in SOBRESCRITAS_TREINO:
            cfg[SOBRESCRITAS_TREINO[chave]] = valor
        else:
            raise ValueError(f"sobrescrita desconhecida: {chave!r}")

    if corpus_val:
        cfg["corpus_val"] = corpus_val
    _validar(cfg)
    return cfg


def _validar(cfg: dict) -> None:
    m = cfg["modelo"]
    if m["dim"] % m["cabecas"]:
        raise ValueError(f"dim ({m['dim']}) precisa ser divisível por cabecas ({m['cabecas']})")
    if m["janela_ctx"] < 8:
        raise ValueError("janela_ctx mínima é 8")
    for campo in ("passos", "lote", "vocab_bpe"):
        if cfg[campo] < 1:
            raise ValueError(f"{campo} precisa ser >= 1 (recebi {cfg[campo]})")
    if cfg["warmup"] >= cfg["passos"]:
        raise ValueError(f"warmup ({cfg['warmup']}) precisa ser menor que passos ({cfg['passos']})")
    if not cfg["minimo_lr"] <= cfg["lr"]:
        raise ValueError("minimo_lr precisa ser <= lr")
    stride = cfg.get("stride") or 0
    if stride and stride > m["janela_ctx"]:
        raise ValueError(f"stride ({stride}) maior que janela_ctx ({m['janela_ctx']}) desperdiça dados")
    especialistas = m.get("n_especialistas", 0)
    if especialistas and especialistas > 1 and m.get("top_k", 1) > especialistas:
        raise ValueError("top_k não pode ser maior que n_especialistas")


def texto_yaml(cfg: dict, cabecalho: list[str], comentarios_extra: dict[str, str] | None = None) -> str:
    """YAML comentado, escrito à mão para os comentários sobreviverem ao arquivo."""
    extra = comentarios_extra or {}
    linhas = [f"# {l}" if l else "#" for l in cabecalho]
    linhas.append("")
    for campo in (
        "nome", "corpus", "corpus_val", "vocab_bpe", "passos", "lote", "stride",
        "avaliar_a_cada", "salvar_a_cada", "iters_avaliacao", "lr", "minimo_lr",
        "warmup", "peso_decay", "grad_clip", "semente", "dispositivo",
    ):
        if campo not in cfg:
            continue
        comentario = extra.get(campo) or COMENTARIOS_TREINO.get(campo, "")
        linhas.append(_linha(campo, cfg[campo], comentario, 0))
    linhas.append("modelo:")
    for campo, valor in cfg["modelo"].items():
        linhas.append(_linha(campo, valor, COMENTARIOS_MODELO.get(campo, ""), 2))
    return "\n".join(linhas) + "\n"


def _linha(campo: str, valor, comentario: str, recuo: int) -> str:
    prefixo = " " * recuo
    texto = f"{prefixo}{campo}: {_yaml_valor(valor)}"
    if comentario:
        texto = texto.ljust(max(len(texto), 30)) + "  # " + comentario
    return texto


def _yaml_float(valor: float) -> str:
    """Float que o PyYAML reconhece como número.

    Armadilha real: repr(3e-5) = '3e-05', e o resolvedor de float do YAML 1.1 exige
    ponto na mantissa — sem ele o valor vira STRING e o treino quebra no meio do
    laço com TypeError. Aqui a mantissa sempre ganha '.0' quando precisa.
    """
    texto = repr(float(valor))
    if "e" in texto or "E" in texto:
        mantissa, _, expoente = texto.replace("E", "e").partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        if not expoente.startswith(("+", "-")):
            expoente = "+" + expoente
        return f"{mantissa}e{expoente}"
    return texto if "." in texto else texto + ".0"


def _yaml_valor(valor) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, float):
        return _yaml_float(valor)
    if isinstance(valor, str):
        # aspas simples de YAML não interpretam escapes: caminho do Windows sobrevive
        return valor if valor and all(c.isalnum() or c in "._-/:" for c in valor) else "'" + valor.replace("'", "''") + "'"
    return str(valor)


def gerar(
    nome: str,
    dados: str,
    preset: str = "equilibrado",
    destino: str | Path | None = None,
    raiz: str | Path = ".",
    tokens_por_s: float | None = None,
    forcar: bool = False,
    sobrescritas: dict | None = None,
) -> dict:
    """Gera o YAML de treino para um dataset preparado (ou um arquivo solto).

    Devolve {"caminho", "config", "estimativas", "texto"} — o CLI usa isso para
    imprimir o relatório antes de qualquer GPU ser acordada.
    """
    raiz = Path(raiz)
    from .dados import carregar as carregar_dataset  # import tardio: evita ciclo

    pasta_dataset = raiz / "data" / dados
    corpus_val: str | None = None
    manifesto: dict | None = None
    if (pasta_dataset / "manifesto.json").exists():
        _, _, manifesto = carregar_dataset(dados, raiz)
        corpus = str(Path("data") / dados / "trem.txt")
        corpus_val = str(Path("data") / dados / "val.txt")
    elif Path(dados).exists() or (raiz / dados).exists():
        caminho_corpus = Path(dados)
        if not caminho_corpus.exists():
            caminho_corpus = raiz / dados
        corpus = str(caminho_corpus)
    else:
        raise FileNotFoundError(
            f"dados {dados!r} não são um dataset preparado (data/{dados}/) nem um arquivo existente"
        )

    cfg = montar_config(nome, corpus, corpus_val, preset=preset, sobrescritas=sobrescritas)
    m = cfg["modelo"]
    params = contar_parametros(
        cfg["vocab_bpe"], m["dim"], m["camadas"], m["janela_ctx"],
        m.get("n_especialistas", 0), m.get("top_k", 1),
    )
    desempenho = estimar_desempenho(
        params, m["camadas"], m["dim"], m["janela_ctx"], cfg["lote"], cfg["passos"], tokens_por_s
    )
    vram = estimar_vram(
        params, m["camadas"], m["dim"], m["janela_ctx"], cfg["lote"], m["cabecas"], cfg["vocab_bpe"]
    )

    epocas = None
    tokens_corpus = None
    janelas = None
    if manifesto:
        tokens_corpus = manifesto["estimativas"]["tokens_aprox_trem"]
        efetivo = cfg.get("stride") or m["janela_ctx"]
        janelas = max(1, (tokens_corpus - m["janela_ctx"] - 1) // efetivo + 1)
        passos_por_epoca = max(1, janelas // cfg["lote"])
        epocas = round(cfg["passos"] / passos_por_epoca, 2)

    cabecalho = [
        f"Lab-IA — config de treino do zero (preset: {preset})",
        f"gerado por 'lab-ia novo' — {PRESETS[preset]['descricao']}",
        "",
        f"parâmetros: {params['total']:,} totais / {params['ativos']:,} ativos por token".replace(",", "."),
        f"estimativa: {desempenho['tokens_por_s_previsto']:,} tokens/s → {_duracao(desempenho['segundos_previstos'])}".replace(",", "."),
        f"VRAM estimada: {vram['mb_total']} MB ({vram['mb_estados_otimizador']} MB de otimizador)",
    ]
    if epocas is not None:
        cabecalho.append(
            f"corpus: ~{tokens_corpus:,} tokens → {janelas:,} janelas → {epocas} épocas em {cfg['passos']} passos".replace(",", ".")
        )
    cabecalho.append("")
    cabecalho.append("edite à vontade: o treino recusa chave desconhecida em vez de ignorar em silêncio")

    texto = texto_yaml(cfg, cabecalho)

    destino_final = Path(destino) if destino else (raiz / "configs" / f"{nome}.yaml")
    if not destino_final.is_absolute():
        destino_final = raiz / destino_final
    if destino_final.exists() and not forcar:
        raise FileExistsError(f"config já existe: {destino_final} (use --forcar para sobrescrever)")
    destino_final.parent.mkdir(parents=True, exist_ok=True)
    destino_final.write_text(texto, encoding="utf-8")

    try:
        relativo = str(destino_final.relative_to(raiz))
    except ValueError:
        relativo = str(destino_final)

    return {
        "caminho": str(destino_final),
        "caminho_relativo": relativo,
        "config": cfg,
        "parametros": params,
        "desempenho": desempenho,
        "vram": vram,
        "epocas": epocas,
        "janelas": janelas,
        "tokens_corpus": tokens_corpus,
        "manifesto_dataset": manifesto,
        "texto": texto,
    }


def _duracao(segundos: float) -> str:
    """Duração legível: 7,6 s · 12,3 min · 2,10 h."""
    if segundos < 60:
        return f"{segundos:.1f} s".replace(".", ",")
    if segundos < 3600:
        return f"{segundos / 60:.1f} min".replace(".", ",")
    return f"{segundos / 3600:.2f} h".replace(".", ",")


def resumo_texto(resultado: dict) -> str:
    """Relatório legível do que foi gerado (usado pelo CLI e pela UI)."""
    cfg = resultado["config"]
    m = cfg["modelo"]
    p = resultado["parametros"]
    d = resultado["desempenho"]
    v = resultado["vram"]
    linhas = [
        f"config  : {resultado['caminho']}",
        f"run     : {cfg['nome']}",
        f"corpus  : {cfg['corpus']}" + (f"  (validação: {cfg.get('corpus_val')})" if cfg.get("corpus_val") else "  (split 95/5 no treino)"),
        f"modelo  : dim={m['dim']} camadas={m['camadas']} cabecas={m['cabecas']} janela={m['janela_ctx']}"
        + (f" MoE={m['n_especialistas']}x top-{m.get('top_k', 1)}" if m.get("n_especialistas", 0) > 1 else ""),
        f"tamanho : {p['total']:,} parâmetros ({p['ativos']:,} ativos/token)".replace(",", "."),
        f"treino  : {cfg['passos']} passos x lote {cfg['lote']} x janela {m['janela_ctx']} = {d['tokens_totais']:,} tokens".replace(",", "."),
        f"tempo   : ~{_duracao(d['segundos_previstos'])} · {d['segundos_por_passo']:.4f} s/passo · {d['tokens_por_s_previsto']:,} tokens/s".replace(",", "."),
        f"          {d['origem_taxa']}",
        f"          {d['flops_por_passo'] / 1e9:.1f} GFLOP por passo",
        f"          {d['tokens_por_passo']} tokens por passo ({cfg['lote']} x {m['janela_ctx']})",
        f"VRAM    : ~{v['mb_total']} MB (otimizador {v['mb_estados_otimizador']} MB · ativações {v['mb_ativacoes']} MB · logits {v['mb_logits']} MB)",
    ]
    if resultado["epocas"] is not None:
        linhas.append(
            f"dados   : ~{resultado['tokens_corpus']:,} tokens → {resultado['janelas']:,} janelas → {resultado['epocas']} épocas".replace(",", ".")
        )
    if d.get("custo_fixo_s") and d["custo_fixo_s"] / max(1e-9, d["segundos_por_passo"]) > 0.5:
        linhas.append(
            f"NOTA    : {d['custo_fixo_s'] / d['segundos_por_passo']:.0%} do tempo por passo é custo fixo "
            "(Python/otimizador/kernel), não conta de GPU — aumentar --lote ou --janela rende mais por hora"
        )
    if v["mb_total"] > 7000:
        linhas.append("AVISO   : VRAM estimada acima de 7 GB — reduza --lote ou --janela antes de treinar")
    if resultado["epocas"] is not None and resultado["epocas"] > 60:
        linhas.append(
            f"AVISO   : {resultado['epocas']} épocas sobre o corpus — risco alto de decorar (overfit); "
            "reduza --passos ou aumente o corpus"
        )
    linhas.append("")
    linhas.append(f"próximo : lab-ia train --config {resultado.get('caminho_relativo') or resultado['caminho']}")
    return "\n".join(linhas)
