"""Varredura de hiperparâmetros: trocar uma coisa por vez e medir o efeito.

Sem isto, "otimizar modelo" continua sendo chute: muda-se o lr, treina-se, olha-se o
número, muda-se outra coisa, e no fim ninguém sabe qual mudança causou o quê. Aqui a
grade inteira roda com orçamento igual, cada variante vira um run normal (reproduzível
por 'lab-ia train'), e o resumo responde duas perguntas:

  1. qual combinação generalizou melhor;
  2. o que cada chave fez — média da melhor validação por valor testado.

Regra que o laboratório segue: comparação só vale com o mesmo orçamento de passos.
Por isso --passos existe e é aplicado a todas as variantes.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..experiments.runner import dir_run
from ..trainer.treino import ConfigTreino, executar_treino
from . import comparar as bancada_comparar
from .presets import SOBRESCRITAS_MODELO, SOBRESCRITAS_TREINO, _validar

LIMITE_COMBINACOES = 64  # teto de segurança: 64 treinos já é sessão longa
# Chaves que a varredura aceita: as mesmas sobrescritas validadas do gerador de config.
CHAVES_VARREIVEIS = set(SOBRESCRITAS_MODELO) | set(SOBRESCRITAS_TREINO)


def agora_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --- leitura da linha de comando -------------------------------------------

def _coagir(valor: str, referencia):
    """Converte o texto da grade para o tipo do valor que já existe na config base.

    Sem isso, 'lr=0.0003' viraria string e o treino quebraria no primeiro cálculo
    (o mesmo problema do '3e-05' do YAML).
    """
    if isinstance(referencia, bool):
        if valor.lower() in ("true", "1", "sim"):
            return True
        if valor.lower() in ("false", "0", "nao", "não"):
            return False
        raise ValueError(f"esperava booleano, recebi {valor!r}")
    if isinstance(referencia, int):
        return int(valor)
    if isinstance(referencia, float):
        return float(valor)
    return valor


def valor_atual(cfg: ConfigTreino, chave: str):
    """Valor da chave na config base (campo direto ou dentro de 'modelo')."""
    if chave in SOBRESCRITAS_MODELO:
        return cfg.modelo.get(SOBRESCRITAS_MODELO[chave])
    if chave in SOBRESCRITAS_TREINO:
        return getattr(cfg, SOBRESCRITAS_TREINO[chave])
    raise ValueError(f"chave não varreível: {chave!r} (use {sorted(CHAVES_VARREIVEIS)})")


def interpretar_grades(pares: list[str], cfg: ConfigTreino) -> dict[str, list]:
    """Converte ['lr=1e-4,3e-4', 'lote=16,32'] em {'lr': [0.0001, 0.0003], ...}."""
    grades: dict[str, list] = {}
    for par in pares:
        if "=" not in par:
            raise ValueError(f"grade {par!r} precisa estar no formato chave=v1,v2")
        chave, _, lista = par.partition("=")
        chave = chave.strip()
        if chave not in CHAVES_VARREIVEIS:
            raise ValueError(f"chave não varreível: {chave!r} (use {sorted(CHAVES_VARREIVEIS)})")
        referencia = valor_atual(cfg, chave)
        valores = [_coagir(v.strip(), referencia) for v in lista.split(",") if v.strip()]
        if not valores:
            raise ValueError(f"grade {chave!r} sem valores")
        if len(set(valores)) != len(valores):
            raise ValueError(f"grade {chave!r} tem valor repetido: {valores}")
        if len(valores) == 1:
            raise ValueError(f"grade {chave!r} com um valor só não é varredura (mude ou tire a chave)")
        grades[chave] = valores
    if not grades:
        raise ValueError("nenhuma grade informada: use --grade chave=v1,v2")
    return grades


def montar_variantes(grades: dict[str, list], modo: str = "grade", n: int | None = None, semente: int = 42) -> list[dict]:
    """Lista de sobrescritas a testar; 'grade' é o produto cartesiano, 'aleatorio' sorteia."""
    chaves = sorted(grades)
    todas = [dict(zip(chaves, combinacao)) for combinacao in itertools.product(*(grades[c] for c in chaves))]
    if modo == "grade":
        if len(todas) > LIMITE_COMBINACOES:
            raise ValueError(
                f"a grade gera {len(todas)} combinações (teto {LIMITE_COMBINACOES}). "
                "Use --modo aleatorio --n K ou reduza as listas."
            )
        return todas
    if modo != "aleatorio":
        raise ValueError(f"modo desconhecido: {modo!r} (use grade ou aleatorio)")
    # 'n or 8' engoliria n=0 em silêncio: o pedido do usuário tem de ser respeitado
    # (ou recusado), nunca trocado por um padrão sem avisar.
    quantidade = min(len(todas), 8) if n is None else n
    if quantidade < 1:
        raise ValueError("--n precisa ser >= 1")
    if quantidade >= len(todas):
        return todas
    return random.Random(semente).sample(todas, quantidade)


# --- execução ---------------------------------------------------------------

@dataclass
class ConfigVarredura:
    base: str
    grades: dict[str, list]
    prefixo: str
    modo: str = "grade"
    n: int | None = None
    passos: int | None = None
    semente: int = 42
    raiz: str = "."
    seco: bool = False
    variantes: list[dict] = field(default_factory=list)


def _aplicar(cfg: ConfigTreino, sobrescritas: dict) -> None:
    """Aplica as sobrescritas na config e revalida (mesma régua do gerador)."""
    for chave, valor in sobrescritas.items():
        if chave in SOBRESCRITAS_MODELO:
            cfg.modelo[SOBRESCRITAS_MODELO[chave]] = valor
        else:
            setattr(cfg, SOBRESCRITAS_TREINO[chave], valor)
    _validar(
        {
            "nome": cfg.nome,
            "modelo": cfg.modelo,
            "passos": cfg.passos,
            "lote": cfg.lote,
            "vocab_bpe": cfg.vocab_bpe,
            "warmup": cfg.warmup,
            "lr": cfg.lr,
            "minimo_lr": cfg.minimo_lr,
            "stride": cfg.stride,
        }
    )
    if cfg.passos <= 0 or cfg.lote <= 0:
        raise ValueError("passos e lote precisam ser positivos")


def _config_da_variante(caminho_base: Path, cfg_varrer: "ConfigVarredura", sobrescritas: dict) -> ConfigTreino:
    """Config base + orçamento da varredura + sobrescritas, já validada.

    Vale também no modo seco: é para isso que ele existe — descobrir que a grade é
    impossível (dim não divisível por cabeças, warmup maior que passos) antes de
    gastar GPU.
    """
    cfg = ConfigTreino.de_arquivo(caminho_base)
    cfg.semente = cfg_varrer.semente
    cfg.arquivo_eventos = str(Path(".lab-ia") / "eventos.jsonl")
    if cfg_varrer.passos:
        cfg.passos = cfg_varrer.passos
        cfg.warmup = min(cfg.warmup, max(1, cfg.passos // 10))
        cfg.avaliar_a_cada = max(1, cfg.passos // 10)
        cfg.salvar_a_cada = max(1, cfg.passos // 5)
        cfg.minimo_lr = min(cfg.minimo_lr, cfg.lr)
    _aplicar(cfg, sobrescritas)
    return cfg


def _yaml_da_variante(cfg: ConfigTreino, destino: Path) -> Path:
    dados = {campo: getattr(cfg, campo) for campo in cfg.para_dict()}
    dados.pop("modelo", None)
    dados["modelo"] = cfg.modelo
    destino.write_text(yaml.safe_dump(dados, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destino


def _efeito_por_chave(variantes: list[dict], grades: dict[str, list], base_por_chave: dict) -> dict:
    """Média e melhor validação por valor testado — o 'o que essa chave fez'."""
    efeito: dict[str, dict] = {}
    for chave, valores in grades.items():
        por_valor: dict[str, dict] = {}
        for valor in valores:
            medidas = [
                v["melhor_val"]
                for v in variantes
                if v.get("sobrecritas", {}).get(chave) == valor and v.get("melhor_val") is not None
            ]
            por_valor[str(valor)] = {
                "n": len(medidas),
                "media_melhor_val": round(sum(medidas) / len(medidas), 4) if medidas else None,
                "melhor_val": round(min(medidas), 4) if medidas else None,
            }
        referencia = base_por_chave.get(chave)
        efeito[chave] = {
            "base": referencia,
            "por_valor": por_valor,
            "melhor_valor": (
                min(
                    (v for v in por_valor if por_valor[v]["media_melhor_val"] is not None),
                    key=lambda v: por_valor[v]["media_melhor_val"],
                    default=None,
                )
            ),
        }
    return efeito


def executar(cfg_varrer: ConfigVarredura, executar_treino_fn=executar_treino) -> dict:
    """Roda a varredura inteira (ou só lista, em modo seco) e devolve o relatório."""
    raiz = Path(cfg_varrer.raiz)
    caminho_base = Path(cfg_varrer.base)
    if not caminho_base.is_absolute():
        caminho_base = raiz / caminho_base
    if not caminho_base.exists():
        raise FileNotFoundError(f"config base não encontrada: {caminho_base}")

    cfg_base = ConfigTreino.de_arquivo(caminho_base)
    base_por_chave = {chave: valor_atual(cfg_base, chave) for chave in cfg_varrer.grades}
    variantes = cfg_varrer.variantes or montar_variantes(
        cfg_varrer.grades, cfg_varrer.modo, cfg_varrer.n, cfg_varrer.semente
    )

    relatorio = {
        "versao": 1,
        "prefixo": cfg_varrer.prefixo,
        "base": str(caminho_base),
        "modo": cfg_varrer.modo,
        "semente": cfg_varrer.semente,
        "passos_por_variante": cfg_varrer.passos or cfg_base.passos,
        "criado_em": agora_iso(),
        "grades": {c: [str(v) for v in vals] for c, vals in cfg_varrer.grades.items()},
        "valores_base": {c: base_por_chave[c] for c in cfg_varrer.grades},
        "seco": cfg_varrer.seco,
        "variantes": [],
    }

    if cfg_varrer.seco:
        for indice, sobrescritas in enumerate(variantes, start=1):
            registro = {"indice": indice, "run_id": f"{cfg_varrer.prefixo}-{indice:02d}", "sobrecritas": sobrescritas}
            try:
                _config_da_variante(caminho_base, cfg_varrer, sobrescritas)
            except ValueError as e:
                registro["erro"] = str(e)
            relatorio["variantes"].append(registro)
        relatorio["falhas"] = [v["run_id"] for v in relatorio["variantes"] if v.get("erro")]
        return relatorio

    pasta = raiz / "runs" / "_varredura" / cfg_varrer.prefixo
    pasta.mkdir(parents=True, exist_ok=True)

    for indice, sobrescritas in enumerate(variantes, start=1):
        run_id = f"{cfg_varrer.prefixo}-{indice:02d}"
        registro = {"indice": indice, "run_id": run_id, "sobrecritas": sobrescritas, "erro": None}
        try:
            cfg = _config_da_variante(caminho_base, cfg_varrer, sobrescritas)
            cfg.nome = run_id
            caminho_variante = _yaml_da_variante(cfg, raiz / "configs" / f"{run_id}.yaml")
            registro["config"] = str(caminho_variante.relative_to(raiz))
            # dir_run cria ckpt/ e tokens/ e valida o run-id — o mesmo caminho que o
            # CLI usa; sem isso o primeiro save de checkpoint estoura com
            # "Parent directory ... does not exist" (visto na primeira varredura real).
            run_dir = dir_run(raiz, run_id)
            registros_anteriores = _metricas_existentes(run_dir)
            executar_treino_fn(cfg, run_dir, raiz=raiz)
            resumo = bancada_comparar.resumir_run(run_id, raiz)
            registro.update(
                {
                    "melhor_val": resumo.melhor_val,
                    "passo_melhor_val": resumo.passo_melhor_val,
                    "val_final": resumo.val_final,
                    "drift": resumo.drift,
                    "diagnostico": resumo.diagnostico,
                    "tokens_por_s": resumo.tokens_por_s,
                    "parametros": resumo.parametros,
                    "metricas_antes": registros_anteriores,
                    "reaproveitado": registros_anteriores > 0,
                }
            )
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            registro["erro"] = str(e)
        relatorio["variantes"].append(registro)
        (pasta / "varredura.json").write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8"
        )

    com_dados = [v for v in relatorio["variantes"] if v.get("melhor_val") is not None]
    ranking = sorted(com_dados, key=lambda v: v["melhor_val"])
    relatorio["ranking"] = [v["run_id"] for v in ranking]
    relatorio["melhor"] = (
        {
            "run_id": ranking[0]["run_id"],
            "melhor_val": ranking[0]["melhor_val"],
            "sobrecritas": ranking[0]["sobrecritas"],
        }
        if ranking
        else None
    )
    relatorio["efeito_por_chave"] = _efeito_por_chave(relatorio["variantes"], cfg_varrer.grades, base_por_chave)
    relatorio["falhas"] = [v["run_id"] for v in relatorio["variantes"] if v.get("erro")]
    relatorio["onde_salvou"] = str((pasta / "varredura.json").relative_to(raiz))
    (pasta / "varredura.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8"
    )
    return relatorio


def _metricas_existentes(run_dir: Path) -> int:
    arquivo = run_dir / "metricas.jsonl"
    if not arquivo.exists():
        return 0
    return len([l for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()])


# --- apresentação -----------------------------------------------------------

def tabela_texto(relatorio: dict) -> str:
    linhas = [
        f"varredura : {relatorio['prefixo']}  (modo {relatorio['modo']}, base {relatorio['base']})",
        f"orçamento : {relatorio['passos_por_variante']} passos por variante · semente {relatorio['semente']}",
    ]
    if relatorio.get("seco"):
        linhas.append("seco      : nada foi treinado — só a lista de combinações (tire --seco para rodar)")
    linhas.append("")
    chaves = sorted(relatorio["grades"])
    cabecalho = f"{'run':<18} " + " ".join(f"{c:>10}" for c in chaves) + f" {'melhor val':>11} {'passo':>7} {'veredito':<14}"
    linhas += [cabecalho, "-" * len(cabecalho)]
    for variante in relatorio["variantes"]:
        valores = " ".join(f"{str(variante['sobrecritas'].get(c, '')):>10}" for c in chaves)
        if variante.get("melhor_val") is None:
            situacao = "seco" if relatorio.get("seco") else "FALHOU"
            linhas.append(f"{variante['run_id']:<18} {valores} {'—':>11} {'—':>7} {situacao:<14}")
            if variante.get("erro"):
                linhas.append(f"    erro: {variante['erro']}")
            continue
        linhas.append(
            f"{variante['run_id']:<18} {valores} {variante['melhor_val']:>11.4f} "
            f"{(variante.get('passo_melhor_val') or 0):>7} {variante.get('diagnostico', ''):<14}"
        )
    if relatorio.get("melhor"):
        melhor = relatorio["melhor"]
        linhas += [
            "",
            f"melhor: {melhor['run_id']} — val {melhor['melhor_val']:.4f} com "
            + ", ".join(f"{c}={v}" for c, v in melhor["sobrecritas"].items()),
        ]
    efeito = relatorio.get("efeito_por_chave") or {}
    if efeito:
        linhas += ["", "efeito de cada chave (média da melhor validação por valor):"]
        for chave, dados in efeito.items():
            partes = []
            for valor, medido in dados["por_valor"].items():
                if medido["media_melhor_val"] is None:
                    continue
                marca = " (base)" if str(dados.get("base")) == valor else ""
                partes.append(f"{valor} → {medido['media_melhor_val']:.4f}{marca}")
            linhas.append(f"  {chave}: " + " · ".join(partes))
            if dados.get("melhor_valor"):
                linhas.append(f"    → melhor valor testado: {dados['melhor_valor']}")
    if relatorio.get("onde_salvou"):
        linhas += ["", f"relatório: {relatorio['onde_salvou']}", f"próximo  : lab-ia comparar {' '.join(relatorio.get('ranking', [])[:4])}"]
    return "\n".join(linhas)


def relatorio_markdown(relatorio: dict) -> str:
    chaves = sorted(relatorio["grades"])
    linhas = [
        f"# Varredura {relatorio['prefixo']}",
        "",
        f"- base: {relatorio['base']}",
        f"- orçamento: {relatorio['passos_por_variante']} passos por variante, semente {relatorio['semente']}",
        "- chaves: " + " · ".join(f"{c} em {{{', '.join(relatorio['grades'][c])}}}" for c in chaves),
        "",
        "| run | " + " | ".join(chaves) + " | melhor val | passo | deriva | veredito |",
        "|---|" + "---|" * (len(chaves) + 4),
    ]
    for v in relatorio["variantes"]:
        valores = " | ".join(str(v["sobrecritas"].get(c, "")) for c in chaves)
        if v.get("melhor_val") is None:
            linhas.append(f"| {v['run_id']} | {valores} | — | — | — | FALHOU: {v.get('erro')} |")
            continue
        deriva = "" if v.get("drift") is None else f"{v['drift']:+.4f}"
        linhas.append(
            f"| {v['run_id']} | {valores} | {v['melhor_val']:.4f} | {v.get('passo_melhor_val')} | {deriva} | {v.get('diagnostico')} |"
        )
    if relatorio.get("melhor"):
        m = relatorio["melhor"]
        linhas += ["", f"**Melhor:** {m['run_id']} — val {m['melhor_val']:.4f} com " + ", ".join(f"{c}={v}" for c, v in m["sobrecritas"].items())]
    if relatorio.get("efeito_por_chave"):
        linhas += ["", "## Efeito de cada chave", ""]
        for chave, dados in relatorio["efeito_por_chave"].items():
            linhas.append(f"- **{chave}** (base {dados.get('base')}):")
            for valor, medido in dados["por_valor"].items():
                if medido["media_melhor_val"] is not None:
                    linhas.append(f"  - {valor}: média {medido['media_melhor_val']:.4f} em {medido['n']} variante(s)")
    linhas += [
        "",
        "## Como ler",
        "Comparação só vale com o mesmo orçamento: todas as variantes treinaram o mesmo",
        "número de passos. Diferença menor que o ruído entre sementes não é achado.",
    ]
    return "\n".join(linhas) + "\n"


def salvar_relatorio(relatorio: dict, destino: Path | str) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(relatorio_markdown(relatorio), encoding="utf-8")
    return destino
