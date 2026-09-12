"""Ingestão de corpus próprio: limpeza, deduplicação, split e manifesto auditável.

Substitui o caminho antigo (dois livros do Gutenberg com ID cravado no script) por
um caminho genérico: você aponta para arquivos ou pastas suas, e o laboratório
devolve data/<id>/trem.txt, data/<id>/val.txt e data/<id>/manifesto.json
com hashes, contagens e estatísticas — pronto para o treino consumir.

Decisões de projeto (todas auditáveis no manifesto):
  * codificação detectada por tentativa estrita (utf-8 → cp1252 → latin-1);
  * parágrafos são a unidade de deduplicação e de split;
  * deduplicação exata por chave normalizada + quase-duplicata por Jaccard de
    5-gramas com LSH de balde único (min-hash), limiar configurável;
  * split por parágrafos embaralhados com semente fixa, reusando a mesma função
    do treino (labia.trainer.dados.dividir_corpus) para não haver dois
    comportamentos de split no projeto.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..trainer.dados import dividir_corpus

CODIFICACOES = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
EXTENSOES_TEXTO = (".txt", ".md", ".text", ".rst", ".jsonl")
_ID_OK = re.compile(r"[A-Za-z0-9._-]+")
_PALAVRA = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+")
_CARACTERE_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Palavras funcionais usadas só como pista de idioma no manifesto.
# Quantas chaves de min-hash indexam cada parágrafo (bandas do LSH).
BANDAS = 4

_STOPWORDS = {
    "pt": (" de ", " a ", " o ", " que ", " e ", " do ", " da ", " em ", " um ", " para ", " com ", " não ", " uma ", " os ", " no "),
    "en": (" the ", " of ", " and ", " to ", " in ", " is ", " that ", " it ", " for ", " was ", " with ", " as ", " on ", " be "),
}

LIMITE_MB_PADRAO = 256.0


def agora_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validar_id(identificador: str) -> str:
    """Mesma regra de run-id do treino: sem barra, sem '..', sem surpresa."""
    if not identificador or identificador in (".", "..") or not _ID_OK.fullmatch(identificador):
        raise ValueError(
            f"id inválido: {identificador!r} (use letras, dígitos, ponto, hífen e _)"
        )
    return identificador


def detectar_codificacao(bruto: bytes) -> str:
    """Codificação do arquivo: BOM explícito, senão utf-8 estrito, cp1252, latin-1.

    Ordem importa: utf-8-sig decodifica qualquer utf-8 (o BOM é opcional), então
    testá-lo primeiro rotularia todo arquivo utf-8 comum como "utf-8-sig". Aqui o
    BOM é checado nos bytes, não por tentativa.
    """
    if bruto.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    for codificacao in ("utf-8", "cp1252", "latin-1"):
        try:
            texto = bruto.decode(codificacao)
        except UnicodeDecodeError:
            continue
        if codificacao == "latin-1" or "\ufffd" not in texto:
            return codificacao
    return "latin-1"


def normalizar(texto: str) -> str:
    """Normaliza quebras, remove caracteres de controle e colapsa espaços em branco."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _CARACTERE_CONTROLE.sub("", texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def separar_paragrafos(texto: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]


def resolver_fontes(fontes: list[str], raiz: Path | str = ".") -> list[Path]:
    """Aceita arquivos, pastas (recursivo) e curingas; devolve arquivos de texto ordenados."""
    raiz = Path(raiz)
    achados: list[Path] = []
    for fonte in fontes:
        caminho = Path(fonte)
        if not caminho.is_absolute():
            candidato = raiz / fonte
            caminho = candidato if candidato.exists() else caminho
        if caminho.is_dir():
            achados.extend(
                p
                for p in sorted(caminho.glob("**/*"))
                if p.is_file() and p.suffix.lower() in EXTENSOES_TEXTO
            )
        elif any(c in caminho.name for c in "*?["):
            # curinga (o shell do Windows não expande glob para o processo)
            pai = caminho.parent if str(caminho.parent) not in ("", ".") else raiz
            if not pai.exists():
                pai = raiz
            achados.extend(sorted(p for p in pai.glob(caminho.name) if p.is_file()))
        elif caminho.exists():
            achados.append(caminho)
        else:
            raise FileNotFoundError(f"fonte não encontrada: {fonte}")
    vistos: set[Path] = set()
    unicos: list[Path] = []
    for p in achados:
        chave = p.resolve()
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(p)
    if not unicos:
        raise FileNotFoundError(f"nenhum arquivo de texto encontrado em {fontes}")
    return unicos


def _extrair_jsonl(bruto: str, campo: str | None, caminho: Path) -> str:
    """Lê .jsonl pegando um campo de texto por linha (export de datasets)."""
    candidatos = [campo] if campo else ["texto", "text", "conteudo", "content"]
    partes: list[str] = []
    for numero, linha in enumerate(bruto.splitlines(), start=1):
        linha = linha.strip()
        if not linha:
            continue
        try:
            registro = json.loads(linha)
        except json.JSONDecodeError as e:
            raise ValueError(f"{caminho}: linha {numero} não é JSON válido ({e})") from e
        if not isinstance(registro, dict):
            partes.append(str(registro))
            continue
        escolhido = next((c for c in candidatos if c and isinstance(registro.get(c), str)), None)
        if escolhido is None:
            raise ValueError(
                f"{caminho}: linha {numero} sem campo de texto (procurei {candidatos}; use --campo)"
            )
        partes.append(registro[escolhido])
    return "\n\n".join(partes)


def ler_fonte(caminho: Path, campo: str | None = None) -> dict:
    """Lê um arquivo e devolve texto normalizado + metadados de leitura."""
    if not caminho.exists():
        raise FileNotFoundError(f"fonte não encontrada: {caminho}")
    bruto = caminho.read_bytes()
    if not bruto.strip():
        raise ValueError(f"fonte vazia: {caminho}")
    codificacao = detectar_codificacao(bruto)
    texto = bruto.decode(codificacao, errors="replace")
    if caminho.suffix.lower() == ".jsonl":
        texto = _extrair_jsonl(texto, campo, caminho)
    return {
        "caminho": str(caminho),
        "bytes": len(bruto),
        "codificacao": codificacao,
        "chars_brutos": len(texto),
        "texto": normalizar(texto),
    }


def _hash_texto(texto: str) -> str:
    return hashlib.blake2b(texto.encode("utf-8"), digest_size=8).hexdigest()


def _hash_int(texto: str) -> int:
    """Hash estável entre processos (hash() de str é aleatorizado por PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.blake2b(texto.encode("utf-8"), digest_size=8).digest(), "big")


def _chave_exata(paragrafo: str) -> str:
    return _hash_texto(" ".join(_PALAVRA.findall(paragrafo.lower())))


def _shingles(paragrafo: str, n: int = 5) -> set[int]:
    palavras = _PALAVRA.findall(paragrafo.lower())
    if len(palavras) < n:
        return {_hash_int(" ".join(palavras))} if palavras else set()
    return {_hash_int(" ".join(palavras[i : i + n])) for i in range(len(palavras) - n + 1)}


def deduplicar(
    paragrafos: list[str], limiar: float = 0.8, quase: bool = True, max_balde: int = 64
) -> tuple[list[str], dict]:
    """Remove repetidos exatos e quase-repetidos (Jaccard de 5-gramas >= limiar).

    Semântica das duas passadas, de propósito:
      * EXATA compara a sequência de palavras ignorando caixa e pontuação — pega
        "fim." contra "fim!" e diferenças de acento gráfico na pontuação;
      * QUASE compara conjuntos de 5-gramas de palavras. Com limiar 0,8 só passa
        boilerplate praticamente idêntico (troca de uma ou duas palavras em texto
        longo). É conservador de propósito: fundir parágrafos diferentes apaga
        dados em silêncio, e apagar dado é pior que treinar com repetido. Baixe
        --limiar-quase para limpar corpus raspado com mais agressividade.

    Quase-duplicata usa LSH de balde único: só compara parágrafos que compartilham
    o menor min-hash, o que evita O(n^2) sem perder os pares realmente parecidos.
    """
    mantidos: list[str] = []
    exatos: set[str] = set()
    baldes: dict[int, list[set[int]]] = {}
    removidos = {"exatas": 0, "quase": 0}
    for paragrafo in paragrafos:
        chave = _chave_exata(paragrafo)
        if chave in exatos:
            removidos["exatas"] += 1
            continue
        exatos.add(chave)
        if quase:
            assinatura = _shingles(paragrafo)
            if assinatura:
                # LSH com BANDAS chaves: indexar só pelo menor min-hash erra o par
                # quando o menor shingle de um dos dois é justamente o que mudou.
                chaves = sorted(assinatura)[:BANDAS]
                candidatos: list[set[int]] = []
                for chave_banda in chaves:
                    candidatos.extend(baldes.get(chave_banda, ()))
                if any(
                    len(assinatura & outro) / len(assinatura | outro) >= limiar
                    for outro in candidatos[:max_balde]
                ):
                    removidos["quase"] += 1
                    continue
                for chave_banda in chaves:
                    baldes.setdefault(chave_banda, []).append(assinatura)
        mantidos.append(paragrafo)
    return mantidos, removidos


def estatisticas(texto: str) -> dict:
    """Retrato do corpus: volume, vocabulário, entropia de caracteres e pista de idioma."""
    palavras = [p.lower() for p in _PALAVRA.findall(texto)]
    contagem = Counter(palavras)
    total_chars = max(1, len(texto))
    frequencias = Counter(texto)
    entropia = -sum((n / total_chars) * math.log2(n / total_chars) for n in frequencias.values())
    minusculo = " " + texto.lower() + " "
    pistas = {
        lingua: sum(minusculo.count(marca) for marca in marcas)
        for lingua, marcas in _STOPWORDS.items()
    }
    idioma = max(pistas, key=lambda k: pistas[k]) if any(pistas.values()) else "indefinido"
    return {
        "chars": len(texto),
        "linhas": texto.count("\n") + 1,
        "palavras": len(palavras),
        "palavras_unicas": len(contagem),
        "chars_por_palavra": round(len(texto) / max(1, len(palavras)), 2),
        "entropia_caractere_bits": round(entropia, 3),
        "fracao_ascii": round(sum(1 for c in texto if ord(c) < 128) / total_chars, 4),
        "idioma_provavel": idioma,
        "pistas_idioma": pistas,
        "top_palavras": contagem.most_common(20),
    }


def estimar_tokens(chars: int, chars_por_token: float = 3.6) -> int:
    """Estimativa grosseira de tokens BPE pt-BR (3,6 chars/token é a média observada).

    Serve para dimensionar passos/lote antes de treinar o tokenizador de verdade;
    o número real sai do tokenizer treinado no run.
    """
    return int(chars / max(0.5, chars_por_token))


@dataclass
class ConfigPreparo:
    """Parâmetros do preparo; tudo que muda o resultado vai para o manifesto."""

    fontes: list[str]
    id: str
    destino: str = "data"
    frac_val: float = 0.05
    semente: int = 42
    min_chars_paragrafo: int = 20  # 20 chars limpa lixo sem comer linhas curtas de diálogo
    quase_duplicata: bool = True
    limiar_quase: float = 0.8
    campo_jsonl: str | None = None
    limite_mb: float = LIMITE_MB_PADRAO
    forcar: bool = False
    janela_referencia: int = 256
    raiz: str = "."

    def __post_init__(self) -> None:
        validar_id(self.id)
        if not 0.0 < self.frac_val < 0.5:
            raise ValueError("frac_val deve ficar entre 0 e 0.5 (validação pequena)")
        if self.min_chars_paragrafo < 0:
            raise ValueError("min_chars_paragrafo não pode ser negativo")

    def para_dict(self) -> dict:
        return {
            "fontes": list(self.fontes),
            "id": self.id,
            "destino": self.destino,
            "frac_val": self.frac_val,
            "semente": self.semente,
            "min_chars_paragrafo": self.min_chars_paragrafo,
            "quase_duplicata": self.quase_duplicata,
            "limiar_quase": self.limiar_quase,
            "campo_jsonl": self.campo_jsonl,
            "limite_mb": self.limite_mb,
            "janela_referencia": self.janela_referencia,
        }


def preparar(cfg: ConfigPreparo) -> dict:
    """Executa o preparo completo e devolve o manifesto (também gravado em disco)."""
    raiz = Path(cfg.raiz)
    fontes = resolver_fontes(cfg.fontes, raiz)
    lidos = [ler_fonte(p, cfg.campo_jsonl) for p in fontes]
    total_mb = sum(lido["bytes"] for lido in lidos) / (1024 * 1024)
    if total_mb > cfg.limite_mb and not cfg.forcar:
        raise ValueError(
            f"fontes somam {total_mb:.1f} MB > limite {cfg.limite_mb:.0f} MB "
            f"(use --forcar ou --limite-mb para assumir)"
        )

    texto = normalizar("\n\n".join(lido["texto"] for lido in lidos))
    paragrafos = separar_paragrafos(texto)
    if not paragrafos:
        raise ValueError("nenhum parágrafo utilizável depois da limpeza")

    acima_minimo = [p for p in paragrafos if len(p) >= cfg.min_chars_paragrafo]
    descartados_curtos = len(paragrafos) - len(acima_minimo)
    utilizaveis = acima_minimo or paragrafos  # corpus curto não pode virar vazio

    mantidos, removidos = deduplicar(
        utilizaveis, limiar=cfg.limiar_quase, quase=cfg.quase_duplicata
    )
    if len(mantidos) < 2:
        raise ValueError(
            f"sobrou {len(mantidos)} parágrafo após limpeza/dedup — corpus pequeno ou repetitivo demais"
        )

    trem, val = dividir_corpus(
        "\n\n".join(mantidos), frac_trem=1.0 - cfg.frac_val, semente=cfg.semente
    )

    destino = Path(cfg.destino)
    if not destino.is_absolute():
        destino = raiz / destino
    pasta = destino / cfg.id
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_trem = pasta / "trem.txt"
    arquivo_val = pasta / "val.txt"
    arquivo_trem.write_text(trem + "\n", encoding="utf-8")
    arquivo_val.write_text(val + "\n", encoding="utf-8")

    est_trem = estatisticas(trem)
    est_val = estatisticas(val)
    vocabulario_val = set(_PALAVRA.findall(val.lower()))
    vocabulario_trem = set(_PALAVRA.findall(trem.lower()))

    def _relativo(p: Path) -> str:
        try:
            return str(p.relative_to(raiz))
        except ValueError:
            return str(p)

    tokens_trem = estimar_tokens(len(trem))
    janela = max(1, cfg.janela_referencia)
    manifesto = {
        "versao": 1,
        "id": cfg.id,
        "gerado_em": agora_iso(),
        "parametros": cfg.para_dict(),
        "fontes": [
            {
                "caminho": lido["caminho"],
                "bytes": lido["bytes"],
                "codificacao": lido["codificacao"],
                "chars_brutos": lido["chars_brutos"],
                "sha256": hashlib.sha256(Path(lido["caminho"]).read_bytes()).hexdigest()[:16],
            }
            for lido in lidos
        ],
        "limpeza": {
            "paragrafos_brutos": len(paragrafos),
            "descartados_por_tamanho": descartados_curtos,
            "removidas_exatas": removidos["exatas"],
            "removidas_quase": removidos["quase"],
            "paragrafos_finais": len(mantidos),
        },
        "saidas": {
            "trem": {
                "arquivo": _relativo(arquivo_trem),
                "sha256": hashlib.sha256(arquivo_trem.read_bytes()).hexdigest()[:16],
                "chars": len(trem),
                "paragrafos": len(separar_paragrafos(trem)),
            },
            "val": {
                "arquivo": _relativo(arquivo_val),
                "sha256": hashlib.sha256(arquivo_val.read_bytes()).hexdigest()[:16],
                "chars": len(val),
                "paragrafos": len(separar_paragrafos(val)),
            },
        },
        "estatisticas": {"trem": est_trem, "val": est_val},
        "vazamento_val_para_trem": {
            "palavras_val_no_trem": round(
                len(vocabulario_val & vocabulario_trem) / max(1, len(vocabulario_val)), 4
            ),
            "nota": "fração do vocabulário da validação que já aparece no treino (normal em corpus pequeno)",
        },
        "estimativas": {
            "tokens_aprox_trem": tokens_trem,
            "tokens_aprox_val": estimar_tokens(len(val)),
            "janelas_ctx": {
                str(janela): max(1, (tokens_trem - janela - 1) // janela + 1),
            },
            "nota_tokens": "estimativa por 3,6 chars/token; o número real sai do tokenizer do run",
        },
    }

    (pasta / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifesto["pasta"] = str(pasta)
    return manifesto


def carregar(identificador: str, raiz: Path | str = ".") -> tuple[str, str, dict]:
    """Carrega um dataset preparado: (texto_trem, texto_val, manifesto)."""
    validar_id(identificador)
    pasta = Path(raiz) / "data" / identificador
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    trem = (pasta / "trem.txt").read_text(encoding="utf-8")
    val = (pasta / "val.txt").read_text(encoding="utf-8")
    return trem, val, manifesto


def listar(raiz: Path | str = ".") -> list[dict]:
    """Lista os datasets preparados (pastas com manifesto.json dentro de data/)."""
    base = Path(raiz) / "data"
    if not base.exists():
        return []
    saida = []
    for pasta in sorted(base.iterdir()):
        arquivo_manifesto = pasta / "manifesto.json"
        if pasta.is_dir() and arquivo_manifesto.exists():
            manifesto = json.loads(arquivo_manifesto.read_text(encoding="utf-8"))
            saida.append(
                {
                    "id": pasta.name,
                    "gerado_em": manifesto.get("gerado_em"),
                    "chars_trem": manifesto["saidas"]["trem"]["chars"],
                    "chars_val": manifesto["saidas"]["val"]["chars"],
                    "idioma": manifesto["estatisticas"]["trem"]["idioma_provavel"],
                    "tokens_aprox_trem": manifesto["estimativas"]["tokens_aprox_trem"],
                    "fontes": [f["caminho"] for f in manifesto.get("fontes", [])],
                }
            )
    return saida
