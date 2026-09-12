"""Trilha de estudo: lições mapeadas ao código do núcleo, com experimentos executáveis.

Material didático apodrece em silêncio: alguém renomeia uma função e a apostila
continua citando o nome antigo por anos. Aqui as lições citam arquivo e símbolo num
formato conferível, e 'conferir_referencias' quebra quando a citação deixa de existir.
É o mesmo princípio do resto do laboratório: afirmação sem verificação vira folclore.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# 'caminho/arquivo.py' → 'Simbolo.metodo' — o formato que as lições usam para citar código
_PADRAO_REFERENCIA = re.compile(r"`([A-Za-z0-9_./\\-]+\.py)`\s*→\s*`([A-Za-z_][A-Za-z0-9_.]*)`".replace("`", chr(96)))
_PADRAO_ARQUIVO_PY = re.compile(r"`([A-Za-z0-9_./\\-]+\.py)`".replace("`", chr(96)))
_PADRAO_EXPERIMENTO = re.compile(r"(e\d+_[a-z_]+\.py)")
_PADRAO_TITULO = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def raiz_do_projeto() -> Path:
    """Raiz do repositório a partir deste arquivo (core/labia/trilha.py)."""
    return Path(__file__).resolve().parents[2]


@dataclass
class Licao:
    numero: int
    titulo: str
    arquivo: Path
    experimento: Path | None
    resumo: str = ""

    def para_dict(self) -> dict:
        return {
            "numero": self.numero,
            "titulo": self.titulo,
            "arquivo": str(self.arquivo),
            "experimento": str(self.experimento) if self.experimento else None,
        }


def _resumo_do_texto(texto: str) -> str:
    """Primeira frase útil depois do título, para a listagem da CLI."""
    corpo = texto.split("\n", 1)[1] if "\n" in texto else ""
    for linha in corpo.splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and not linha.startswith("**"):
            return linha[:160]
    return ""


def licoes(raiz: Path | str | None = None) -> list[Licao]:
    """Lê trilha/licoes/*.md em ordem e liga cada uma ao seu experimento."""
    base = Path(raiz) if raiz else raiz_do_projeto()
    pasta = base / "trilha" / "licoes"
    if not pasta.is_dir():
        return []
    encontradas: list[Licao] = []
    for ordem, arquivo in enumerate(sorted(pasta.glob("*.md")), start=1):
        texto = arquivo.read_text(encoding="utf-8")
        titulo = (_PADRAO_TITULO.search(texto).group(1) if _PADRAO_TITULO.search(texto) else arquivo.stem)
        numero_texto = re.match(r"(\d+)", arquivo.stem)
        numero = int(numero_texto.group(1)) if numero_texto else ordem
        achado = _PADRAO_EXPERIMENTO.search(texto)
        experimento = base / "trilha" / "experimentos" / achado.group(1) if achado else None
        encontradas.append(
            Licao(numero=numero, titulo=titulo, arquivo=arquivo, experimento=experimento, resumo=_resumo_do_texto(texto))
        )
    return sorted(encontradas, key=lambda l: l.numero)


def achar_licao(numero: int, raiz: Path | str | None = None) -> Licao:
    for licao in licoes(raiz):
        if licao.numero == numero:
            return licao
    disponiveis = [l.numero for l in licoes(raiz)]
    raise ValueError(f"lição {numero} não existe (disponíveis: {disponiveis})")


def ler_licao(numero: int, raiz: Path | str | None = None) -> str:
    return achar_licao(numero, raiz).arquivo.read_text(encoding="utf-8")


def rodar_experimento(numero: int, raiz: Path | str | None = None, capturar: bool = True):
    """Roda o experimento da lição em subprocesso — isolado e igual ao que o aluno faz."""
    base = Path(raiz) if raiz else raiz_do_projeto()
    licao = achar_licao(numero, base)
    if licao.experimento is None or not licao.experimento.exists():
        raise FileNotFoundError(f"lição {numero} não tem experimento em trilha/experimentos/")
    return subprocess.run(
        [sys.executable, str(licao.experimento)],
        cwd=str(base),
        capture_output=capturar,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def rodar_todos(raiz: Path | str | None = None) -> list[dict]:
    """Roda todos os experimentos, em ordem, devolvendo o resultado de cada um."""
    resultados = []
    for licao in licoes(raiz):
        if licao.experimento is None or not licao.experimento.exists():
            continue
        processo = rodar_experimento(licao.numero, raiz)
        resultados.append(
            {
                "licao": licao.numero,
                "titulo": licao.titulo,
                "experimento": licao.experimento.name,
                "codigo": processo.returncode,
                "saida": processo.stdout,
                "erro": processo.stderr,
            }
        )
    return resultados


# --- conferência das citações ao código ------------------------------------

def _simbolo_existe(caminho: Path, simbolo: str) -> bool:
    """Confere pelo AST se 'Classe.metodo' (ou um nome solto) existe no arquivo."""
    try:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    partes = simbolo.split(".")

    def nomes(corpo) -> set[str]:
        encontrados = set()
        for no in corpo:
            if isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                encontrados.add(no.name)
            elif isinstance(no, ast.Assign):
                for alvo in no.targets:
                    if isinstance(alvo, ast.Name):
                        encontrados.add(alvo.id)
            elif isinstance(no, ast.AnnAssign) and isinstance(no.target, ast.Name):
                encontrados.add(no.target.id)
            elif isinstance(no, ast.If):
                encontrados |= nomes(no.body) | nomes(no.orelse)
            elif isinstance(no, ast.Try):
                encontrados |= nomes(no.body) | nomes(no.orelse) | nomes(no.finalbody)
        return encontrados

    if len(partes) == 1:
        return partes[0] in nomes(arvore.body)
    for no in arvore.body:
        if isinstance(no, ast.ClassDef) and no.name == partes[0]:
            return partes[1] in nomes(no.body)
    return False


def _resolver_arquivo(referencia: str, base: Path) -> Path | None:
    """Resolve 'core/labia/models/gpt.py' ou só 'gpt.py' (busca por nome)."""
    candidato = base / referencia
    if candidato.exists():
        return candidato
    nome = Path(referencia).name
    for achado in base.rglob(nome):
        if achado.is_file():
            return achado
    return None


def conferir_referencias(raiz: Path | str | None = None) -> list[dict]:
    """Lista problemas nas citações das lições: arquivo sumiu ou símbolo não existe mais."""
    base = Path(raiz) if raiz else raiz_do_projeto()
    problemas: list[dict] = []
    for licao in licoes(base):
        texto = licao.arquivo.read_text(encoding="utf-8")
        for arquivo_citado, simbolo in _PADRAO_REFERENCIA.findall(texto):
            resolvido = _resolver_arquivo(arquivo_citado, base)
            if resolvido is None:
                problemas.append({"licao": licao.numero, "arquivo": arquivo_citado, "motivo": "arquivo não existe"})
                continue
            if simbolo and not _simbolo_existe(resolvido, simbolo):
                problemas.append(
                    {
                        "licao": licao.numero,
                        "arquivo": arquivo_citado,
                        "simbolo": simbolo,
                        "motivo": "símbolo não existe mais no arquivo",
                    }
                )
        for arquivo_citado in set(_PADRAO_ARQUIVO_PY.findall(texto)):
            if _resolver_arquivo(arquivo_citado, base) is None:
                problemas.append({"licao": licao.numero, "arquivo": arquivo_citado, "motivo": "arquivo não existe"})
    return problemas


def indice(raiz: Path | str | None = None) -> dict:
    """Estrutura da trilha, para a CLI e para a API."""
    todas = licoes(raiz)
    return {
        "total": len(todas),
        "licoes": [
            {**licao.para_dict(), "resumo": licao.resumo, "tem_experimento": bool(licao.experimento and licao.experimento.exists())}
            for licao in todas
        ],
    }
