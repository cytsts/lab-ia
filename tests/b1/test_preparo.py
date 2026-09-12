"""Testes da bancada B1 — ingestão de corpus próprio (specs/B1.md).

Cobrem: detecção de codificação, normalização, resolução de fontes (pasta/curinga),
leitura de .jsonl, deduplicação exata e quase-duplicata, determinismo entre
processos, split, manifesto, guardas de segurança e as funções de listagem/carga.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labia.bancada import dados as bd
from common import gerar_corpus


# --- leitura e normalização -------------------------------------------------

def test_detectar_codificacao_utf8_e_variantes():
    assert bd.detectar_codificacao("ação ção".encode("utf-8")) == "utf-8"
    assert bd.detectar_codificacao("ação".encode("utf-8-sig")) == "utf-8-sig"
    # cp1252 decodifica bytes que o utf-8 estrito recusa
    assert bd.detectar_codificacao("ação ção".encode("cp1252")) == "cp1252"


def test_normalizar_colapsa_espacos_e_remove_controle():
    bruto = "a  b\t\tc\r\n\r\n\r\n\r\nd\x00e   \n"
    saida = bd.normalizar(bruto)
    assert "\x00" not in saida
    assert "\r" not in saida
    assert "\n\n\n" not in saida
    assert saida == "a b c\n\nde"  # o \x00 sai e cola as letras vizinhas


def test_separar_paragrafos_ignora_vazios():
    assert bd.separar_paragrafos("um\n\n\n\ndois\n   \n\n") == ["um", "dois"]


def test_resolver_fontes_pasta_recursiva_e_curinga(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("conteudo a", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("conteudo b", encoding="utf-8")
    (tmp_path / "sub" / "ignorar.bin").write_bytes(b"\x00\x01")
    achados = bd.resolver_fontes([str(tmp_path)], tmp_path)
    assert {p.name for p in achados} == {"a.txt", "b.md"}
    curinga = bd.resolver_fontes([str(tmp_path / "*.txt")], tmp_path)
    assert [p.name for p in curinga] == ["a.txt"]


def test_resolver_fontes_inexistente_falha(tmp_path):
    with pytest.raises(FileNotFoundError):
        bd.resolver_fontes([str(tmp_path / "nao-existe.txt")], tmp_path)


def test_ler_fonte_jsonl_pega_campo(tmp_path):
    arquivo = tmp_path / "dados.jsonl"
    arquivo.write_text(
        json.dumps({"texto": "primeira linha"}) + "\n" + json.dumps({"text": "segunda linha"}) + "\n",
        encoding="utf-8",
    )
    lido = bd.ler_fonte(arquivo)
    assert "primeira linha" in lido["texto"]
    assert "segunda linha" in lido["texto"]


def test_ler_fonte_jsonl_sem_campo_de_texto_falha(tmp_path):
    arquivo = tmp_path / "ruim.jsonl"
    arquivo.write_text(json.dumps({"outro": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sem campo de texto"):
        bd.ler_fonte(arquivo)


def test_ler_fonte_vazia_falha(tmp_path):
    arquivo = tmp_path / "vazio.txt"
    arquivo.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="vazia"):
        bd.ler_fonte(arquivo)


# --- deduplicação -----------------------------------------------------------

def test_deduplicar_exata_ignora_caixa_e_pontuacao():
    """A chave exata é a sequência de palavras: 'fim.' e 'FIM!' são o mesmo parágrafo."""
    paragrafos = [
        "O mar batia forte contra as pedras do quebra-mar naquela manha fria de inverno.",
        "O mar batia forte contra as pedras do quebra-mar naquela manha fria de inverno.",  # idêntica
        "o MAR batia forte contra as pedras do quebra-mar, naquela manha fria de inverno!",  # caixa/pontuação
        "Um texto completamente diferente sobre outra coisa qualquer neste laboratorio.",
    ]
    mantidos, removidos = bd.deduplicar(paragrafos, quase=True)
    assert len(mantidos) == 2
    assert removidos["exatas"] == 2
    assert removidos["quase"] == 0


def test_deduplicar_quase_pega_paragrafo_repetido_com_edicao():
    """Boilerplate: mesmo bloco longo repetido com um pedaço trocado no fim."""
    corpo = " ".join(f"palavra{i}" for i in range(60))
    paragrafos = [corpo + " encerramento original", corpo + " encerramento trocado", corpo + " encerramento original"]
    mantidos, removidos = bd.deduplicar(paragrafos, limiar=0.5, quase=True)
    assert len(mantidos) == 1  # sobra só o primeiro; os outros dois são repetição
    assert removidos["exatas"] == 1  # o terceiro é igual ao primeiro
    assert removidos["quase"] == 1  # o segundo é quase-duplicata do primeiro


def test_deduplicar_sem_quase_preserva_quase_duplicatas():
    corpo = " ".join(f"palavra{i}" for i in range(60))
    paragrafos = [corpo + " original", corpo + " trocado"]
    mantidos, removidos = bd.deduplicar(paragrafos, limiar=0.5, quase=False)
    assert len(mantidos) == 2
    assert removidos == {"exatas": 0, "quase": 0}


def test_hash_interno_e_estavel_entre_processos(tmp_path):
    """PYTHONHASHSEED diferente não pode mudar o resultado: hash() de str seria aleatório."""
    codigo = (
        "from labia.bancada.dados import _hash_int, deduplicar;"
        "print(_hash_int('paragrafo de teste'), len(deduplicar(['a b c d e f g h i j']*3)[0]))"
    )
    saidas = []
    for semente in ("1", "4242"):
        ambiente = dict(os.environ, PYTHONHASHSEED=semente)
        resultado = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, env=ambiente
        )
        assert resultado.returncode == 0, resultado.stderr
        saidas.append(resultado.stdout.strip())
    assert saidas[0] == saidas[1]
    assert saidas[0].split()[1] == "1"


# --- preparo completo -------------------------------------------------------

def _preparar(tmp_path, **extras):
    fonte = tmp_path / "fonte.txt"
    fonte.write_text(gerar_corpus(120, semente=3), encoding="utf-8")
    cfg = bd.ConfigPreparo(fontes=[str(fonte)], id="teste", destino="data", raiz=str(tmp_path), **extras)
    return bd.preparar(cfg), tmp_path / "data" / "teste"


def test_preparar_gera_trem_val_e_manifesto(tmp_path):
    manifesto, pasta = _preparar(tmp_path)
    assert (pasta / "trem.txt").exists() and (pasta / "val.txt").exists()
    gravado = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    assert gravado["id"] == "teste"
    assert gravado["saidas"]["trem"]["sha256"] == manifesto["saidas"]["trem"]["sha256"]
    assert gravado["limpeza"]["paragrafos_finais"] > 0
    assert gravado["estatisticas"]["trem"]["idioma_provavel"] == "pt"
    assert 0.0 <= gravado["vazamento_val_para_trem"]["palavras_val_no_trem"] <= 1.0


def test_preparar_e_deterministico(tmp_path):
    primeiro, _ = _preparar(tmp_path)
    segundo, _ = _preparar(tmp_path)
    assert primeiro["saidas"]["trem"]["sha256"] == segundo["saidas"]["trem"]["sha256"]


def test_preparar_respeita_frac_val(tmp_path):
    manifesto, pasta = _preparar(tmp_path, frac_val=0.25)
    trem = (pasta / "trem.txt").read_text(encoding="utf-8")
    val = (pasta / "val.txt").read_text(encoding="utf-8")
    assert len(val) > 0
    assert len(val) / (len(val) + len(trem)) > 0.15  # 25% pedido, tolerância de parágrafos
    assert manifesto["parametros"]["frac_val"] == 0.25


def test_preparar_descarta_paragrafos_curtos_mas_nunca_zera(tmp_path):
    fonte = tmp_path / "curto.txt"
    fonte.write_text("oi\n\nola\n\ntudo bem\n\nbom dia\n\nate logo\n", encoding="utf-8")
    cfg = bd.ConfigPreparo(
        fontes=[str(fonte)], id="curto", destino="data", raiz=str(tmp_path), min_chars_paragrafo=1000
    )
    manifesto = bd.preparar(cfg)  # não pode estourar: cai no plano B (usa todos)
    assert manifesto["limpeza"]["paragrafos_finais"] >= 2


def test_preparar_fonte_pequena_demais_falha(tmp_path):
    fonte = tmp_path / "um.txt"
    fonte.write_text("uma linha unica sem nada mais", encoding="utf-8")
    cfg = bd.ConfigPreparo(fontes=[str(fonte)], id="um", destino="data", raiz=str(tmp_path))
    with pytest.raises(ValueError, match="corpus pequeno"):
        bd.preparar(cfg)


def test_preparar_limite_mb_exige_forcar(tmp_path):
    fonte = tmp_path / "grande.txt"
    fonte.write_text(gerar_corpus(60, semente=9), encoding="utf-8")
    apertado = bd.ConfigPreparo(
        fontes=[str(fonte)], id="grande", destino="data", raiz=str(tmp_path), limite_mb=0.0001
    )
    with pytest.raises(ValueError, match="limite"):
        bd.preparar(apertado)
    folgado = bd.ConfigPreparo(
        fontes=[str(fonte)], id="grande", destino="data", raiz=str(tmp_path), limite_mb=0.0001, forcar=True
    )
    assert bd.preparar(folgado)["id"] == "grande"


@pytest.mark.parametrize("identificador", ["", "..", "a/b", "a\\b", "com espaco"])
def test_validar_id_rejeita_perigoso(identificador):
    with pytest.raises(ValueError):
        bd.validar_id(identificador)


def test_config_preparo_valida_frac_val():
    with pytest.raises(ValueError, match="frac_val"):
        bd.ConfigPreparo(fontes=["x"], id="ok", frac_val=0.9)


def test_carregar_e_listar_roundtrip(tmp_path):
    _, pasta = _preparar(tmp_path)
    trem, val, manifesto = bd.carregar("teste", tmp_path)
    assert trem == (pasta / "trem.txt").read_text(encoding="utf-8")
    assert val == (pasta / "val.txt").read_text(encoding="utf-8")
    assert manifesto["id"] == "teste"
    listados = bd.listar(tmp_path)
    assert [d["id"] for d in listados] == ["teste"]
    assert listados[0]["tokens_aprox_trem"] > 0


def test_listar_sem_data_devolve_vazio(tmp_path):
    assert bd.listar(tmp_path) == []


def test_estimar_tokens_usa_chars_por_token():
    assert bd.estimar_tokens(360) == 100
    assert bd.estimar_tokens(0) == 0
