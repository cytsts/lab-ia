"""Testes da bancada B5 — varredura de hiperparâmetros (specs/B5.md)."""
from __future__ import annotations

import json

import pytest
import yaml

from labia.bancada import varrer as bv
from labia.trainer.treino import ConfigTreino
from common import gerar_corpus


@pytest.fixture()
def base_micro(tmp_path, corpus_arquivo):
    """Config base pequena e rápida, no formato que o CLI gera."""
    pasta = tmp_path / "configs"
    pasta.mkdir()
    cfg = {
        "nome": "base",
        "corpus": str(corpus_arquivo),
        "vocab_bpe": 256,
        "modelo": {"dim": 64, "camadas": 2, "cabecas": 2, "janela_ctx": 64, "abandono": 0.0},
        "passos": 12,
        "lote": 2,
        "stride": 0,
        "avaliar_a_cada": 4,
        "salvar_a_cada": 4,
        "iters_avaliacao": 2,
        "lr": 1e-3,
        "minimo_lr": 1e-4,
        "warmup": 2,
        "peso_decay": 0.0,
        "grad_clip": 1.0,
        "semente": 42,
        "dispositivo": "cpu",
        "arquivo_eventos": str(tmp_path / "eventos.jsonl"),
    }
    caminho = pasta / "base.yaml"
    caminho.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return caminho


# --- leitura das grades -----------------------------------------------------

def test_interpretar_grades_coage_tipo_da_base(base_micro):
    cfg = ConfigTreino.de_arquivo(base_micro)
    grades = bv.interpretar_grades(["lr=0.0001,0.0003", "lote=4,8"], cfg)
    assert grades["lr"] == [0.0001, 0.0003]
    assert grades["lote"] == [4, 8]
    assert all(isinstance(v, float) for v in grades["lr"])
    assert all(isinstance(v, int) for v in grades["lote"])


def test_interpretar_grades_recusa_o_que_nao_e_varredura(base_micro):
    cfg = ConfigTreino.de_arquivo(base_micro)
    with pytest.raises(ValueError, match="formato chave=v1,v2"):
        bv.interpretar_grades(["lr"], cfg)
    with pytest.raises(ValueError, match="não varreível"):
        bv.interpretar_grades(["turbo=1,2"], cfg)
    with pytest.raises(ValueError, match="sem valores"):
        bv.interpretar_grades(["lr="], cfg)
    with pytest.raises(ValueError, match="valor repetido"):
        bv.interpretar_grades(["lr=0.1,0.1"], cfg)
    with pytest.raises(ValueError, match="um valor só"):
        bv.interpretar_grades(["lr=0.1"], cfg)
    with pytest.raises(ValueError, match="nenhuma grade"):
        bv.interpretar_grades([], cfg)


def test_valor_atual_le_campo_do_modelo_e_do_treino(base_micro):
    cfg = ConfigTreino.de_arquivo(base_micro)
    assert bv.valor_atual(cfg, "dim") == 64
    assert bv.valor_atual(cfg, "lote") == 2
    with pytest.raises(ValueError, match="não varreível"):
        bv.valor_atual(cfg, "corpus")


def test_montar_variantes_grade_e_aleatorio():
    grades = {"lr": [0.1, 0.2], "lote": [4, 8]}
    grade = bv.montar_variantes(grades, "grade")
    assert len(grade) == 4
    assert {"lr": 0.1, "lote": 4} in grade
    assert {"lr": 0.2, "lote": 8} in grade

    sorteio = bv.montar_variantes(grades, "aleatorio", n=2, semente=1)
    assert len(sorteio) == 2
    assert bv.montar_variantes(grades, "aleatorio", n=2, semente=1) == sorteio  # determinístico
    assert bv.montar_variantes(grades, "aleatorio", n=99) == grade  # n maior que o espaço


def test_montar_variantes_limites():
    with pytest.raises(ValueError, match="modo desconhecido"):
        bv.montar_variantes({"lr": [1, 2]}, "turbinado")
    with pytest.raises(ValueError, match="teto"):
        bv.montar_variantes({"lr": list(range(20)), "lote": list(range(20))}, "grade")
    with pytest.raises(ValueError, match="--n"):
        bv.montar_variantes({"lr": [1, 2]}, "aleatorio", n=0)


# --- execução ---------------------------------------------------------------

def _varredura(tmp_path, base_micro, **extras):
    cfg = ConfigTreino.de_arquivo(base_micro)
    parametros = {
        "base": str(base_micro),
        "grades": bv.interpretar_grades(["lr=0.0005,0.002"], cfg),
        "prefixo": "sw",
        "raiz": str(tmp_path),
    }
    parametros.update(extras)
    return bv.executar(bv.ConfigVarredura(**parametros))


def test_modo_seco_nao_treina_e_valida_a_grade(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, seco=True)
    assert relatorio["seco"] is True
    assert len(relatorio["variantes"]) == 2
    assert not (tmp_path / "runs").exists()

    # grade impossível é pega no seco, antes de gastar GPU: a base micro tem 2 cabeças,
    # então dim ímpar não fecha
    cfg = ConfigTreino.de_arquivo(base_micro)
    ruim = bv.executar(
        bv.ConfigVarredura(
            base=str(base_micro),
            grades=bv.interpretar_grades(["dim=64,65"], cfg),
            prefixo="sw-ruim",
            raiz=str(tmp_path),
            seco=True,
        )
    )
    assert ruim["falhas"] == ["sw-ruim-02"]
    assert "divisível" in ruim["variantes"][1]["erro"]
    assert ruim["variantes"][0].get("erro") is None


def test_varredura_real_treina_cada_variante_e_ranqueia(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, passos=12)
    assert relatorio["passos_por_variante"] == 12
    assert [v["run_id"] for v in relatorio["variantes"]] == ["sw-01", "sw-02"]
    for variante in relatorio["variantes"]:
        assert variante["erro"] is None
        assert variante["melhor_val"] is not None
        assert (tmp_path / variante["config"]).exists()
        assert (tmp_path / "runs" / variante["run_id"] / "metricas.jsonl").exists()
    assert relatorio["ranking"][0] == relatorio["melhor"]["run_id"]
    assert relatorio["melhor"]["sobrecritas"]["lr"] in (0.0005, 0.002)


def test_config_da_variante_e_reproduzivel_fora_da_varredura(tmp_path, base_micro):
    """A variante vira um run normal: 'lab-ia train --config configs/sw-01.yaml' funciona."""
    relatorio = _varredura(tmp_path, base_micro, passos=8)
    caminho = tmp_path / relatorio["variantes"][0]["config"]
    lido = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    assert lido["nome"] == "sw-01"
    assert lido["passos"] == 8
    cfg = ConfigTreino.de_arquivo(caminho)  # levanta se o YAML escrito não for válido
    assert cfg.passos == 8
    assert cfg.semente == 42


def test_efeito_por_chave_agrega_medias(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, passos=8)
    efeito = relatorio["efeito_por_chave"]["lr"]
    assert set(efeito["por_valor"]) == {"0.0005", "0.002"}
    assert efeito["por_valor"]["0.0005"]["n"] == 1
    assert efeito["por_valor"]["0.0005"]["media_melhor_val"] is not None
    assert efeito["base"] == 1e-3
    assert efeito["melhor_valor"] in ("0.0005", "0.002")


def test_varredura_com_variante_que_falha_nao_derruba_o_resto(tmp_path, base_micro):
    """Warmup maior que os passos: uma variante morre, as outras têm de sobreviver."""
    relatorio = bv.executar(
        bv.ConfigVarredura(
            base=str(base_micro),
            grades={"warmup": [1, 400]},
            prefixo="sw-falha",
            raiz=str(tmp_path),
            passos=10,
        )
    )
    situacoes = {v["sobrecritas"]["warmup"]: v for v in relatorio["variantes"]}
    assert situacoes[1]["melhor_val"] is not None
    assert relatorio["ranking"] == ["sw-falha-01"]
    assert relatorio["falhas"] == ["sw-falha-02"]


def test_relatorio_json_fica_no_disco(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, passos=8)
    arquivo = tmp_path / relatorio["onde_salvou"]
    assert arquivo.exists()
    gravado = json.loads(arquivo.read_text(encoding="utf-8"))
    assert gravado["prefixo"] == "sw"
    assert len(gravado["variantes"]) == 2


def test_base_inexistente_falha(tmp_path):
    with pytest.raises(FileNotFoundError, match="config base"):
        bv.executar(bv.ConfigVarredura(base="nao-existe.yaml", grades={"lr": [1, 2]}, prefixo="x", raiz=str(tmp_path)))


# --- apresentação -----------------------------------------------------------

def test_tabela_e_relatorio_markdown(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, passos=8)
    tabela = bv.tabela_texto(relatorio)
    assert "varredura : sw" in tabela
    assert "sw-01" in tabela and "sw-02" in tabela
    assert "efeito de cada chave" in tabela
    assert "melhor:" in tabela

    markdown = bv.relatorio_markdown(relatorio)
    assert markdown.startswith("# Varredura sw")
    assert "| run |" in markdown
    assert "Efeito de cada chave" in markdown
    destino = bv.salvar_relatorio(relatorio, tmp_path / "rel" / "sw.md")
    assert destino.read_text(encoding="utf-8") == markdown


def test_tabela_marca_falha(tmp_path, base_micro):
    relatorio = _varredura(tmp_path, base_micro, seco=True)
    relatorio["variantes"][0]["erro"] = "warmup (99) precisa ser menor que passos (10)"
    tabela = bv.tabela_texto(relatorio)
    assert "erro: warmup" in tabela
