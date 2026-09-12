"""Testes da bancada B2 — comparação e diagnóstico de runs (specs/B2.md)."""
from __future__ import annotations

import json
import math

import pytest

from labia.bancada import comparar as bc


def escrever_run(tmp_path, run_id, registros, **estado):
    pasta = tmp_path / "runs" / run_id
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "metricas.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8"
    )
    (pasta / "estado.json").write_text(
        json.dumps({"run_id": run_id, **estado}, ensure_ascii=False), encoding="utf-8"
    )
    return pasta


def curva(passos, valores, loss_trem=None, unigram=None, inicio=0):
    registros = []
    for indice, (passo, val) in enumerate(zip(passos, valores)):
        registros.append(
            {
                "passo": passo,
                "loss_val": val,
                "loss_trem": (loss_trem[indice] if loss_trem else (val - 0.05 if val is not None else None)),
                "lr": 3e-4,
                "tokens_por_s": 300_000.0,
                "tempo_s": float(passo),
                **({"entropia_unigram": unigram} if indice == 0 and unigram is not None else {}),
            }
        )
    return registros


# --- diagnóstico ------------------------------------------------------------

def test_diagnostica_overfit_quando_val_piora(tmp_path):
    escrever_run(
        tmp_path, "overfit", curva([0, 500, 1000, 1500], [8.0, 5.0, 4.7, 5.1]),
        passo=1500, passos_totais=1500, concluido=True, tempo_decorrido_s=10.0,
    )
    resumo = bc.resumir_run("overfit", tmp_path)
    assert resumo.diagnostico == "overfit"
    assert resumo.melhor_val == 4.7
    assert resumo.passo_melhor_val == 1000
    assert resumo.drift == pytest.approx(0.4)
    assert "decorar" in resumo.comentario


def test_diagnostica_estavel_quando_deriva_pequena(tmp_path):
    escrever_run(
        tmp_path, "estavel", curva([0, 500, 1000], [8.0, 4.7, 4.75]),
        passo=1000, passos_totais=1000, concluido=True,
    )
    resumo = bc.resumir_run("estavel", tmp_path)
    assert resumo.diagnostico == "estavel"
    assert "Dentro da barra" in resumo.comentario


def test_diagnostica_ainda_caindo_quando_minimo_e_a_ultima(tmp_path):
    escrever_run(tmp_path, "caindo", curva([0, 500, 1000], [8.0, 5.2, 5.0]), passo=1000, passos_totais=1000)
    resumo = bc.resumir_run("caindo", tmp_path)
    assert resumo.diagnostico == "ainda_caindo"
    assert "não convergiu" in resumo.comentario


def test_diagnostica_deteriorando_quando_ultimas_avaliacoes_sobem(tmp_path):
    escrever_run(
        tmp_path, "piorando", curva([0, 500, 1000, 1500, 2000], [8.0, 4.7, 4.72, 4.76, 4.80]),
        passo=2000, passos_totais=2000,
    )
    resumo = bc.resumir_run("piorando", tmp_path)
    assert resumo.diagnostico == "deteriorando"
    assert "já passou do ponto" in resumo.comentario


def test_diagnostica_estagnado_quando_ganho_desprezivel(tmp_path):
    escrever_run(
        tmp_path,
        "plato",
        curva([0, 250, 500, 750, 1000, 1250, 1500, 1750], [8.0, 6.0, 5.0, 4.80, 4.70, 4.695, 4.694, 4.6935]),
        passo=1750, passos_totais=1750,
    )
    resumo = bc.resumir_run("plato", tmp_path)
    assert resumo.diagnostico == "estagnado"
    assert "platô" in resumo.comentario


def test_queda_inicial_do_modelo_aleatorio_nao_e_instabilidade(tmp_path):
    """Regressão: passo 0 é o modelo aleatório; cair 2 nats dele para o passo 100 é normal."""
    escrever_run(tmp_path, "queda", curva([0, 100, 200, 300], [8.33, 6.32, 5.90, 5.70]), passo=300, passos_totais=300)
    resumo = bc.resumir_run("queda", tmp_path)
    assert resumo.instavel is False
    assert resumo.diagnostico != "instavel"


def test_salto_entre_avaliacoes_treinadas_e_instabilidade(tmp_path):
    escrever_run(
        tmp_path, "tremida", curva([0, 100, 200, 300], [8.33, 5.00, 7.40, 5.10]),
        passo=300, passos_totais=300,
    )
    resumo = bc.resumir_run("tremida", tmp_path)
    assert resumo.instavel is True
    assert resumo.diagnostico == "instavel"


def test_run_divergido(tmp_path):
    escrever_run(tmp_path, "nan", curva([0, 100, 200], [8.0, float("nan"), float("inf")]), passo=200, passos_totais=200)
    resumo = bc.resumir_run("nan", tmp_path)
    assert resumo.diagnostico == "divergiu"
    assert "divergiu" in resumo.comentario


def test_run_ausente_e_run_sem_metricas(tmp_path):
    assert bc.resumir_run("fantasma", tmp_path).diagnostico == "ausente"
    (tmp_path / "runs" / "vazio").mkdir(parents=True)
    (tmp_path / "runs" / "vazio" / "estado.json").write_text("{}", encoding="utf-8")
    assert bc.resumir_run("vazio", tmp_path).diagnostico == "sem_metricas"


def test_run_de_quantizacao_nao_entra_no_ranking(tmp_path):
    pasta = escrever_run(
        tmp_path, "quant", curva([0], [8.729]), tipo="quantizacao", modo="nf4", fonte="base", concluido=True
    )
    (pasta / "tamanhos.json").write_text(
        json.dumps({"fator_alvos": 7.11, "bits_efetivos_por_parametro": 4.5,
                    "perda_val_antes": 8.729, "perda_val_depois": 8.881}),
        encoding="utf-8",
    )
    escrever_run(tmp_path, "treino", curva([0, 500, 1000], [8.0, 4.7, 4.75]), passo=1000, passos_totais=1000)
    resultado = bc.comparar(["quant", "treino"], tmp_path)
    assert resultado["ranking"] == ["treino"]
    quant = next(r for r in resultado["runs"] if r["run_id"] == "quant")
    assert quant["familia"] == "quantizacao"
    assert quant["diagnostico"] == "quantizacao"
    assert "7.11x" in quant["comentario"]


# --- comparação -------------------------------------------------------------

def test_orcamento_comum_compara_no_mesmo_tamanho(tmp_path):
    escrever_run(tmp_path, "curto", curva([0, 500, 1000], [8.0, 5.0, 4.9]), passo=1000, passos_totais=1000)
    escrever_run(tmp_path, "longo", curva([0, 500, 1000, 2000], [8.0, 5.2, 5.1, 4.0]), passo=2000, passos_totais=2000)
    resultado = bc.comparar(["curto", "longo"], tmp_path)
    assert resultado["orcamento_comum"] == 1000
    longo = next(r for r in resultado["runs"] if r["run_id"] == "longo")
    assert longo["melhor_val"] == 4.0
    assert longo["melhor_val_orcamento_comum"] == 5.1  # no mesmo orçamento, o curto lidera
    assert "orçamento comum" in resultado["veredito"]


def test_quantizacao_mostra_perdas_mas_fica_fora_do_ranking(tmp_path):
    """As perdas de um run quantizado são úteis na tabela, mesmo sem entrar no ranking."""
    pasta = escrever_run(
        tmp_path, "quant", curva([0], [8.729]), tipo="quantizacao", modo="int8", fonte="base", concluido=True
    )
    with open(pasta / "metricas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"passo": 1, "loss_val": 8.7322, "loss_trem": None}) + "\n")
    (pasta / "tamanhos.json").write_text(json.dumps({"fator_alvos": 3.95}), encoding="utf-8")
    resumo = bc.resumir_run("quant", tmp_path)
    assert resumo.melhor_val == 8.729
    assert resumo.val_final == 8.7322
    assert resumo.diagnostico == "quantizacao"
    assert bc.comparar(["quant"], tmp_path)["ranking"] == []


def test_run_sem_passos_nao_zera_o_orcamento(tmp_path):
    """Regressão: run de quantização (sem 'passo') fazia o orçamento comum virar 0."""
    pasta = escrever_run(tmp_path, "quant", curva([0], [8.7]), tipo="quantizacao", modo="int8", concluido=True)
    (pasta / "tamanhos.json").write_text(json.dumps({"fator_alvos": 3.95}), encoding="utf-8")
    escrever_run(tmp_path, "treino", curva([0, 500, 1000], [8.0, 4.9, 4.8]), passo=1000, passos_totais=1000)
    resultado = bc.comparar(["quant", "treino"], tmp_path)
    assert resultado["orcamento_comum"] == 1000
    treino = next(r for r in resultado["runs"] if r["run_id"] == "treino")
    assert treino["melhor_val_orcamento_comum"] == 4.8


def test_veredito_avisa_familias_diferentes(tmp_path):
    escrever_run(tmp_path, "do-zero", curva([0, 500], [8.0, 4.7]), passo=500, passos_totais=500)
    escrever_run(tmp_path, "ajuste", curva([0, 500], [8.0, 0.4]), tipo="lora", r=8, passo=500, passos_totais=500)
    resultado = bc.comparar(["do-zero", "ajuste"], tmp_path)
    assert "famílias diferentes" in resultado["veredito"]
    assert set(resultado["familias"]) == {"treino", "lora"}


def test_parametros_vem_da_config_gravada(tmp_path):
    escrever_run(
        tmp_path, "descritivo", curva([0, 100], [8.0, 5.0]), passo=100, passos_totais=100,
        config_modelo={"vocab": 4096, "dim": 256, "camadas": 6, "cabecas": 8, "janela_ctx": 256,
                       "abandono": 0.1, "n_especialistas": 0, "top_k": 1},
        config_treino={"lote": 32, "passos": 100},
    )
    resumo = bc.resumir_run("descritivo", tmp_path)
    assert resumo.parametros == 5_847_040
    assert resumo.tokens_vistos == 32 * 256 * 100


def test_ganho_sobre_entropia_unigram(tmp_path):
    escrever_run(tmp_path, "unigram", curva([0, 500], [8.0, 4.5], unigram=6.529), passo=500, passos_totais=500)
    resumo = bc.resumir_run("unigram", tmp_path)
    assert resumo.entropia_unigram == pytest.approx(6.529)
    assert resumo.ganho_vs_unigram == pytest.approx(2.029)


def test_comparar_sem_runs(tmp_path):
    resultado = bc.comparar([], tmp_path)
    assert resultado["ranking"] == []
    assert "nada a comparar" in resultado["veredito"]


def test_listar_runs_ignora_pastas_sem_estado(tmp_path):
    escrever_run(tmp_path, "b", curva([0], [8.0]), passo=1, passos_totais=1)
    escrever_run(tmp_path, "a", curva([0], [8.0]), passo=1, passos_totais=1)
    (tmp_path / "runs" / "lixo").mkdir(parents=True)
    assert bc.listar_runs(tmp_path) == ["a", "b"]


# --- saída ------------------------------------------------------------------

def test_tabela_e_relatorio_cobrem_todos_os_runs(tmp_path):
    escrever_run(tmp_path, "um", curva([0, 500, 1000], [8.0, 4.7, 4.75]), passo=1000, passos_totais=1000)
    escrever_run(tmp_path, "dois", curva([0, 500, 1000], [8.0, 5.0, 5.1]), passo=1000, passos_totais=1000)
    resultado = bc.comparar(["um", "dois"], tmp_path)
    tabela = bc.tabela_texto(resultado)
    assert "um" in tabela and "dois" in tabela
    assert "veredito:" in tabela
    markdown = bc.relatorio_markdown(resultado)
    assert markdown.startswith("# Comparação de runs")
    assert "| run |" in markdown
    destino = bc.salvar_relatorio(resultado, tmp_path / "saida" / "rel.md")
    assert destino.read_text(encoding="utf-8") == markdown


def test_json_serializavel_nao_quebra_com_dados_estranhos(tmp_path):
    escrever_run(tmp_path, "x", curva([0], [float("nan")]), passo=1, passos_totais=1)
    texto = bc.json_serializavel(bc.comparar(["x"], tmp_path))
    assert json.loads(texto)["runs"][0]["run_id"] == "x"
