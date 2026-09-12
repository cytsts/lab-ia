"""Testes da bancada B2 — curvas (specs/B2.md).

Os testes que exigem matplotlib são pulados se a lib não estiver instalada; a
coleta de séries e o payload da UI não dependem dela.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from labia.bancada import curva as bcv


def escrever_run(tmp_path, run_id, registros):
    pasta = tmp_path / "runs" / run_id
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "metricas.jsonl").write_text(
        "\n".join(json.dumps(r) for r in registros) + "\n", encoding="utf-8"
    )
    return pasta


def registros(passos, vals, trem=None, lr=3e-4, tokens=250_000.0):
    return [
        {
            "passo": p,
            "loss_val": v,
            "loss_trem": (trem[i] if trem else (None if v is None else v - 0.1)),
            "lr": lr,
            "tokens_por_s": tokens,
        }
        for i, (p, v) in enumerate(zip(passos, vals))
    ]


def test_coletar_series_ignora_metricas_ausentes(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100, 200], [8.0, 5.0, None]))
    series = bcv.coletar_series(["a"], tmp_path, "loss_val")
    assert len(series) == 1
    assert series[0]["pontos"] == [(0, 8.0), (100, 5.0)]
    assert series[0]["run_id"] == "a"


def test_coletar_series_pula_run_sem_metrica(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100], [8.0, 5.0]))
    (tmp_path / "runs" / "b").mkdir(parents=True)
    assert len(bcv.coletar_series(["a", "b"], tmp_path, "loss_val")) == 1
    assert bcv.coletar_series(["fantasma"], tmp_path, "loss_val") == []


def test_cores_diferentes_por_run(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100], [8.0, 5.0]))
    escrever_run(tmp_path, "b", registros([0, 100], [8.0, 6.0]))
    series = bcv.coletar_series(["a", "b"], tmp_path, "loss_val")
    assert series[0]["cor"] != series[1]["cor"]


def test_dados_para_ui_tem_paineis_rotulados(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100], [8.0, 5.0]))
    dados = bcv.dados_para_ui(["a"], tmp_path)
    assert dados["runs"] == ["a"]
    assert [p["metrica"] for p in dados["paineis"]] == list(bcv.PAINEIS_PADRAO)
    assert all(p["rotulo"] for p in dados["paineis"])
    assert dados["paineis"][0]["series"][0]["pontos"]


def test_dados_para_ui_ignora_metrica_desconhecida(tmp_path):
    escrever_run(tmp_path, "a", registros([0], [8.0]))
    dados = bcv.dados_para_ui(["a"], tmp_path, metricas=("loss_val", "inexistente"))
    assert [p["metrica"] for p in dados["paineis"]] == ["loss_val"]


def test_desenhar_sem_dados_falha(tmp_path):
    with pytest.raises(ValueError, match="nenhuma métrica"):
        bcv.desenhar(["fantasma"], raiz=tmp_path, saida=tmp_path / "x.svg")


def test_desenhar_metricas_invalidas_falha(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100], [8.0, 5.0]))
    with pytest.raises(ValueError, match="nenhum painel válido"):
        bcv.desenhar(["a"], raiz=tmp_path, saida=tmp_path / "x.svg", paineis=("nada",))


def test_svg_e_gerado_sem_dependencia_nenhuma(tmp_path):
    """SVG é o caminho padrão: o laboratório precisa funcionar em máquina sem rede."""
    escrever_run(tmp_path, "a", registros([0, 100, 200], [8.0, 5.5, 5.0]))
    escrever_run(tmp_path, "b", registros([0, 100, 200], [8.0, 6.0, 5.6]))
    destino = tmp_path / "figuras" / "curvas.svg"
    saida = bcv.desenhar(["a", "b"], raiz=tmp_path, saida=destino, paineis=("loss_val", "lr"))
    texto = saida.read_text(encoding="utf-8")
    raiz_xml = ET.fromstring(texto)  # XML válido, não string montada na mão
    assert raiz_xml.tag.endswith("svg")
    assert "a" in texto and "b" in texto
    assert bcv.ROTULOS["loss_val"] in texto and bcv.ROTULOS["lr"] in texto


def test_svg_tem_um_traco_por_run_com_todos_os_pontos(tmp_path):
    escrever_run(tmp_path, "a", registros([0, 100, 200, 300], [8.0, 6.0, 5.2, 5.0]))
    escrever_run(tmp_path, "b", registros([0, 100, 200, 300], [8.0, 6.5, 6.0, 5.9]))
    destino = bcv.desenhar(["a", "b"], raiz=tmp_path, saida=tmp_path / "c.svg", paineis=("loss_val",))
    raiz_xml = ET.fromstring(destino.read_text(encoding="utf-8"))
    tracos = raiz_xml.findall(".//{http://www.w3.org/2000/svg}polyline")
    assert len(tracos) == 2
    for traco in tracos:
        coordenadas = traco.get("points").split()
        assert len(coordenadas) == 4
        for par in coordenadas:
            x, y = (float(v) for v in par.split(","))
            assert 0 < x < bcv.LARGURA
            assert 0 < y < bcv.ALTURA_PAINEL * 4


def test_svg_escapa_caracteres_perigosos(tmp_path):
    """Título com & e < precisa sair escapado, senão o SVG deixa de ser XML válido."""
    escrever_run(tmp_path, "a", registros([0, 100], [8.0, 5.0]))
    destino = bcv.desenhar(
        ["a"], raiz=tmp_path, saida=tmp_path / "d.svg", paineis=("loss_val",), titulo="perda & ganho <br>"
    )
    texto = destino.read_text(encoding="utf-8")
    assert "perda &amp; ganho &lt;br&gt;" in texto
    ET.fromstring(texto)
    assert bcv._escapar('a"b') == "a&quot;b"


def test_png_usa_matplotlib_quando_disponivel(tmp_path):
    pytest.importorskip("matplotlib")
    escrever_run(tmp_path, "a", registros([0, 100, 200], [8.0, 5.5, 5.0]))
    destino = bcv.desenhar(["a"], raiz=tmp_path, saida=tmp_path / "curvas.png", paineis=("loss_val",))
    assert destino.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
