"""Testes da trilha de estudo — as lições precisam ser verdadeiras, não só bonitas.

O que se cobra aqui não é o texto: é que (a) as citações ao código do núcleo continuem
existentes e (b) cada experimento comprove a afirmação que a lição faz.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from labia import trilha
from common import gerar_corpus

RAIZ = Path(__file__).resolve().parents[2]
EXPERIMENTOS = RAIZ / "trilha" / "experimentos"


def carregar(nome: str):
    caminho = EXPERIMENTOS / nome
    spec = importlib.util.spec_from_file_location(nome.removesuffix(".py"), caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# --- estrutura --------------------------------------------------------------

def test_indice_tem_as_cinco_licoes_com_experimento():
    indice = trilha.indice(RAIZ)
    assert indice["total"] == 10
    numeros = [l["numero"] for l in indice["licoes"]]
    assert numeros == list(range(1, 11))
    assert all(l["tem_experimento"] for l in indice["licoes"])
    assert all(l["resumo"] for l in indice["licoes"])


def test_indice_liga_licao_ao_experimento_certo():
    por_numero = {l.numero: l for l in trilha.licoes(RAIZ)}
    assert por_numero[3].experimento.name == "e03_atencao.py"
    assert "atenção" in por_numero[3].titulo.lower()


def test_licao_inexistente_explica_o_que_existe():
    with pytest.raises(ValueError, match="disponíveis"):
        trilha.achar_licao(99, RAIZ)


def test_ler_licao_devolve_markdown_da_licao():
    texto = trilha.ler_licao(1, RAIZ)
    assert texto.startswith("# Lição 1")
    assert "No núcleo" in texto
    assert "Exercícios" in texto


# --- citações ao código não podem apodrecer --------------------------------

def test_todas_as_referencias_ao_codigo_estao_vivas():
    problemas = trilha.conferir_referencias(RAIZ)
    assert problemas == [], f"lições citam código que não existe mais: {problemas}"


def test_conferidor_pega_simbolo_renomeado(tmp_path):
    """O teste do teste: se alguém renomear a função, a conferência tem de acusar."""
    gpt = RAIZ / "core" / "labia" / "models" / "gpt.py"
    assert trilha._simbolo_existe(gpt, "AtencaoMultiCabeca.forward")
    assert trilha._simbolo_existe(gpt, "GPT")
    assert not trilha._simbolo_existe(gpt, "AtencaoMultiCabeca.nao_existe_mais")
    assert not trilha._simbolo_existe(gpt, "ClasseInventada.metodo")


def test_conferidor_acusa_arquivo_ausente_no_texto_da_licao(tmp_path):
    pasta = tmp_path / "trilha" / "licoes"
    pasta.mkdir(parents=True)
    (pasta / "01-falsa.md").write_text(
        "# Lição 1 — falsa\n\n**No núcleo:** " + chr(96) + "core/labia/nao_existe.py" + chr(96) + " → " + chr(96) + "nada" + chr(96) + "\n",
        encoding="utf-8",
    )
    problemas = trilha.conferir_referencias(tmp_path)
    assert problemas and problemas[0]["motivo"] == "arquivo não existe"


# --- os experimentos comprovam o que as lições afirmam ----------------------

def test_experimento_1_autograd_e_exato_e_acumula():
    modulo = carregar("e01_autograd.py")
    resultado = modulo.main()
    assert resultado["analitico"]["erro_maximo"] == 0.0
    # sem zerar, o gradiente soma: 3, 6, 9 (a causa nº 1 de treino que não aprende)
    assert resultado["acumulacao"]["grad_apos_cada_backward"] == [3.0, 6.0, 9.0]
    assert resultado["acumulacao"]["apos_zero_grad"] == 3.0
    assert resultado["numerico"]["erro_relativo"] < 1e-6


def test_experimento_2_vocabulario_maior_encurta_a_sequencia(tmp_path):
    modulo = carregar("e02_tokenizacao.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(gerar_corpus(60, semente=4), encoding="utf-8")
    resultado = modulo.main(corpus=str(corpus), tamanhos=(128, 512))
    tabela = resultado["tabela"]
    assert [l["vocab_pedido"] for l in tabela] == [128, 512]
    # o alfabeto byte-level força os 256 bytes: nunca dá para pedir menos que isso
    assert tabela[0]["vocab_real"] >= 256
    assert tabela[1]["chars_por_token"] > tabela[0]["chars_por_token"]
    assert tabela[1]["tokens_da_frase"] < tabela[0]["tokens_da_frase"]
    assert sum(len(p) for p in tabela[1]["pedacos_da_frase"]) > 0


def test_experimento_3_causalidade_e_vazamento_sem_mascara():
    modulo = carregar("e03_atencao.py")
    resultado = modulo.main()
    causal = resultado["causalidade"]
    assert causal["causal"] is True
    assert causal["diferenca_nas_posicoes_anteriores"] == 0.0
    assert causal["diferenca_na_ultima_posicao"] > 0
    vazamento = resultado["vazamento"]
    assert vazamento["posicao_0_muda_com_causal"] == 0.0
    assert vazamento["posicao_0_muda_sem_causal"] > 0.1


def test_experimento_4_escala_da_inicializacao_segura_o_residuo():
    modulo = carregar("e04_bloco.py")
    resultado = modulo.main()
    assert resultado["crescimento_sem_escala"] > resultado["crescimento_com_escala"] > 1.0
    assert resultado["perfil_sem_escala"][-1] > resultado["perfil_com_escala"][-1]
    assert len(resultado["perfil_com_escala"]) == 7  # entrada + 6 blocos


def test_experimento_5_schedule_e_lr_exagerado():
    modulo = carregar("e05_treino.py")
    resultado = modulo.main(passos=60)
    curva = {p["passo"]: p["lr"] for p in resultado["curva_lr"]}
    assert curva[0] < curva[20]                       # warmup sobe
    assert curva[20] > curva[60] > curva[119]         # cosseno desce
    bom, alto = resultado["treinos"]
    assert alto["lr"] > bom["lr"]
    assert alto["maior_perda"] > bom["maior_perda"]   # lr exagerado estoura no meio
    assert bom["perda_final"] < bom["perda_inicial"]  # com lr sadio, a perda cai


def test_experimento_roda_como_script_de_verdade():
    """A lição manda rodar por linha de comando; isso precisa funcionar de fato."""
    processo = trilha.rodar_experimento(1, RAIZ)
    assert processo.returncode == 0, processo.stderr
    assert "autograd x derivada analítica" in processo.stdout

# --- notebooks: gerados do experimento, nunca escritos à mão -----------------

def _importar_gerador():
    caminho = RAIZ / "scripts" / "gera_notebooks.py"
    spec = importlib.util.spec_from_file_location("gera_notebooks", caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_gera_um_notebook_por_experimento():
    gerador = _importar_gerador()
    notebooks = gerador.gerar_todos(RAIZ)
    assert len(notebooks) == 10
    nomes = sorted(p.name for p in notebooks)
    assert nomes[0] == "e01_autograd.ipynb" and nomes[-1] == "e10_agentes.ipynb"


def test_notebook_tem_estrutura_valida_do_nbformat():
    gerador = _importar_gerador()
    for destino, notebook in gerador.gerar_todos(RAIZ).items():
        assert notebook["nbformat"] == 4
        assert notebook["cells"], destino
        assert notebook["cells"][0]["cell_type"] == "markdown"
        for celula in notebook["cells"]:
            assert celula["cell_type"] in ("markdown", "code")
            assert isinstance(celula["source"], list)
            assert "metadata" in celula
            if celula["cell_type"] == "code":
                assert celula["outputs"] == []
        assert notebook["metadata"]["labia"]["experimento"] in destino.name.replace(".ipynb", ".py")


def test_notebook_leva_cada_funcao_do_experimento_em_uma_celula():
    gerador = _importar_gerador()
    import ast as _ast

    for destino, notebook in gerador.gerar_todos(RAIZ).items():
        fonte = (EXPERIMENTOS / notebook["metadata"]["labia"]["experimento"]).read_text(encoding="utf-8")
        funcoes = [n.name for n in _ast.parse(fonte).body if isinstance(n, _ast.FunctionDef)]
        codigo = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        for nome in funcoes:
            assert f"def {nome}(" in codigo, f"{destino.name} não tem a função {nome}"
        assert codigo.rstrip().endswith("resultado = main()")
        assert "from pathlib import Path" in codigo  # célula de preparo


def test_notebooks_versionados_estao_em_dia_com_os_experimentos():
    """Anti-apodrecimento: mudar o experimento sem regerar o notebook quebra aqui."""
    gerador = _importar_gerador()
    for destino, notebook in gerador.gerar_todos(RAIZ).items():
        assert destino.exists(), f"notebook ausente: {destino.name} (rode scripts/gera_notebooks.py)"
        assert destino.read_text(encoding="utf-8") == gerador._serializar(notebook), (
            f"{destino.name} desatualizado — rode scripts/gera_notebooks.py"
        )
        json.loads(destino.read_text(encoding="utf-8"))  # JSON válido de verdade

# --- lições 6 a 10: a metade "otimizar modelos" -----------------------------

def test_experimento_6_lora_e_no_op_posto_baixo_e_mesclavel():
    modulo = carregar("e06_lora.py")
    resultado = modulo.main()
    assert resultado["no_op"]["identico"] is True
    assert resultado["no_op"]["diferenca_maxima"] == 0.0
    proporcao = resultado["proporcao"]
    assert 0 < proporcao["treinaveis"] < proporcao["total"]
    assert proporcao["modulos_alterados"] > 0
    assert proporcao["antes_treinaveis"] == proporcao["antes_total"]  # antes do LoRA, tudo treinava
    assert proporcao["total"] == proporcao["antes_total"] + proporcao["adaptador_adicionado"]
    posto = resultado["posto"]
    assert posto["posto_da_atualizacao"] == posto["r"]          # a hipótese do método
    assert posto["parametros_do_ramo"] < 4 * posto["parametros_da_base"]
    mescla = resultado["mesclagem"]
    assert mescla["diferenca_maxima"] < 1e-5
    assert mescla["ainda_tem_nolinear"] is False
    assert mescla["tipos_de_linear"] == ["Linear"]


def test_experimento_7_quantizacao_troca_tamanho_por_erro():
    modulo = carregar("e07_quantizacao.py")
    resultado = modulo.main()
    modos = {m["modo"]: m for m in resultado["modos"]}
    int8, nf4 = modos["int8"], modos["nf4"]
    # int8: ~4x menos bytes, ~8 bits por parâmetro
    assert 3.5 < int8["fator"] < 4.5
    assert 7.5 < int8["bits_por_parametro"] < 8.5
    # nf4: encolhe mais e guarda menos bits por parâmetro (4 + 16/64 de escala)
    assert nf4["fator"] > int8["fator"]
    assert 4.0 < nf4["bits_por_parametro"] < 4.6
    # e paga em erro: o preço que a propaganda costuma omitir
    assert nf4["erro_relativo"] > int8["erro_relativo"]
    assert resultado["nf4"]["niveis"] == 16
    assert resultado["nf4"]["espaco_na_borda"] > resultado["nf4"]["espaco_no_centro"]
    faixa = resultado["erro_por_faixa"]
    assert faixa["erro_medio_pesos_grandes"] > faixa["erro_medio_pesos_pequenos"]


def test_experimento_8_moe_esparsidade_e_balanceamento():
    modulo = carregar("e08_moe.py")
    resultado = modulo.main(passos_treino=60)
    top1, top2 = resultado["parametros"]["top1"], resultado["parametros"]["top2"]
    assert top1["ativos"] < top1["total"]
    assert top2["ativos"] > top1["ativos"]           # mais especialistas ativos por token
    assert top1["total"] == top2["total"]            # mesmos parâmetros guardados
    aux = resultado["auxiliar"]
    assert aux["uniforme"] == 1.0                    # piso
    assert aux["colapsado"] == 4.0                   # teto = n
    assert aux["uniforme"] < aux["meio"] < aux["colapsado"]
    # o mecanismo: o gradiente abaixa o logit do dominante e levanta o do subutilizado
    gradiente = resultado["gradiente"]
    projecoes = gradiente["projecao_do_gradiente"]
    assert any(p > 0 for p in projecoes), "nenhum especialista seria penalizado"
    assert any(p < 0 for p in projecoes), "nenhum especialista seria favorecido"


def test_experimento_8_auxiliar_desfaz_o_colapso():
    """A prova de que o coeficiente auxiliar faz algo — o bug que esta lição achou."""
    modulo = carregar("e08_moe.py")
    resultado = modulo.main(passos_treino=60)
    sem = resultado["sem_auxiliar"]
    com = resultado["com_auxiliar"]
    assert sem["concentracao_inicial"] == com["concentracao_inicial"]  # partem do mesmo colapso
    assert sem["concentracao_inicial"] > 0.9
    assert com["concentracao_final"] < sem["concentracao_final"]
    assert com["concentracao_final"] < 0.5, "com auxiliar o roteador deveria voltar ao equilíbrio"


def test_experimento_9_raciocinio_custo_e_formato():
    modulo = carregar("e09_raciocinio.py")
    resultado = modulo.main()
    formato = resultado["formato"]
    assert formato["formato correto"] == 12
    assert formato["com passo a passo"] == 12
    assert formato["texto livre"] is None            # resposta certa, formato errado = erro
    assert formato["sem a palavra-chave"] is None
    custo = resultado["custo"]
    assert custo["tokens_prompt_cot"] > custo["tokens_prompt_direta"]
    assert custo["tokens_gerados_cot"] > custo["tokens_gerados_direta"]
    assert resultado["benchmark"]["itens_teste"] == 152
    g5 = resultado["resultado_g5"]
    if g5:  # depende do run real existir neste repositório
        assert g5["cot"] > g5["direta"]
        assert g5["cot_menos_direta_pp"] > 0


def test_experimento_10_agentes_sao_auditaveis():
    modulo = carregar("e10_agentes.py")
    resultado = modulo.main()
    inv = resultado["inventario"]
    assert inv["agentes"] == ["arquiteto", "avaliador", "treinador"]
    assert inv["total_skills"] == 8
    for dados in inv["por_agente"].values():
        assert dados["papel"] and dados["skills"] and dados["assinaturas"]
        assert all("(" in a for a in dados["assinaturas"])   # assinatura declarada, não só nome
    sucesso = resultado["sucesso"]
    assert sucesso["eventos"] and sucesso["eventos"][0]["ok"] is True
    assert sucesso["resultado"]["parametros_estimados"] > 0
    falha = resultado["falha"]
    assert falha["levantou_erro"] is True
    assert falha["eventos"] and falha["eventos"][0]["ok"] is False
    assert falha["eventos"][0]["erro"] == "skill desconhecida"
