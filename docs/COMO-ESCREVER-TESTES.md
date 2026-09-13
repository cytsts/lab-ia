# Como escrever testes nesta suíte

Guia prático, escrito a partir do que este repositório já faz. A regra que vale acima
de todas: **teste é a prova de uma afirmação, não uma formalidade de cobertura.**

Se você não consegue dizer em uma frase o que o teste prova, ele ainda não está pronto.

---

## 1. Onde o teste mora

```
tests/
  common.py          helpers compartilhados (gerar_corpus, config_micro, criar_dir_run)
  conftest.py        fixtures de sessão (corpus_arquivo)
  g1/ ... g10/       testes das metas originais
  b1/ ... b12/       testes da bancada, da trilha e das frentes novas
  trilha/            verificação do material didático
  logica/            verificação do gerador de lógica
```

Uma pasta por meta. O nome do arquivo diz o assunto: `test_preparo.py`, `test_comparar.py`,
`test_api_bancada.py`. Testes que exigem treino real levam a marca `@pytest.mark.slow`.

## 2. O ciclo (a convenção do projeto)

```
SPEC  -->  TESTE (falhando)  -->  CÓDIGO  -->  TESTE verde  -->  EVIDÊNCIA registrada
```

1. Escreva a spec (ou pelo menos os critérios de aceite em Given-When-Then) com o
   **número esperado** de cada um.
2. Escreva o teste que cobra aquele número. Rode e veja falhar — se ele passar de
   primeira, ou o teste não testa nada, ou o código já existia.
3. Implemente até ficar verde.
4. Registre a evidência (comando + saída) em `specs/PLANO.md`.

## 3. Anatomia de um teste aqui

```python
def test_experimento_1_autograd_e_exato_e_acumula():
    modulo = carregar("e01_autograd.py")
    resultado = modulo.main()
    assert resultado["analitico"]["erro_maximo"] == 0.0
    # sem zerar, o gradiente soma: 3, 6, 9 (a causa nº 1 de treino que não aprende)
    assert resultado["acumulacao"]["grad_apos_cada_backward"] == [3.0, 6.0, 9.0]
```

Três coisas para notar:

- **o número está no assert.** `== 0.0`, `== [3.0, 6.0, 9.0]`. Nada de `assert resultado`.
- **o comentário explica o porquê**, não o quê. Quem quebrar o teste precisa entender o
  que ele protegia.
- **sem rede, sem GPU, sem espera.** Roda em segundos e não depende do ambiente.

## 4. Padrões que valem a pena copiar

| padrão | exemplo no repo | por quê |
|---|---|---|
| Testar a **invariante**, não a implementação | `test_o_cot_nunca_discorda_da_resposta` | sobrevive a refatoração |
| Números independentes | `test_tabela_verdade_de_cada_operador` escreve a tabela à mão | a referência não é o próprio código |
| **Dublês injetados** | `test_gpt_roda_e_aprende...`, núcleo portátil | testa o caminho sem subprocesso nem espera |
| Determinismo com semente | `test_determinismo_e_sensibilidade_a_semente` | mesma entrada, mesma saída |
| **Teste do teste** | `test_conferidor_pega_simbolo_renomeado` | o verificador também pode estar errado |
| Anti-apodrecimento | `test_notebooks_versionados_estao_em_dia` | material que envelhece em silêncio |
| Anti-flaky por poll | `_esperar_checkpoint` na G8 | `sleep` fixo é corrida esperando acontecer |
| Falha isolada | `test_varredura_com_variante_que_falha...` | uma variante ruim não derruba o lote |

## 5. Rodando

No seu terminal:

```bat
.venv\Scripts\python -m pytest -q                       :: suíte completa
.venv\Scripts\python -m pytest -q -m "not slow"         :: ciclo rápido (sem treino real)
.venv\Scripts\python -m pytest tests/logica -q          :: uma pasta
.venv\Scripts\python -m pytest tests/b7 -q -k rope      :: por nome
.venv\Scripts\python -m pytest --cov=core/labia -q      :: com cobertura
pnpm --dir ui test                                      :: testes da interface
```

**Nota para agentes:** dentro do sandbox do harness, o pytest cria o diretório de
temporários com permissão que o próprio sandbox nega (WinError 5). O contorno fica em
`.lab-ia/` (não versionado):

```bat
$env:PYTHONPATH='.lab-ia'; .venv\Scripts\python -m pytest -q -m "not slow" ^
  -p no:cacheprovider -p contorno_sandbox --basetemp=.lab-ia/ptmp1
```

Isso é limitação do ambiente do agente, não do laboratório.

## 6. Erros que já custaram tempo aqui (evite)

| erro | o que aconteceu | conserto |
|---|---|---|
| Tolerância apertada demais em float32 | RoPE "falhava" com 5e-8 de diferença | `abs=1e-6` com o motivo escrito no teste |
| Teste que mede outra coisa | gerei vetor diferente por posição e cobrei a propriedade de posição relativa | a mesma direção em todas as posições |
| Asserção sobre conteúdo gerado | `test_gerar_no_moe` exigia letras; modelo micro devolveu espaços | cobrar o **mecanismo**, não o texto |
| Lista fixa que cresce | teste da API fixava cinco presets; o sexto quebrou | `<=` para o núcleo estável, `in` para o novo |
| Dependência de ordem entre testes | estado global vazando | `tmp_path` sempre; nada de arquivo no repo |

## 7. Checklist antes de dizer "está testado"

- [ ] o teste falha se eu quebrar o código de propósito (teste de mutação manual)
- [ ] nenhum assert sem número ou sem propriedade clara
- [ ] roda em CPU, sem rede, em menos de 30 s (ou marcado `slow`)
- [ ] usa `tmp_path` e não escreve no repositório
- [ ] o nome do teste diz o que ele prova, não o que ele chama
- [ ] a evidência (comando + saída) foi para `specs/PLANO.md`
