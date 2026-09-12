# Trilha de estudo do Lab-IA

Material para **entender** o que os números do laboratório estão dizendo. Escrito em
português, apontando para o código real deste repositório — não para exemplos
genéricos que não existem aqui.

Cada lição tem a mesma forma: uma ideia curta, **onde ela vive no núcleo** (arquivo e
função, com o trecho citado), **um experimento que você roda** e imprime números
medidos nesta máquina, e exercícios com resultado esperado.

## Como usar

```bat
:: ver a trilha inteira
.venv\Scripts\python -m labia.cli trilha

:: ler uma lição (imprime o texto)
.venv\Scripts\python -m labia.cli trilha --licao 3

:: rodar o experimento da lição 3
.venv\Scripts\python -m labia.cli trilha --licao 3 --rodar

:: rodar todos os experimentos de uma vez (~3 min)
.venv\Scripts\python -m labia.cli trilha --rodar-tudo

:: conferir se as lições ainda citam código que existe
.venv\Scripts\python -m labia.cli trilha --conferir
```

Os experimentos também rodam direto, sem passar pela CLI:
`.venv\Scripts\python trilha\experimentos\e03_atencao.py`. E há um caderno Jupyter
gerado de cada experimento em `trilha/notebooks/` (precisa de
`pip install jupyterlab`).

**Ritmo sugerido:** uma lição por sessão, com o código aberto do lado. Ler sem rodar não
funciona aqui — os números são o ponto.

## Parte 1 — criar um modelo (do zero)

| # | Lição | Ideia central | Experimento |
|---|---|---|---|
| 1 | [O tensor, o grafo e o gradiente](licoes/01-tensor-e-gradiente.md) | autograd exato, acúmulo de gradiente | `e01_autograd.py` |
| 2 | [Tokenização](licoes/02-tokenizacao.md) | BPE, chars/token, o custo do vocabulário | `e02_tokenizacao.py` |
| 3 | [Atenção e máscara causal](licoes/03-atencao-causal.md) | Q·Kᵀ, softmax, esconder o futuro | `e03_atencao.py` |
| 4 | [O bloco e o resíduo](licoes/04-bloco-e-residuo.md) | LayerNorm, resíduo, escala da inicialização | `e04_bloco.py` |
| 5 | [O laço de treino](licoes/05-laco-de-treino.md) | lr, warmup, clip, ler a perda | `e05_treino.py` |

## Parte 2 — otimizar e adaptar um modelo

| # | Lição | Ideia central | Experimento |
|---|---|---|---|
| 6 | [LoRA](licoes/06-lora.md) | adaptar 1–3% dos parâmetros, posto `r`, mesclar | `e06_lora.py` |
| 7 | [Quantização](licoes/07-quantizacao.md) | int8 x NF4: tamanho, erro e bits efetivos | `e07_quantizacao.py` |
| 8 | [MoE e roteamento](licoes/08-moe.md) | parâmetros ativos, colapso, perda auxiliar | `e08_moe.py` |
| 9 | [Raciocínio (CoT/ToT)](licoes/09-raciocinio.md) | tokens por acerto, e a métrica medindo formato | `e09_raciocinio.py` |
| 10 | [Agentes](licoes/10-agentes.md) | skills com assinatura e log auditável | `e10_agentes.py` |

## Onde isso se encaixa

A bancada responde *o que fazer a seguir* (`comparar`, `varrer`, curvas). A trilha responde
*por quê*. Uma sem a outra vira fé: números sem entendimento, ou teoria sem medição.

- Guia de uso do laboratório: `docs/bancada.md`
- Decisões de engenharia e evidências: `specs/PLANO.md` e `specs/B*.md`
- Código do modelo: `core/labia/models/gpt.py` (o arquivo mais denso do projeto, e o
  mais curto dada a importância)

## A trilha já encontrou um bug

A Lição 8 foi escrita para medir o mecanismo da perda auxiliar do MoE — e descobriu que
ela **não tinha gradiente**: somada à perda como constante, o `coef_auxiliar` não fazia
nada. Está corrigido, com dois testes de regressão, e a lição conta a história inteira.
É o argumento mais forte a favor deste formato de material: medir para ensinar expõe o
que a leitura não expõe.
