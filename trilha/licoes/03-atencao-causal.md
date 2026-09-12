# Lição 3 — atenção e a máscara causal

**Pré-requisito:** Lições 1 e 2. **Tempo:** ~40 min.

## O que você vai entender

O que a atenção calcula, por que ela é dividida por `sqrt(d)`, e qual é o erro
silencioso mais caro de um GPT escrito à mão: esquecer de esconder o futuro.

## A ideia

Para cada posição, a atenção responde a uma pergunta só: **de quem eu deveria
puxar informação para decidir o próximo token?**

```
Attention(Q, K, V) = softmax(Q·Kᵀ / sqrt(d)) · V
```

- `Q` (query) é o que a posição *procura*;
- `K` (key) é o que cada posição *oferece*;
- `V` (value) é o que cada posição *entrega* de fato;
- `Q·Kᵀ` dá uma nota de compatibilidade entre cada par de posições;
- `softmax` transforma as notas em pesos que somam 1;
- a divisão por `sqrt(d)` impede que as notas cresçam com a dimensão — sem ela, a
  softmax satura e o gradiente some.

"Causal" é a máscara: na posição `t`, as notas para `t+1, t+2, …` são zeradas antes da
softmax. **Sem isso o modelo vê a resposta que deveria prever.**

## No núcleo

**No núcleo:** `core/labia/models/gpt.py` → `AtencaoMultiCabeca.forward`

```python
q, k, v = self.qkv(x).split(d, dim=2)            # uma projeção só para os três
q = q.view(b, t, self.cabecas, self.cabeca_dim).transpose(1, 2)
...
saida = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)
```

Três coisas para notar:

1. `is_causal=True` é a máscara inteira. É uma linha — e é a diferença entre um
   modelo que aprende linguagem e um que aprende a copiar.
2. Uma projeção só (`qkv`) produz Q, K e V juntos. É mais rápido que três camadas
   separadas, e o resultado é idêntico.
3. `scaled_dot_product_attention` é a implementação fundida do PyTorch: a mesma
   fórmula acima, sem materializar a matriz de atenção inteira na memória.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e03_atencao.py
```

Saída real:

```
1) causalidade na AtencaoMultiCabeca do núcleo (core/labia/models/gpt.py)
   mudança vista nas posições ANTERIORES: 0.000e+00
   mudança vista na própria última     : 1.617
   causal? True

2) o que aconteceria sem a máscara (scaled_dot_product_attention)
   posição 0 com is_causal=True : 0.000e+00
   posição 0 com is_causal=False: 1.950
```

O experimento bagunça de propósito o **último** token e olha o que muda antes dele.
Com máscara, mudança **exatamente zero** (0,000e+00) — não é "pequena", é zero: a
posição 0 não tem caminho até a última. Sem máscara, a posição 0 muda 1,95: o futuro
vazou para o começo da frase.

## O que isso muda na prática

Um treino com vazamento tem uma assinatura inconfundível: **a perda cai rápido demais
e a validação também**, e o texto gerado é lixo. O modelo não aprendeu a prever: ele
encontrou um atalho — a resposta estava na entrada.

Se um dia você vir perda de 1,2 nats num modelo de 5 M parâmetros em corpus pequeno,
desconfie da máscara antes de comemorar. O valor "bom demais" é sintoma, não conquista.

## Exercícios

1. **Preveja.** No experimento, troque o deslocamento de +5,0 por +0,01 na perturbação.
   *Esperado:* a mudança na última posição cai proporcionalmente, e as anteriores
   continuam 0,000e+00. Causalidade não depende da magnitude.
2. **Ache o vazamento.** Comente `is_causal=True` em
   `core/labia/models/gpt.py` e rode `.venv\Scripts\python trilha\experimentos\e03_atencao.py`.
   *Esperado:* `causal? False`. Depois descomente — e rode a suíte
   (`.venv\Scripts\python -m pytest tests/g1 -q`) para ver o que mais quebra.
3. **Cabeças.** Mude `cabecas` de 4 para 1 em `config_pequena()`. *Esperado:*
   causalidade continua valendo (0,000e+00), mas `cabeca_dim` quadruplica e cada
   cabeça passa a olhar o todo em vez de especializar.

## Se quiser ir mais fundo

- `specs/G4.md`: o roteador do MoE decide *qual FFN* processa o token; a atenção decide
  *qual posição*. São dois mecanismos de escolha diferentes no mesmo bloco.
- `core/labia/models/gpt.py` → `GPT.gerar`: na geração, a janela é cortada em
  `idx[:, -janela_ctx:]` — a atenção causal é o que permite gerar token a token
  reaproveitando o contexto.
