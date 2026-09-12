# Lição 1 — o tensor, o grafo e o gradiente

**Pré-requisito:** nenhum. **Tempo:** ~25 min (10 lendo, 15 mexendo no código).

## O que você vai entender

Por que existe `.backward()`, o que exatamente ele calcula, e por que esquecer
`zero_grad` é o erro nº 1 de quem escreve um laço de treino à mão.

## A ideia

Um tensor do PyTorch é um número (ou uma grade de números) **mais um livro-caixa**.
Quando você marca `requires_grad=True`, toda operação feita com ele entra nesse
livro: o PyTorch monta um grafo direcional das operações, guardando o que precisa
para calcular a derivada depois.

A derivada que interessa em IA é sempre a mesma pergunta: *se eu mexer um
tiquinho neste parâmetro, quanto a perda muda?* O `backward()` percorre o grafo de
trás para frente (regra da cadeia) e **soma** essa resposta no campo `.grad` de
cada tensor que participou.

Duas consequências que confundem muita gente:

1. `.grad` **acumula**. Cada `backward()` soma ao que já estava lá. Se você não zerar
   entre os passos, o "gradiente" do passo 3 é a soma dos três passos.
2. O otimizador não sabe de nada disso. `otimizador.step()` só lê `.grad` e mexe no
   parâmetro. Quem decide *quando* o gradiente é somado, lido e zerado é o seu laço.

## No núcleo

**No núcleo:** `core/labia/trainer/treino.py` → `executar_treino`

Procure estas quatro linhas seguidas dentro do laço (por volta da linha 180):

```python
perda.backward()                                  # calcula e ACUMULA em .grad
torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.grad_clip)
otimizador.step()                                 # lê .grad e atualiza os pesos
otimizador.zero_grad(set_to_none=True)            # zera para o próximo passo
```

A ordem não é decorativa: o clip age **sobre o gradiente já calculado** e **antes** do
step; o zero vem **depois** do step, senão você apagaria o gradiente antes de usá-lo.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e01_autograd.py
```

Saída real deste laboratório:

```
1) autograd x derivada analítica  (f = soma(x^3))
   x        = [2.0, -3.0, 0.5]
   autograd = [12.0, 27.0, 0.75]
   analítico= [12.0, 27.0, 0.75]
   erro máximo = 0.00e+00

2) acúmulo de gradiente sem zero_grad  (f = x^2 em x=1.5, esperado 3.0)
   .grad depois de cada backward: [3.0, 6.0, 9.0]
   depois de zerar             : 3.0

3) autograd x diferença finita  (f = sin(x)·x²)
   autograd        = 2.9573243
   diferença finita= 2.9573243
   erro relativo   = 0.00e+00
```

Três leituras:

- **1)** o autograd não é aproximação: a derivada de `x³` em `x=2` é `3·2² = 12`, e é
  exatamente o que sai.
- **2)** o gradiente vira 3, depois 6, depois 9. O modelo não está "aprendendo três
  vezes mais rápido": está descendo uma ladeira que não existe. Zerar devolve 3,0.
- **3)** a diferença finita é o teste independente. Se você escrever uma camada à mão
  e o autograd discordar dela, o bug é seu — e esse teste pega.

## O que isso muda na prática

Quando um treino "não sai do lugar", a ordem de suspeitas é: (1) o gradiente está
zerado em algum ponto? (2) `requires_grad` está ligado no que deveria? (3) algum
`torch.no_grad()` ou `.detach()` cortou o grafo no meio? Este último é comum ao avaliar:
medir perda dentro de `no_grad` é correto, mas se você avaliar *durante* o treino e
esquecer de sair, o próximo `backward()` reclama.

## Exercícios

1. **Preveja antes de rodar.** No experimento 1, troque `x**3` por `torch.exp(x)` e a
   conta analítica por `torch.exp(x)`. *Esperado:* erro máximo 0,0 de novo — a derivada
   da exponencial é ela mesma.
2. **Quebre de propósito.** Troque `zero_grad(set_to_none=True)` por `zero_grad()` no
   laço de `e05_treino.py`. *Esperado:* o treino continua funcionando (a diferença entre
   os dois é só liberar memória: `set_to_none` deixa `.grad` como `None` em vez de um
   tensor de zeros) — mas o pico de memória cai.
3. **Confira na unha.** Comente a linha do `zero_grad` no experimento 5 e rode. *Esperado:*
   a perda para de cair e o `maior_perda` dispara — é o acúmulo do item 2 agindo num
   treino de verdade.

## Se quiser ir mais fundo

- `core/labia/trainer/treino.py` → `_perda_media`: onde o `no_grad()` aparece de verdade.
- O capítulo "automatic differentiation" de qualquer material de *deep learning
  systems*; aqui o que importa é o contrato: derivada exata, acumulada, por grafo.
