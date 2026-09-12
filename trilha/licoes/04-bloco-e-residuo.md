# Lição 4 — o bloco, o resíduo e a inicialização

**Pré-requisito:** Lições 1 a 3. **Tempo:** ~35 min.

## O que você vai entender

O que há dentro de um bloco transformer, o que o "+" do resíduo faz com a
magnitude da ativação ao longo da profundidade, e por que a inicialização dos pesos
não é detalhe.

## A ideia

Um bloco é sempre a mesma forma:

```
x = x + Atencao(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

- **LayerNorm** normaliza cada posição (média 0, desvio 1) e devolve escala e viés
  aprendíveis. Serve para manter os números numa faixa utilizável.
- **Resíduo** (o `x +`) é o que faz a profundidade funcionar: a informação tem um
  caminho direto de ponta a ponta, e cada bloco só precisa aprender um *ajuste*.
- **FFN** é uma rede densa que expande para `4·dim` e volta. É onde mora a maior
  parte dos parâmetros de um transformer.

O detalhe que a intuição erra: como o resíduo **soma**, se cada bloco devolver algo
com magnitude comparável à entrada, isso se acumula. Em 6 camadas pode virar 5x; em
48 camadas, catástrofe — e com ela o gradiente explode ou some. A defesa clássica
(GPT-2) é escalar as projeções de saída por `1/sqrt(2·camadas)`.

## No núcleo

**No núcleo:** `core/labia/models/gpt.py` → `BlocoTransformer.forward`

```python
x = x + self.atencao(self.ln1(x))
saida, aux = self.mlp(self.ln2(x))
return x + saida, aux
```

**No núcleo:** `core/labia/models/gpt.py` → `GPT.init_pesos`

```python
escala = 1.0 / math.sqrt(2 * self.cfg.camadas)
for bloco in self.blocos:
    bloco.atencao.proj.weight.mul_(escala)   # a projeção de SAÍDA da atenção
    red.fc2.weight.mul_(escala)              # a projeção de SAÍDA da FFN
```

Escala só a **saída**, nunca a entrada: é a contribuição do bloco para o resíduo que
precisa ser pequena.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e04_bloco.py
```

Saída real (modelo de 6 camadas, dim 64, mesmo dado, só a escala muda):

```
desvio-padrão da ativação depois de cada bloco
  com escala sqrt(2·camadas): [0.0142, 0.0162, 0.0181, 0.0195, 0.0211, 0.0227, 0.0243]
  sem escala                : [0.0142, 0.0302, 0.0425, 0.0518, 0.0607, 0.0687, 0.0767]

crescimento da entrada até a última camada:
  com escala: 1.71x
  sem escala: 5.40x
```

O experimento constrói dois modelos idênticos e **desfaz** a escala em um deles
(multiplica de volta por `sqrt(2·camadas)`). Com a escala, a ativação cresce 1,71x
da entrada até a saída do último bloco; sem ela, 5,40x. Com 6 camadas a diferença é
incômoda; com 24 seria inviável.

## O que isso muda na prática

Sintomas de inicialização errada num modelo que você mesmo montou:

- perda que vira `nan` nos primeiros passos (gradiente explodindo);
- perda que não desce de jeito nenhum (gradiente sumindo);
- sensibilidade absurda ao `lr`: só funciona numa faixa estreitíssima.

A bancada tem um diagnóstico para isso: se um run seu morre assim, `lab-ia comparar`
o marca como `divergiu` ou `instavel`, com o comentário dizendo o que olhar.

## Exercícios

1. **Preveja.** Mude `camadas` de 6 para 2 no experimento. *Esperado:* a diferença
   encolhe — a escala é `1/sqrt(2·2)` = 0,5 contra `1/sqrt(2·6)` = 0,29; menos camadas,
   menos acúmulo.
2. **Do outro lado.** Multiplique as projeções por 3 em vez de desfazer a escala.
   *Esperado:* crescimento muito acima de 5,40x e, com sorte, `inf` na última camada.
3. **Confira o gradiente.** Depois de rodar o experimento, imprima
   `max(p.abs().max() for p in modelo.parameters())` para os dois modelos. *Esperado:* a
   norma do gradiente acompanha o crescimento da ativação — é o mesmo fenômeno visto
   por outro ângulo.

## Se quiser ir mais fundo

- `specs/G1.md`: onde a arquitetura do nano-GPT do laboratório foi decidida.
- A regra `1/sqrt(2·camadas)` vem do GPT-2; a variante `1/sqrt(camadas)` aparece em
  outros modelos — o que não muda é o princípio: manter a contribuição residual pequena.
