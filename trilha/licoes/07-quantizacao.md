# Lição 7 — quantização: o que se perde ao encolher os pesos

**Pré-requisito:** Lições 1 a 6. **Tempo:** ~35 min.

## O que você vai entender

O que int8 e NF4 fazem com uma matriz de pesos, quanto cada um encolhe de verdade, e
por que 4 bits não-uniformes conseguem competir com 8 bits uniformes (e onde eles
perdem).

## A ideia

Guardar cada peso em 32 bits é confortável e caro. Quantizar é escolher um conjunto
pequeno de valores possíveis e guardar só o *índice* — mais uma escala para recuperar
a magnitude.

**int8** (simétrico, por linha de saída): `q = round(w / absmax · 127)`, e
`w ≈ q · absmax / 127`. 256 níveis igualmente espaçados.

**NF4** (QLoRA): 16 níveis **não** uniformes, posicionados nos quantis de uma
gaussiana — mais níveis perto de zero, onde os pesos se concentram, e menos nas
bordas. Mais uma escala por bloco de 64 pesos, para não perder os outliers.

O ganho de armazenamento é direto: int8 = 8 bits por peso, NF4 = 4 bits + 16/64 bits
de escala ≈ **4,25 bits por peso**.

## No núcleo

**No núcleo:** `core/labia/models/quant.py` → `quant_int8`

```python
absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)   # escala por linha
q = torch.clamp((w / absmax * 127).round(), -127, 127).to(torch.int8)
```

**No núcleo:** `core/labia/models/quant.py` → `quant_nf4`

```python
b = w.view(out, ent // bloco, bloco)          # blocos de 64
xn = (b / absmax).clamp(-1.0, 1.0)            # normaliza por bloco
codigos = torch.argmin((xn.unsqueeze(-1) - niveis).abs(), dim=-1)   # vizinho mais próximo
empacotado = (pares[..., 1] << 4) | pares[..., 0]                   # 2 códigos por byte
```

Note o empacotamento: dois códigos de 4 bits cabem em um byte. É por isso que o
`state_dict` quantizado é realmente menor — o laboratório mede o tamanho dos tensores
persistidos, não um número teórico.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e07_quantizacao.py
```

Saída real (matriz 256×256 com uma coluna outlier, como acontece em transformers):

```
  modo  bytes fp32  bytes quant   fator  bits/param  erro relativo
  int8      262144        66576   3.94x       8.127        0.03372
   nf4      262144        34832   7.53x       4.252        0.13039

NF4 tem 16 níveis entre -1.0 e 1.0; o espaço entre eles é
  0.3038 na borda e 0.0796 no centro — quantis de gaussiana, não passos iguais.

onde o erro cai (NF4):
  pesos pequenos: 0.005791
  pesos grandes : 0.012947
```

- **fator 3,94× para int8** — perto do 4× teórico (o resto é a escala fp32 por linha).
- **fator 7,53× para NF4** — melhor que o 8× teórico? Não: os bytes contados são
  apenas os dos tensores quantizados (códigos + escalas fp16). O `bits_por_parametro`
  de **4,252** é a conta honesta.
- **o erro relativo do NF4 é ~3,9× o do int8** (0,130 contra 0,034). Isto é a parte que
  costuma ser omitida: NF4 encolhe mais e erra mais **por peso**. O que salva o NF4 é que
  o erro é distribuído de forma inteligente (item abaixo) e que, ao treinar o adaptador
  por cima, o modelo compensa.
- **erro maior nos pesos grandes** (0,0129) do que nos pequenos (0,0058): os níveis são
  finos perto de zero e grossos nas bordas. É o preço de gastar níveis onde há mais peso.

## O que isso muda na prática

O laboratório mede o efeito no modelo inteiro, não numa matriz isolada — o run da G3
está em `runs/g3-g1-int8` e `runs/g3-g1-nf4`:

| modo | fator nos alvos | bits efetivos | perda antes → depois |
|---|---|---|---|
| int8 | 3,95× | 8,09 | 8,729 → 8,732 |
| nf4 | 7,11× | 4,50 | 8,729 → 8,881 |

(Leia em `lab-ia comparar`: os dois saem com veredito `quantizacao`, com o
comentário trazendo exatamente esses números.)

Ou seja: **8 bits custou 0,003 nats de perda; 4 bits custou 0,152 nats** — 50× mais
caro em qualidade, por 1,8× menos memória. Essa é a troca, medida e não estimada.

## Exercícios

1. **Preveja.** Rode o experimento com `bloco=32` em vez de 64. *Esperado:* o erro
  relativo do NF4 **cai** (blocos menores = escalas mais locais) e os bits por parâmetro
  **sobem** (mais escalas por peso). Confirme os dois lados da troca.
2. **Onde dói.** Remova a coluna outlier (`w[:, 0] *= 40.0`) e rode de novo.
   *Esperado:* o erro relativo do int8 despenca — a quantização por linha sofre justamente
   com outliers.
3. **No modelo.** Rode `lab-ia quantizar --run g1-treino-zero --saida g3-teste --modo nf4`
   e confira o `tamanhos.json`. *Esperado:* os mesmos 7,11× e 4,50 bits do run da G3
   (mesma semente, mesmo corpus de avaliação).

## Se quiser ir mais fundo

- `specs/G3.md`: a meta de quantização, incluindo o comparativo com `bitsandbytes`.
- `core/labia/models/lora.py` → `NoLinear`: onde a base quantizada e o adaptador
  treinável convivem (é o QLoRA).
