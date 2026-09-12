# Lição 5 — o laço de treino: lr, warmup, clip e o que a perda diz

**Pré-requisito:** Lições 1 a 4. **Tempo:** ~45 min.

## O que você vai entender

Como o learning rate é agendado, por que existe warmup, o que o grad clip faz (e o
que ele não faz), e como distinguir "ainda não aprendeu" de "está divergindo".

## A ideia

Treinar é descer uma ladeira de milhões de dimensões. O **learning rate** é o tamanho
do passo. Três decisões práticas:

1. **Warmup.** Nos primeiros passos o modelo é aleatório e o gradiente é errático.
   Começar com um passo grande nesse momento costuma explodir. O warmup sobe o lr
   linearmente de ~0 até o pico.
2. **Decaimento.** No fim, passos grandes fazem a perda oscilar em volta do mínimo.
   O agendamento cosseno desce suavemente do pico até um piso.
3. **Clip.** Limita a norma do gradiente a um teto. Protege contra um lote ruim, não
   conserta um lr errado.

## No núcleo

**No núcleo:** `core/labia/trainer/treino.py` → `fator_lr`

```python
def fator_lr(cfg, passo):
    if passo < cfg.warmup:
        return cfg.lr * (passo + 1) / max(1, cfg.warmup)          # sobe linear
    span = max(1, cfg.passos - cfg.warmup)
    coef = 0.5 * (1.0 + math.cos(math.pi * (passo - cfg.warmup) / span))
    return cfg.minimo_lr + (cfg.lr - cfg.minimo_lr) * coef         # desce em cosseno
```

**No núcleo:** `core/labia/trainer/treino.py` → `executar_treino`

No laço, a ordem é sempre a mesma e importa:

```python
for grupo in otimizador.param_groups:      # 1. aplica o lr do passo
    grupo["lr"] = lr
xb, yb = lote_trem(...)                    # 2. sorteia o lote (determinístico pela semente)
with torch.autocast(...): _, perda = modelo(xb, yb)
perda.backward()                           # 3. gradiente
clip_grad_norm_(modelo.parameters(), 1.0)  # 4. corta a norma
otimizador.step()                          # 5. atualiza
otimizador.zero_grad(set_to_none=True)     # 6. limpa
```

O AdamW merece uma nota: ele mantém duas médias móveis por parâmetro (primeiro e
segundo momento) e por isso "lembra" dos gradientes anteriores. É o que permite um
lr de 3e-4 funcionar onde SGD puro precisaria de outro mundo de ajuste.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e05_treino.py
```

Saída real:

```
1) a curva de learning rate do treino (warmup + cosseno)
    passo           lr
        0   0.00015000
       10   0.00165000
       20   0.00300000
       60   0.00206717
       90   0.00085649
      119   0.00030067

2) dois treinos curtos em CPU (mesmo dado, mesma semente, só o lr muda)
   lr=0.003  inicial 5.027 → final 0.856 · validação 0.818 · maior perda no meio 6.391
   lr=1.0    inicial 20.649 → final 3.346 · validação 3.101 · maior perda no meio 92.330
```

Leituras:

- **a curva**: no passo 0 o lr é `lr/warmup`, sobe até o pico no passo do warmup, e
  desce até perto do piso no fim. O piso `minimo_lr` nunca é ultrapassado.
- **lr=1,0 é 333× maior que 0,003.** Não "aprendeu mais devagar": começou em 20,6 de
  perda (o modelo já estava sendo jogado para fora antes de aprender qualquer coisa) e
  no meio do caminho chegou a **92,3**. Só terminou em 3,35 porque o `grad_clip` segurou.
- **a validação (0,818) ficou próxima do treino (0,856)** — com corpus sintético
  repetitivo, decorar é fácil; é o caso em que "validação boa" não significa muita
  coisa. Em corpus real, a distância entre as duas é o seu medidor de overfit.

## O que isso muda na prática

O sinal que você procura em `metricas.jsonl`:

| o que você vê | o que provavelmente é | o que fazer |
|---|---|---|
| train e val caem juntos | aprendendo | continue |
| train cai, val sobe | decorando (overfit) | pare antes, mais dados, mais dropout |
| perda oscila muito | lr alto demais | reduza o lr ou aumente o warmup |
| perda em `nan` | divergiu | lr ÷ 10, warmup maior, olhe os dados |
| perda parada desde o começo | gradiente cortado ou lr baixo demais | verifique `zero_grad` e `lr` |

A bancada automatiza essa leitura: `lab-ia comparar` classifica cada run em
`estavel`, `estagnado`, `deteriorando`, `overfit`, `instavel` ou `divergiu`, com
o comentário dizendo o que fazer. Mas o critério é este aqui — vale saber de cor.

## Exercícios

1. **Preveja.** No experimento, mude `warmup` de 20 para 60 (mantendo 120 passos).
   *Esperado:* o pico do lr acontece no passo 60 e a descida em cosseno fica comprimida
   — e o treino curto termina com perda um pouco pior, porque passou mais tempo
   aquecendo do que aprendendo.
2. **O clip segura, não conserta.** Rode o experimento com `lr_alto=100.0`.
   *Esperado:* perda inicial ainda maior e `maior_perda` na casa das centenas — o clip
   limita o *passo*, não impede o modelo de sair da região útil.
3. **Compare de verdade.** Rode
   ```bat
   lab-ia varrer --base configs/livros-rapido.yaml --grade lr=0.0003,0.0006 --passos 300 --prefixo sw-licao5
   ```
   e leia o "efeito de cada chave". *Esperado:* o lr maior ganha nesta fase inicial — e o
   veredito de cada variante deve sair `ainda_caindo`, porque 300 passos é pouco.

## Se quiser ir mais fundo

- `specs/B1.md`: por que a estimativa de tempo tem dois termos (custo fixo por passo
  + FLOPs) — a parte do laço que não é conta de GPU.
- `specs/B2.md`: os limiares que a bancada usa para chamar um run de "overfit" ou
  "estagnado", e por que eles são heurísticas declaradas, não leis.
