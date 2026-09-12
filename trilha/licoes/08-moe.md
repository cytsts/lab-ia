# Lição 8 — MoE: mais parâmetros sem mais conta

**Pré-requisito:** Lições 1 a 7. **Tempo:** ~50 min.

> Esta lição encontrou um bug de verdade no laboratório. Ele está no fim, com a
> medição que o expôs — vale ler até lá.

## O que você vai entender

Como uma camada com 4 especialistas calcula só 1 por token, qual é o risco de
colapso do roteador, e o que a perda auxiliar de balanceamento faz — medido pelo
gradiente, não por fé.

## A ideia

Um FFN denso aplica a mesma conta a todos os tokens. O MoE troca isso por:

```
g = softmax(Wr · x)          # probabilidades por especialista
top-k(g)                     # só os k maiores recebem o token
y = soma g_i · Especialista_i(x)   # soma ponderada dos escolhidos
```

Isso separa duas contas que costumavam ser a mesma:

- **parâmetros totais** (o que você armazena): todos os especialistas;
- **parâmetros ativos por token** (o que você paga de conta): só os k escolhidos.

O risco é estrutural: se o roteador aprende cedo que o especialista 0 "funciona", ele
manda tudo para lá, os outros nunca recebem gradiente e o MoE degenera num FFN caro.
A defesa é a **perda auxiliar** de balanceamento:

```
aux = n · soma_i (f_i · P_i)
```

onde `f_i` é a fração de tokens roteada para o especialista `i` (contagem dura) e `P_i` é a
probabilidade média que o roteador dá a ele (suave, diferenciável). O piso é `1,0` com
roteamento uniforme e o teto é `n` no colapso total.

## No núcleo

**No núcleo:** `core/labia/models/gpt.py` → `Roteador.forward`

```python
probs = torch.softmax(self.peso(x), dim=-1)
val, idx = probs.topk(self.k, dim=-1)
destino = torch.zeros_like(probs).scatter_(1, idx, val)
with torch.no_grad():
    fracao = destino.gt(0).float().mean(0)      # f_i: contagem dura
    self.ultima_fracao = fracao.detach()
aux = self.n * (fracao * probs.mean(0)).sum()   # P_i carrega o gradiente
```

**No núcleo:** `core/labia/models/gpt.py` → `CamadaMoE.forward` — o laço que aplica
cada especialista só aos tokens que o escolheram.

**No núcleo:** `core/labia/models/gpt.py` → `GPT.contar_parametros_ativos` — a conta
do custo top-k.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e08_moe.py
```

```
1) parâmetros totais x ativos (4 especialistas, dim 256, 6 camadas)
   top1: 15.313.408 totais · 5.853.184 ativos por token (38.2%)
   top2: 15.313.408 totais · 9.006.592 ativos por token (58.8%)

2) o valor da perda auxiliar (n=4)
   roteamento uniforme  → 1.0  (piso)
   roteamento parcial   → 2.08
   roteamento colapsado → 4.0  (teto = n)

3) para onde a auxiliar empurra (gradiente projetado na direção das entradas)
   roteador colapsado: uso [0.6719, 0.3281, 0.0, 0.0] · aux 1.9036
   especialista 0: projeção +0.0068 → diminui o logit (o gradiente empurra para usar menos)
   especialista 1: projeção -0.0246 → aumenta o logit (o gradiente empurra para usar mais)
   especialista 2: projeção +0.0089 → diminui o logit (o gradiente empurra para usar menos)
   especialista 3: projeção +0.0089 → diminui o logit (o gradiente empurra para usar menos)

4) treino curto partindo do MESMO colapso (120 passos em CPU)
   sem auxiliar         [0.9453, 0.0547, 0.0, 0.0] → [0.8164, 0.1289, 0.0469, 0.0078]
                        dominante 94.5% → 81.6% · aux 2.7213
   com auxiliar (0,5)   [0.9453, 0.0547, 0.0, 0.0] → [0.2656, 0.2383, 0.2578, 0.2383]
                        dominante 94.5% → 26.6% · aux 1.0056
```

- **item 1**: 15,3 M de parâmetros guardados, 5,85 M usados por token — **38,2%** com
  top-1. É o número que importa para tempo e VRAM; o "tamanho do modelo" é o outro.
- **item 2**: a auxiliar tem piso e teto. Não é uma perda que "desce para zero": ela mede
  desequilíbrio, e o alvo é ficar perto de 1,0.
- **item 3**: aqui está o mecanismo. A projeção do gradiente é **positiva no especialista
  dominante** (o que, em descida de gradiente, *diminui* o logit dele) e **negativa no
  subutilizado** (aumenta). A auxiliar não "pede" equilíbrio por decreto: ela inclina a
  ladeira nessa direção.
- **item 4**: partindo do mesmo colapso de 94,5%, sem auxiliar o roteador fica em 81,6%
  (não se recupera sozinho); com auxiliar vai a **26,6%** — praticamente uniforme — e o
  valor da auxiliar cai de 2,72 para **1,0056**, o piso.

## O bug que esta lição encontrou

O experimento foi escrito para medir o item 3 — e não conseguiu: o `backward()` da perda
auxiliar estourava com *"element 0 of tensors does not require grad"*. A causa:

```python
# ANTES (errado)
with torch.no_grad():
    fracao = destino.gt(0).float().mean(0)
    aux = self.n * (fracao * probs.mean(0)).sum()   # <-- dentro do no_grad!
```

O produto inteiro estava dentro do `no_grad`, então `aux` saía **desligada do grafo**.
Somada à perda como `perda + coef_auxiliar * aux`, ela era uma constante: o
`coef_auxiliar` não tinha efeito nenhum no treino. A RF3 da spec G4 pedia exatamente o
contrário ("somada à perda de linguagem"), e o número que aparecia em `metricas.jsonl`
era um **diagnóstico**, não um sinal de treino.

Consertado, o item 4 é a prova: com o coeficiente ativo, o roteador sai de 94,5% para
26,6%; sem ele, fica em 81,6%. Antes do conserto, os dois casos davam exatamente o mesmo
resultado — e foi essa igualdade suspeita que levou ao bug.

Regressão coberta por dois testes em `tests/g4/`: a auxiliar precisa ter gradiente, e o
valor do coeficiente precisa mudar o gradiente do roteador.

## Exercícios

1. **Preveja.** Troque `top_k` de 1 para 2 no item 1. *Esperado:* ativos sobem de 38,2%
   para 58,8% (medido) e o total não muda — mais conta por token, mesmos parâmetros.
2. **Confirme o conserto.** Mova a linha `aux = ...` para dentro do
   `with torch.no_grad()`. Rode `pytest tests/g4 -q`. *Esperado:* dois testes falham —
   os que exigem gradiente e efeito do coeficiente.
3. **Onde o MoE ganha.** No corpus pequeno deste laboratório, o run real da G4
   (`lab-ia comparar g4-moe-zero g1-treino-zero`) mostra **paridade**, não vitória: MoE
   4,7025 contra denso 4,6731. *Esperado ao ler:* em corpus pequeno, dividir capacidade não
   ajuda — o ganho do MoE aparece em escala grande, que não cabe em 8 GB de VRAM.

## Se quiser ir mais fundo

- `specs/G4.md`: a spec, incluindo a correção do critério original (que exigia o MoE
  vencer — folclore de escala grande).
- `core/labia/bancada/comparar.py`: como a bancada classifica um run MoE que decorou
  (`overfit`) — foi exatamente o que aconteceu com o top-2.
