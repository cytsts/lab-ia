# Lição 6 — LoRA: adaptar sem reescrever o modelo

**Pré-requisito:** Lições 1 a 5. **Tempo:** ~40 min.

## O que você vai entender

Como adaptar um modelo já treinado a uma tarefa nova treinando 1–3% dos parâmetros,
por que o adaptador começa como no-op, o que o posto `r` limita, e como desfazer tudo
(mesclar) sem perder o resultado.

## A ideia

Fine-tuning clássico atualiza todos os pesos. LoRA parte de uma observação: a
*mudança* que o ajuste precisa fazer costuma ter posto baixo — ela não precisa de
toda a liberdade de uma matriz completa.

Então congela-se `W` e aprende-se um delta fatorado:

```
W' = W + (α/r) · B·A        A: (r × entrada)   B: (saída × r)   r ≪ min(entrada, saída)
```

Três detalhes que o experimento mede:

- **B começa em zero** → o delta é zero → o modelo adaptado é idêntico à base no
  primeiro passo. Você parte do que já funciona.
- **o posto de B·A é no máximo r** — é a hipótese do método, não um efeito colateral.
- **mesclar** (`W ← W + (α/r)·B·A`) devolve um `Linear` comum: a inferência não paga
  nada pelo adaptador.

## No núcleo

**No núcleo:** `core/labia/models/lora.py` → `NoLinear.forward`

```python
y = F.linear(x, self.peso_base(), self.vies)                     # base congelada
if self.lora_A is not None:
    y = y + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.escala
```

**No núcleo:** `core/labia/models/lora.py` → `aplicar_lora`

```python
for p in modelo.parameters():
    p.requires_grad_(False)          # congela TUDO
...
for nome, p in modelo.named_parameters():
    if nome.endswith(".lora_A") or nome.endswith(".lora_B"):
        p.requires_grad_(True)       # e liga só o adaptador
```

**No núcleo:** `core/labia/models/lora.py` → `mesclar_lora` — a fusão, que devolve
`nn.Linear` e remove os `NoLinear` do modelo.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e06_lora.py
```

Saída real (modelo pequeno: dim 64, 2 camadas, r=8, α=16):

```
1) o adaptador começa como no-op (B iniciado em zero)
   diferença máxima na saída vs modelo base: 0.000e+00  (idêntico: True)

2) quantos parâmetros ficam treináveis
   modelo antes do LoRA : 109.824 parâmetros, TODOS treináveis
   modelo com adaptador : 126.208 (+16.384 do ramo A·B)
   treináveis com r=8   : 16.384 em 8 módulos
   proporção            : 12.98%

3) o posto da atualização é limitado por r
   r = 4 → posto medido da matriz B·A = 4
   ramo treinável: 256 parâmetros contra 1024 da base

4) mesclar devolve um Linear comum com a mesma saída
   módulos mesclados: 8 · diferença máxima: 1.639e-07
   ainda há NoLinear no modelo? False · tipos de Linear: ['Linear']
```

Leituras:

- **item 1**: `0.000e+00` e `idêntico: True` — não é "quase igual", é igual. O
  adaptador entra em cena sem estragar o que a base sabe.
- **item 2**: 12,98% aqui. Esse número **depende do tamanho do modelo e de r**: no run
  real da G2 (dim 256, 6 camadas, r=8) deu **1,3%**. Quanto maior a base, menor a fatia
  que o adaptador ocupa — é aí que LoRA compensa de verdade.
- **item 3**: o posto medido é exatamente `r`. Se você precisar de um ajuste de posto
  maior, aumente `r` — e pague em parâmetros.
- **item 4**: a diferença de 1,6e-07 é arredondamento de ponto flutuante (`bfloat16`/`fp32`),
  não perda de qualidade. Depois de mesclar não existe mais `NoLinear` no modelo.

## O que isso muda na prática

- **Custo de treino**: só o adaptador recebe gradiente; o otimizador guarda estado
  apenas para ele. Em modelo grande isso é a diferença entre caber na VRAM ou não.
- **Custo de armazenamento**: você guarda um adaptador de poucos MB por tarefa em vez
  de um checkpoint inteiro. Dez tarefas = dez adaptadores, não dez modelos.
- **QLoRA** = a mesma ideia com a base já quantizada (Lição 7): `tipo: qlora` na config
  de ajuste. A base congelada ocupa 4 bits por peso e o adaptador continua em fp32.

## Exercícios

1. **Preveja.** Rode o experimento mudando `r` de 8 para 16 em `proporcao_treinavel()`.
   *Esperado:* os treináveis dobram (o ramo tem r·(entrada+saída) parâmetros) e a
   proporção vai de 12,98% para perto de 26% neste modelo pequeno.
2. **Ajuste de verdade.** Rode um ajuste real com o corpus do laboratório:
   ```bat
   lab-ia ajustar --config configs/g2_ajuste_ciencia.yaml --run-id meu-ajuste
   ```
   e compare com `lab-ia comparar g2-ajuste-ciencia meu-ajuste`. *Esperado:* perda na
   tarefa caindo de ~8,8 para menos de 1,0 em poucos passos — o adaptador aprende o
   estilo da tarefa sem tocar na base.
3. **Mesclar muda a geração?** Gere texto antes e depois de mesclar (mesmo prompt,
   mesmo guloso). *Esperado:* o mesmo texto — a mesclagem é algebricamente idêntica;
   se divergir, o bug está na fusão, não no modelo.

## Se quiser ir mais fundo

- `specs/G2.md`: o que a meta exigiu (e mediu) no ajuste LoRA/QLoRA deste laboratório.
- `core/labia/trainer/ajuste.py` → `executar_ajuste`: o laço de ajuste, e como ele
  salva só o adaptador (`adaptador/lora.pt` + `meta.json`).
