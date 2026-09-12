# Lição 2 — tokenização: o que o modelo realmente vê

**Pré-requisito:** Lição 1. **Tempo:** ~30 min.

## O que você vai entender

Por que não se treina em letras nem em palavras, o que o BPE faz, e o que o tamanho
do vocabulário compra e o que ele custa.

## A ideia

Rede neural não come texto: come índices. O tokenizador é a função que transforma
"noite" em, digamos, `[1218]`. As três opções clássicas:

| unidade | vocabulário | sequência | problema |
|---|---|---|---|
| caractere | minúsculo (~50) | longuíssima | o modelo gasta capacidade aprendendo ortografia |
| palavra | enorme (100k+) e aberto | curta | palavra nova = desconhecida |
| **subpalavra (BPE)** | escolhido (2k–50k) | média | nenhum dos dois |

**BPE** (byte-pair encoding) começa com os bytes e vai fundindo os pares mais
frequentes, um por vez, até o vocabulário atingir o tamanho pedido. O resultado é
que palavras frequentes viram um token só e palavras raras se quebram em pedaços —
e nada fica "fora do vocabulário", porque no fundo tudo é byte.

## No núcleo

**No núcleo:** `core/labia/trainer/tokenizacao.py` → `treinar_tokenizer_ptbr`

Três linhas explicam quase tudo:

```python
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
treinador = trainers.BpeTrainer(vocab_size=vocab_tam, special_tokens=ESPECIAIS,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
```

O `initial_alphabet` é o detalhe que quase ninguém nota: ele **força os 256 bytes** a
estarem no vocabulário. É por isso que pedir 128 peças devolve 258 — e é por isso
que nenhum texto, em nenhum idioma, quebra o tokenizador.

**No núcleo:** `core/labia/trainer/dados.py` → `montar_dataset`

É aqui que o texto vira o par (entrada, alvo) do treino:

```python
x = ... ids[i*passo : i*passo + janela]
y = ... ids[i*passo + 1 : i*passo + janela + 1]
```

O alvo é a entrada **deslocada um token**. Prever o próximo token não é uma tarefa
que alguém escolheu: é a única coisa que o texto oferece de graça.

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e02_tokenizacao.py
```

Saída real (200 mil caracteres do corpus pt-BR deste laboratório):

```
  vocab    real  chars/token  tokens da frase
    128     258         0.98               56
    512     512          2.0               30
   4096    4096         3.43               19

frase: A noite estava fria e chuvosa quando ele decidiu partir.
tokenizada com o maior vocabulário:
['A', 'Ġnoite', 'Ġestava', 'Ġf', 'ria', 'Ġe', 'Ġch', 'u', 'vos', 'a',
 'Ġquando', 'Ġe', 'le', 'Ġde', 'ci', 'diu', 'Ġpar', 'tir', '.']
```

- **0,98 chars/token com vocabulário 128**: menos de um caractere por token. A frase
  vira 56 pedaços — o modelo gasta contexto e passos aprendendo a juntar letras.
- **3,43 chars/token com 4096**: a mesma frase em 19 tokens. Sobra contexto para o
  que interessa.
- O `Ġ` que aparece nos pedaços é o espaço (a marca do pré-tokenizador byte-level).
  Repare que "noite" saiu inteira, mas "fria" virou `f` + `ria`: o BPE aprendeu
  frequência, não morfologia.

## O que isso muda na prática

O tamanho do vocabulário é uma troca explícita:

- embedding custa `vocab × dim` parâmetros, e a camada de saída também (no núcleo
  ela é **atada** ao embedding: `self.cabeca.weight = self.wte.weight`, então é o
  mesmo custo, não o dobro);
- vocabulário maior → sequência mais curta → janela cobre mais texto → mais contexto
  por passo;
- mas cada token novo é um parâmetro que só aparece raramente no treino, e fica mal
  treinado.

Em corpus pequeno, 4096 já é generoso. Em corpus de 1 GB, 32k–50k costuma compensar.

## Exercícios

1. **Preveja.** Rode o experimento adicionando `2048` à lista `tamanhos`. *Esperado:*
   chars/token entre o de 512 e o de 4096 (na casa de 2,8–3,1).
2. **Custo do vocabulário.** Com `dim=256`, calcule na mão quanto custa o embedding
   com vocab 4096 e com 32768. *Esperado:* 1,05 M e 8,39 M parâmetros — quase 60% de um
   modelo de 6 M iria só para o embedding. Confira com
   `lab-ia novo --nome sonda --dados livros-ptbr --vocab-bpe 32768 --saida .lab-ia/sonda.yaml`.
3. **Seu corpus.** Rode `.venv\Scripts\python trilha\experimentos\e02_tokenizacao.py` depois de
   apontar `data/corpus_ptbr.txt` para um texto seu. *Esperado:* texto com muitas
   palavras raras (nomes próprios, código) dá chars/token menor — mais peças por palavra.

## Se quiser ir mais fundo

- `core/labia/bancada/dados.py` → `estimar_tokens`: o 3,6 chars/token que a bancada usa
  para dimensionar, medido no corpus do laboratório.
- `specs/G1.md`: por que o vocabulário da G1 ficou em 4096.
