# Bancada — trazer seus dados, gerar o treino, entender os números

Esta é a porta de entrada do laboratório para uso próprio. Ela existe porque antes
era preciso editar Python para treinar com o seu texto: o corpus vinha cravado em
`scripts/prepara_corpus.py` (dois livros do Gutenberg) e as configurações eram
cinco arquivos YAML copiados à mão.

O ciclo completo é:

**seus dados → manifesto → config explicada → treino → comparação → decisão**

---

## 1. Trazer seus dados

```bat
.venv\Scripts\python -m labia.cli dados --de C:\meus\textos --id meus-textos
```

`--de` aceita arquivo único, pasta (varre recursivamente), curinga
(`"C:\meus\textos\*.md"`) e `.jsonl`. A saída é:

```
data/meus-textos/trem.txt         texto de treino
data/meus-textos/val.txt          texto de validação (nunca visto no treino)
data/meus-textos/manifesto.json   o que foi lido, o que foi removido, hashes
```

O relatório impresso é a parte que importa entender:

```
dataset : livros-ptbr  (data\livros-ptbr)
fontes  : 1 arquivo(s)
          data\corpus_ptbr.txt  865 KB  utf-8-sig  sha e2ceee0fc9b0eb01
limpeza : 4000 parágrafos → 3319 (exatas 12, quase 0, curtos 669)
treino  : 800978 chars / 3153 parágrafos  (data\livros-ptbr\trem.txt)  sha fa1c17aeba6136ad
validação: 41138 chars / 166 parágrafos  (data\livros-ptbr\val.txt)  sha 46913c33e5d6a16a
idioma  : pt (pistas {'pt': 32065, 'en': 820}) · entropia 4.457 bits/char · ascii 98%
vocabulário: 16021 palavras únicas · 5.66 chars/palavra
vazamento val→trem: 80.7% do vocabulário de validação
tokens  : ~222493 no treino / ~11427 na validação (estimativa)
```

Leia linha por linha:

- **fontes / sha**: o hash do arquivo de origem. Se o corpus mudar, o hash muda —
  é assim que você sabe que dois experimentos usaram o mesmo dado.
- **limpeza**: quantos parágrafos entraram e saíram, e por quê. *exatas* = mesma
  sequência de palavras (ignorando caixa e pontuação); *quase* = 5-gramas muito
  parecidos; *curtos* = abaixo de `--min-chars` (padrão 20, que descarta lixo sem
  comer linhas de diálogo). Nada é removido em silêncio.
- **idioma / entropia**: pista de português por palavras funcionais; entropia em
  bits por caractere mede a variedade do texto (4,4 bits é prosa literária).
- **vazamento val→trem**: fração do vocabulário da validação que já aparece no
  treino. 80% num corpus de dois livros é esperado — a validação mede generalização
  *dentro do mesmo domínio*, não em texto novo. Corpus pequeno sempre vaza vocabulário.
- **tokens (estimativa)**: ~3,6 caracteres por token BPE em português. Serve para
  dimensionar passos antes de treinar; o número real sai do tokenizer do run.

Opções úteis: `--frac-val 0.1`, `--min-chars 0` (guarda tudo),
`--limiar-quase 0.6` (dedup mais agressiva, para texto raspado da web),
`--campo texto` (em `.jsonl`), `--listar` (ver o que já existe).

---

## 2. Gerar a configuração

```bat
.venv\Scripts\python -m labia.cli novo --nome minha-experiencia --dados meus-textos --preset rapido
```

Presets (`--listar-presets` mostra todos):

| preset | modelo | para quê |
|---|---|---|
| `micro` | dim 128 · 4 cam · janela 128 · 400 passos | teste de fumaça: valida a config em minutos |
| `rapido` | dim 192 · 4 cam · janela 192 · 1000 passos | primeiro treino de verdade |
| `equilibrado` | dim 256 · 6 cam · janela 256 · 2500 passos | mesma receita do run g1-treino-zero |
| `longo` | dim 384 · 6 cam · janela 320 · 6000 passos | modelo maior, corpus maior |
| `moe` | dim 256 · 6 cam + 4 especialistas top-1 | FFN esparsamente ativado |

Qualquer campo pode ser sobrescrito na linha de comando: `--dim --camadas --cabecas
--janela --abandono --especialistas --top-k --passos --lote --stride --lr --minimo-lr
--warmup --vocab-bpe --semente --dispositivo`.

O relatório antes do treino:

```
tamanho : 2.600.064 parâmetros (2.600.064 ativos/token)
treino  : 1000 passos x lote 24 x janela 192 = 4.608.000 tokens
tempo   : ~14.1 s · 0.0141 s/passo · 326.898 tokens/s
          ajuste de 2 termos sobre 3 medições reais (custo fixo + FLOPs)
          80.0 GFLOP por passo
          4608 tokens por passo (24 x 192)
VRAM    : ~193.8 MB (otimizador 39.7 MB · ativações 118.1 MB · logits 36.0 MB)
dados   : ~222.493 tokens → 2.316 janelas → 10.42 épocas
NOTA    : 66% do tempo por passo é custo fixo (Python/otimizador/kernel), não conta de GPU
```

Como ler isso:

- **parâmetros ativos** é o custo real por token. Em MoE, ativos < totais: 4
  especialistas com top-1 gastam 1/4 da FFN por token.
- **GFLOP por passo** é a conta de GPU daquele passo: `6·N·tokens + atenção`.
- **tempo por passo** é o número que decide seu dia. Ele NÃO é proporcional aos
  FLOPs em modelo pequeno: medido nesta máquina, 66% do tempo é custo fixo por
  passo (Python, AdamW, lançamento de kernel). Consequência prática: com modelo
  pequeno, aumentar `--lote` ou `--janela` rende mais por hora do que
  qualquer outra coisa — a GPU está ociosa esperando o processador.
- **épocas** é quantas vezes o corpus inteiro passa pelo modelo. Acima de ~60, o
  risco de decorar (overfit) cresce rápido: o modelo aprende o texto, não o idioma.
- **VRAM** é ordem de grandeza, separada em otimizador (16 bytes por parâmetro:
  peso, gradiente e os dois momentos do AdamW), ativações e logits.

O YAML gerado sai comentado campo a campo, para você editar sabendo o que mexe:

```yaml
lr: 0.0003                      # learning rate de pico
minimo_lr: 3.0e-05              # piso do decaimento cosseno
stride: 96                      # passo entre janelas; menor que janela_ctx = janelas deslizantes
```

---

## 3. Treinar

```bat
.venv\Scripts\python -m labia.cli train --config configs/minha-experiencia.yaml
```

Acompanhe `runs/minha-experiencia/metricas.jsonl` (uma linha JSON por avaliação):
`loss_trem`, `loss_val`, `lr`, `tokens_por_s`, `tempo_s`. Se a queda
acontecer (Ctrl+C, queda de energia, kill), retome do último checkpoint:

```bat
.venv\Scripts\python -m labia.cli train --config configs/minha-experiencia.yaml --retomar
```

Gerar texto do último checkpoint:

```bat
.venv\Scripts\python -m labia.cli gerar --run minha-experiencia --prompt "Uma noite destas"
```

---

## 4. Comparar e decidir

Terminado o treino (ou dois, ou dez), a pergunta é sempre a mesma: **o que aconteceu
e o que eu mudo agora?**

```bat
.venv\Scripts\python -m labia.cli comparar
.venv\Scripts\python -m labia.cli comparar g1-treino-zero livros-rapido --relatorio rel.md
```

Sem argumentos, compara todos os runs com métricas. Saída real deste repositório
(recortada: são os runs do próprio lab mais os dois da bancada):

```
run                  família     passos     params melhor val   passo    final   deriva  val@comum     tok/s veredito
-------------------------------------------------------------------------------------------------------------------------
g1-treino-zero       treino        2500          ?     4.6731    1500   4.7047  +0.0316     5.0610   282.045 deteriorando
g2-ajuste-ciencia    lora          1500          ?     0.3998    1500   0.3998  +0.0000     0.5286   129.176 ainda_caindo
g3-g1-int8           quantizacao       ?          ?     8.7290       0   8.7322  +0.0032          ?         ? quantizacao
g3-g1-nf4            quantizacao       ?          ?     8.7290       0   8.8811  +0.1521          ?         ? quantizacao
g4-moe-top2          treino        2500          ?     4.6996    1250   4.9687  +0.2691     5.0397    70.306 overfit
g4-moe-zero          treino        2500          ?     4.7025    1500   4.7358  +0.0333     5.1304    72.832 deteriorando
g5-ajuste-cot        lora          4000          ?     0.3054    1500   0.3990  +0.0936     0.3181   135.685 deteriorando
livros-equilibrado   treino        2500  5.847.040     4.6259    1500   4.6736  +0.0477     5.0256   294.180 deteriorando
livros-rapido        treino        1000          ?     5.0733    1000   5.0733  +0.0000     5.1899   316.377 ainda_caindo
valida-custo         treino         600          ?     4.8006     600   4.8006  +0.0000     4.8006   671.498 ainda_caindo
```

O `?` em *params* é de run antigo, gravado antes de o treino passar a registrar a
arquitetura no `estado.json` — descobrir agora exigiria carregar checkpoint de até
175 MB. Runs novos (como `livros-equilibrado`) já saem com o número.

O que cada coluna responde:

- **melhor val / passo**: o ponto de melhor generalização e onde ele está. É o
  checkpoint que você quer usar, não necessariamente o último.
- **final**: onde o treino terminou.
- **deriva** = final − melhor. Positiva e grande significa que o modelo piorou
  depois do melhor ponto: passou a decorar. A barra é 0,2 (a mesma da CA1b da G1).
- **val@comum**: melhor validação dentro do menor orçamento de passos entre os runs
  comparados. Sem isso, "treinou 2,5× mais passos" parece mérito, e não é.
- **veredito**: um rótulo por run, com um comentário que diz o que fazer.

O veredito agregado também avisa quando a lista mistura famílias:

> veredito: g5-ajuste-cot generaliza melhor: val 0.3054 contra 0.3998 de
> g2-ajuste-ciencia (diferença de 0.0944). ATENÇÃO: há famílias diferentes na lista
> (lora, treino) — perda de ajuste LoRA não é comparável com perda de treino do
> zero; compare dentro da mesma família.

### Ver as curvas

```bat
.venv\Scripts\python -m labia.cli curva g1-treino-zero g4-moe-top2 --saida curvas.svg
```

Gera quatro painéis (perda de validação, perda de treino, learning rate, tokens/s)
com um traço por run. SVG sai sem instalar nada; PNG precisa de matplotlib
(`.venv\Scripts\pip install matplotlib`).

A curva é o que separa diagnóstico de palpite: dois runs com a mesma perda final
podem ter histórias opostas — um parou no ponto certo, o outro decorou e voltou.

---

## 5 · Varrer hiperparâmetros

Comparar runs responde "qual ficou melhor". A varredura responde "**por quê**" — e é o
passo que transforma ajuste em medição.

```bat
.venv\Scripts\python -m labia.cli varrer --base configs/livros-rapido.yaml ^
  --grade lr=0.00015,0.0003,0.0006 --grade lote=16,32 --passos 500 --prefixo sw-lr
```

Antes de gastar GPU, confira a grade:

```bat
.venv\Scripts\python -m labia.cli varrer --base configs/livros-rapido.yaml --grade dim=64,100 --seco
```

```
run                       dim  melhor val   passo veredito
----------------------------------------------------------------
varrer-base-01             64           —       — seco
varrer-base-02            100           —       — seco
    erro: dim (100) precisa ser divisível por cabecas (6)
```

Saída real de uma varredura na GPU (6 variantes, 500 passos cada):

```
run                      lote         lr  melhor val   passo veredito
---------------------------------------------------------------------------
sw-lr-01                   16    0.00015      5.7858     500 ainda_caindo
sw-lr-02                   16     0.0003      5.5265     500 ainda_caindo
sw-lr-03                   16     0.0006      5.3010     500 ainda_caindo
sw-lr-04                   32    0.00015      5.6813     500 ainda_caindo
sw-lr-05                   32     0.0003      5.4078     500 ainda_caindo
sw-lr-06                   32     0.0006      5.1310     500 ainda_caindo

melhor: sw-lr-06 — val 5.1310 com lote=32, lr=0.0006

efeito de cada chave (média da melhor validação por valor):
  lr: 0.00015 → 5.7336 · 0.0003 → 5.4672 (base) · 0.0006 → 5.2160
    → melhor valor testado: 0.0006
  lote: 16 → 5.5378 · 32 → 5.4067
    → melhor valor testado: 32
```

Como ler:

- **mesmo orçamento**: todas as variantes treinaram 500 passos. Sem isso, "treinou mais"
  se disfarça de "treinou melhor".
- **efeito por chave** é a parte que ensina: aqui as duas chaves têm tendência monótona
  (lr maior ajudou, lote maior ajudou) e o melhor resultado é a combinação das duas.
- **tudo "ainda caindo"** significa que o orçamento foi curto: o achado é a *direção*,
  não o valor final. O uso correto é gastar o orçamento grande na região promissora.
- cada variante é um run normal: aparece no Painel, no `comparar` e nas curvas, e a
  config dela fica em `configs/sw-lr-06.yaml` para você repetir à mão.

Outras opções: `--modo aleatorio --n 8` (sorteia combinações, útil quando a grade é
grande), `--relatorio x.md`, `--json`, `--prefixo` e `--raiz`.

---

## 6. Erros comuns e o que significam

| Mensagem | Causa | Conserto |
|---|---|---|
| `campo 'minimo_lr' veio como texto` | YAML 1.1 não reconhece `3e-05` como número (falta ponto) | escreva `3.0e-05` (o gerador já faz) |
| `dim precisa ser divisível por cabecas` | atenção divide a largura entre as cabeças | ajuste `--dim` ou `--cabecas` |
| `warmup precisa ser menor que passos` | aquecimento maior que o treino inteiro | reduza `--warmup` |
| `corpus pequeno ou repetitivo demais` | dedup/limpeza deixou menos de 2 parágrafos | baixe `--min-chars` ou junte mais fontes |
| `config já existe` | proteção contra sobrescrever experimento | `--forcar` |
| `chaves desconhecidas na config` | erro de digitação no YAML | o treino recusa em vez de ignorar em silêncio |
| `matplotlib não está instalado` | pedido de PNG sem a lib | use `--saida curvas.svg` ou instale matplotlib |

---

## Pela janela (o laboratório portátil)

Tudo acima também funciona sem terminal: abra o app e use as abas.

- **Bancada** — mostra o estado do núcleo (GPU, portas, contagens) e o ciclo em dois
  passos: escreva os caminhos dos seus textos e o id do dataset, clique em
  *Preparar dataset*; escolha o dataset, o nome e o preset, clique em *Gerar config*.
  O resumo com parâmetros, tempo e VRAM aparece na tela, com o botão *Treinar agora*.
  No app empacotado o núcleo sobe junto com a janela; enquanto ele não responde, a
  página escreve *núcleo subindo…* e tenta de novo a cada 2 segundos.
- **Trilha** — as lições de estudo, com o texto da lição e o botão *Rodar experimento*:
  o laboratório portátil leva o material didático junto e mostra se as lições continuam
  citando código que existe.
- **Comparar** — marque os runs e clique em *Comparar*: tabela com melhor val,
  deriva, val@comum e veredito por run, o diagnóstico escrito de cada um e as curvas
  desenhadas na própria página (o SVG vem do núcleo, sem biblioteca gráfica).

---

## O que ainda falta (próximo passo)

- **Varredura de hiperparâmetros** (`lab-ia varrer`): rodar uma grade de configs e
  resumir qual mexeu no quê.
- **Trilha de estudo** com teoria, exercícios e notebooks — a parte de *entender*
  o que os números estão dizendo.
- **Empacotamento portátil**: o app já sobe o próprio núcleo; falta decidir o sabor
  do pacote (CUDA de 4,84 GB, CPU de ~500 MB, ou portátil de arquivo único). Os
  números medidos estão em `specs/B4.md`.