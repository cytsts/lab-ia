# Lição 9 — raciocínio: quando pedir o passo a passo ajuda

**Pré-requisito:** Lições 1 a 8. **Tempo:** ~35 min.

## O que você vai entender

O que "Chain of Thought" muda de fato (dica: não é mágica, é formato e tokens), por que
ToT não é automaticamente melhor, e como a métrica de um benchmark pode medir formato
em vez de raciocínio.

## A ideia

Três estratégias de decodificação, todas usando o mesmo modelo:

| estratégia | o que faz | custo |
|---|---|---|
| **direta** | pede a resposta e para | 1 geração curta |
| **CoT** | pede "Vamos pensar passo a passo", gera o caminho e depois a resposta | muito mais tokens |
| **ToT** | gera vários passos candidatos, pontua pelo log-prob do próprio modelo e expande o melhor k vezes | k × profundidade gerações |

CoT funciona porque o modelo ganha *passos intermediários* para calcular — cada token
gerado vira contexto para o próximo. É a mesma razão pela qual você resolve melhor uma
conta escrevendo no papel.

ToT troca mais computação por escolha: ele explora alternativas. Se o gargalo for a
*decisão*, ele ajuda; se for a *capacidade de calcular*, ele só gasta mais.

## No núcleo

**No núcleo:** `core/labia/reasoning/estrategias.py` → `responder_direta`

**No núcleo:** `core/labia/reasoning/estrategias.py` → `responder_cot`

**No núcleo:** `core/labia/reasoning/estrategias.py` → `responder_tot`

E o detalhe que decide tudo:

**No núcleo:** `core/labia/reasoning/estrategias.py` → `parse_resposta`

```python
_PARSE = re.compile(r"Resposta:\s*(-?\d+)")   # a métrica exige ESTE formato
```

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e09_raciocinio.py
```

```
1) o formato da resposta é parte da métrica
   formato correto      → 12
   com passo a passo    → 12
   texto livre          → None
   sem a palavra-chave  → None

2) quanto cada estratégia gasta (mesmo enunciado)
   prompt da resposta direta: 26 tokens
   prompt do CoT            : 47 tokens (a frase 'Vamos pensar passo a passo.')
   gerados pela direta      : 10 tokens
   gerados pelo CoT         : 64 tokens

3) benchmark deste laboratório: 152 itens de teste

4) resultado real do run g5-ajuste-cot (medido nesta máquina, spec G5)
   direta 0.026 · CoT 0.079 · ToT 0.079
   CoT − direta = +5.3 p.p. · ToT − CoT = +0.0 p.p.
```

- **item 1**: "doze" está certo e conta como **erro**; "12" sem a palavra-chave também. A
  métrica mede o formato tanto quanto a aritmética — e isso é uma escolha de projeto, não
  um detalhe. Parte do ganho do CoT é aprender o formato.
- **item 2**: o prompt do CoT já nasce com 21 tokens a mais, e a geração é 6,4× maior
  (64 contra 10, limitados pelo `n_max` de cada estratégia). Custo real, em tempo e
  contexto.
- **item 4**: o resultado do run real. CoT ganhou **5,3 p.p.** sobre a resposta direta; o
  ToT, que gasta muito mais, **não ganhou nada** além disso.

## O que isso muda na prática

O achado do laboratório (registrado no PLANO) é específico e vale entender o porquê: com
um modelo de ~15 M de parâmetros, a aritmética **com transporte** (7+5, 16+18) não
generaliza. O CoT ajuda porque dá passos, mas o modelo erra a soma de qualquer forma. O
ToT não ajuda porque o problema não é escolher entre caminhos — é calcular.

A lição transferível: **antes de comprar uma técnica mais cara, identifique onde está o
gargalo.** Se o modelo não sabe fazer a conta, mais busca não resolve; se ele sabe mas
hesita entre alternativas, busca resolve.

## Exercícios

1. **Preveja.** Rode `responder_cot` com `guloso=False` (amostragem) e compare.
   *Esperado:* a taxa de formato válido **cai** — amostragem erra o formato com mais
   frequência, mesmo quando acerta a conta.
2. **Aumente o orçamento.** Rode o benchmark completo com o modelo da G5:

   ```bat
   lab-ia raciocinio --run g5-ajuste-cot --comparar --limite 40
   ```

   *Esperado:* números próximos de 2,6% / 7,9% / 7,9% — e o aviso de que o ToT custa muito
   mais para o mesmo resultado.
3. **Mude a métrica.** Aceite também números escritos por extenso em `parse_resposta`.
   *Esperado:* a acurácia sobe nos três casos — o que confirma quanto do resultado era
   formato. (Depois desfaça: métrica frouxa engana.)

## Se quiser ir mais fundo

- `specs/G5.md`: a meta, o benchmark de 152 itens e o achado de escala documentado.
- `data/benchmark_matematica.jsonl`: os itens, com as famílias (soma, subtração…) que
  permitem ver *onde* o modelo erra.
