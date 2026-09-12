# Lição 10 — agentes: o modelo escolhendo o que executar

**Pré-requisito:** Lições 1 a 9. **Tempo:** ~30 min.

## O que você vai entender

O que é (e o que não é) um "agente" neste laboratório, por que a assinatura da skill
importa mais que o nome bonito, e por que o log é a parte séria do assunto.

## A ideia

Um agente é, na prática, três coisas:

1. um **conjunto de funções** com nome, descrição e assinatura declarada (as *skills*);
2. um **plano** que decide qual chamar com quais argumentos;
3. um **registro** de tudo que foi tentado, inclusive o que falhou.

O terceiro item é o que separa agente de script. Um script que funciona em silêncio
não te diz nada quando para de funcionar; um agente auditável deixa rastro.

Neste laboratório os agentes **não** usam um LLM para decidir: a escolha é feita por
você (ou pela CLI). Isso é proposital — o mecanismo fica visível e testável, e o dia
em que um modelo escolher as skills, o que muda é o plano, não a auditoria.

## No núcleo

**No núcleo:** `core/labia/agents/base.py` → `AgenteBase.executar`

```python
def executar(self, skill: str, **argumentos) -> dict:
    if skill not in self._skills:
        self.eventos.registrar("agente_acao", agente=self.nome, skill=skill, ok=False,
                               erro="skill desconhecida")
        raise ValueError(...)
    ...
    self.eventos.registrar("agente_acao", agente=self.nome, skill=skill, ok=True,
                           duracao_s=round(time.time() - inicio, 2), resumo={...})
```

Note a forma do registro: **sucesso e falha viram a mesma linha de log**, com o campo
`ok` dizendo qual dos dois. E só os campos escalares do resultado entram em `resumo` —
o log é um resumo auditável, não um despejo.

Os três agentes do laboratório:

| agente | papel | skills |
|---|---|---|
| **treinador** | conduz treinamento, ajuste e quantização | `train_model` · `fine_tune` · `quantize` |
| **avaliador** | mede, gera amostra, compara runs | `evaluate_model` · `benchmark` · `compare_runs` |
| **arquiteto** | propõe arquitetura para o hardware/corpus | `suggest_architecture` · `optimize_moe` |

## Medindo

```bat
.venv\Scripts\python trilha\experimentos\e10_agentes.py
```

```
1) inventário dos agentes
   arquiteto   — propõe arquiteturas viáveis para o hardware e o corpus disponíveis
      · suggest_architecture(corpus_bytes: int, vram_gb: float = 8.0, alvo_epocas: float = 2.0) -> dict
      · optimize_moe(run: str) -> dict
   avaliador   — avalia modelos, roda benchmarks e compara corridas
      · evaluate_model(run: str, prompt: str = 'Uma noite', passos_max: int = 24) -> dict
      · benchmark(run: str, estrategia: str = 'cot', limite: int | None = None) -> dict
      · compare_runs(run_a: str, run_b: str) -> dict
   treinador   — conduz experimentos de treinamento, fine-tuning e quantização
      · train_model(config: str, run_id: str | None = None, retomar: bool = False) -> dict
      · fine_tune(config: str, run_id: str | None = None, retomar: bool = False) -> dict
      · quantize(run: str, saida: str, modo: str = 'int8', corpus: str = 'data/tarefa_ciencia.txt') -> dict
   total: 3 agentes, 8 skills

2) skill executada com sucesso (arquiteto.suggest_architecture)
   resultado: {'vocab_bpe': 4096, 'passos': 500, 'lote': 8, 'parametros_estimados': 15852288,
   'orcamento_params': 286331153, 'justificativa': 'maior modelo que cabe em 4 GB ...'}
   log: {"tipo": "agente_acao", "agente": "arquiteto", "skill": "suggest_architecture", "ok": true, ...}

3) skill desconhecida: falha alto E fica registrada
   levantou erro? True — treinador não conhece a skill 'skill_que_nao_existe'
   log: {"tipo": "agente_acao", "agente": "treinador", "skill": "skill_que_nao_existe", "ok": false,
         "erro": "skill desconhecida"}
```

- **item 1**: a assinatura é a interface. `quantize(run, saida, modo, corpus) -> dict` diz
  o que entra e o que sai — um "agente" sem assinatura é um prompt com esperança.
- **item 2**: o arquiteto devolveu um orçamento de ~286 M de parâmetros e um modelo de
  ~15,9 M, com a justificativa escrita. Repare que ele **não treina nada**: propõe.
- **item 3**: o item mais importante. A falha levanta exceção (ninguém segue em silêncio)
  **e** deixa linha no log. É isso que permite reconstruir depois o que aconteceu.

## O que isso muda na prática

- **Separe as skills por risco.** As deste laboratório só rodam runners do núcleo; nenhuma
  edita configuração do usuário (está no `escopo` do agente treinador). Um agente que
  edita arquivos precisa de outro nível de cuidado.
- **O log é a interface de depuração.** `.lab-ia/eventos.jsonl` tem uma linha por
  tentativa: agente, skill, ok, duração. Quando algo "não funciona", a resposta costuma
  estar ali, não no código.
- **Trocar o plano não muda a auditoria.** Se você plugar um LLM para escolher as skills,
  `AgenteBase.executar` continua registrando igual.

## Exercícios

1. **Leia o log de verdade.** Rode
   ```bat
   lab-ia agente arquiteto suggest_architecture --json "{\"corpus_bytes\": 2000000, \"vram_gb\": 8}"
   ```
   e depois abra `.lab-ia/eventos.jsonl`. *Esperado:* uma linha `agente_acao` com
   `ok=true`, o resumo escalar e a duração — o mesmo que o experimento mostrou, agora
   no log de verdade do laboratório.
2. **Custo por épocas.** Chame `suggest_architecture` com `alvo_epocas=8`. *Esperado:*
   a proposta muda (mais passos, mesmo orçamento de parâmetros) — o agente negocia uma
   restrição contra outra, não devolve um número fixo.
3. **Agente que treina.** Rode `lab-ia agente treinador train_model --json "{\"config\":
   \"configs/valida-custo.yaml\"}"` e observe o run aparecer no Painel. *Esperado:* a skill é
   um invólucro fino sobre `executar_treino` — o mesmo runner da CLI, com log.
4. **Desenhe um plano.** Escreva (em papel) a sequência de skills que você usaria para
   "melhorar um run que estagnou". *Esperado:* algo como `compare_runs` →
   `suggest_architecture` → `train_model` → `evaluate_model`. Esse é o
   plano que um agente precisaria executar — e nenhuma dessas skills existe sozinha por
   acaso.

## Se quiser ir mais fundo

- `specs/G9.md`: a meta dos agentes (e a nota de que a spec foi escrita *depois* do
  código — desvio de processo registrado).
- `core/labia/agents/avaliador.py`: o agente que amarra as Lições 5 e 9 (avaliação e
  benchmark) numa skill só.
