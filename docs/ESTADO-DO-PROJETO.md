# Estado do projeto — direção atualizada em 15/09/2026

**O núcleo técnico está avançado e agora há uma primeira plataforma de experimentação
livre baseada em JupyterLab.** O objetivo é criar seus próprios testes e experimentos,
fora dos roteiros fechados. Veja [B14](../specs/B14.md) e o
[plano vigente](../specs/PLANO.md).

O núcleo agora abre JupyterLab local, com cadernos autorais em `cadernos/`, sessão
Python persistente, importação direta de `labia` e checkpoints explícitos. A interface
antiga de formulários continua auxiliar. O aceite de um experimento escolhido pelo
usuário e a inclusão no pacote portátil ainda estão pendentes.

## Verificação de 15/09/2026 e limites da avaliação

- Build da UI e 60 testes de UI passaram. Os testes usam dublês em fluxos relevantes
  e não demonstram integração completa nem liberdade de experimentação.
- Chamadas ao backend reproduziram HTTP 400 para `janela_ctx` (treino do zero)
  e `lora_r` (refinamento) na geração de configuração de B13.
- Inspeção do código de B13: a base selecionada não segue no pedido de configuração;
  a edição de YAML fica no estado da tela; o botão Retomar não envia `--retomar`.
- Catálogo da API: 18 runs e nenhum GGUF; `llama_cpp` carregou nesta máquina.
- Referências das lições conferidas; 10 notebooks em dia. Isso não equivale a
  executar os notebooks em uma plataforma interativa.
- A suíte Python foi iniciada durante a avaliação; seu resultado final não estava
  disponível ao encerrá-la. Os totais históricos abaixo não são uma nova medição.

As medições de desempenho abaixo são registros anteriores. Não foram refeitas nesta
revisão. B14 L0–L3 foram implementadas e validadas; L4–L5 permanecem abertas.

## 1. Verificações de engenharia

```bat
.venv\Scripts\python -m pytest -q -m "not slow"    :: suíte do núcleo
pnpm --dir ui test                                  :: suíte da interface
lab-ia trilha --conferir                            :: lições citam código existente?
.venv\Scripts\python scripts\gera_notebooks.py --checar
```

## 2. Capacidades existentes e situação de entrega

| área | o que é | estado |
|---|---|---|
| **Núcleo do zero (G1–G10)** | GPT decoder-only, BPE, treino retomável, LoRA/QLoRA, quantização int8/NF4, MoE, CoT/ToT, agentes — 100% PyTorch escrito aqui | pronto e aprovado |
| **Bancada (B1–B5)** | trazer corpus próprio, config comentada com estimativas, comparar runs com diagnóstico, varredura de hiperparâmetros, curvas SVG/PNG | pronto |
| **Trilha (B6)** | 10 lições em pt-BR mapeadas ao código, 10 experimentos executáveis, 10 notebooks gerados, aba na janela | pronto |
| **Janela** | Electron + React + DS próprio, API local e inicialização do núcleo | existente; fluxos predefinidos, com pendências de integração |
| **Arquiteturas modernas (B7)** | RMSNorm, RoPE, SwiGLU, GQA integrados; preset `moderna`; contagem de parâmetros ciente da arquitetura | A.1–A.3 prontos |
| **Lógica proposicional (B12)** | gerador de datasets com resposta verificada por tabela-verdade (5 famílias, 3 splits, botão de dificuldade) | gerador + CLI + testes prontos |
| **Interface de Treinamento v2 (B13)** | área Laboratório com 6 sub-abas (Dados · Configurar · Executar · Curvas · Testar · Medir), células de caderno, log ao vivo, exploração CoT teacher forcing, benchmark progressivo, catálogo unificado runs+GGUF | parcial; configuração, edição de YAML e retomada têm bloqueios; não equivale a notebook interativo |

Registro histórico de 13/09 (não usar como total atual): **282 testes pytest** (9 novos de B13, 1 pulado: matplotlib) e **60 testes de UI** (6 novos de B13), verdes.
Catálogo consultado em 15/09: 18 runs; datasets `livros-ptbr` e `logica-pq`.

## 3. Execuções reais registradas (não simulações)

| medição | número |
|---|---|
| Treino `livros-rapido` na 3070 | 14,48 s de laço, val 8,33 → 5,07 |
| Treino `livros-equilibrado` | 69,45 s, val 4,6259 |
| Varredura de 6 variantes × 500 passos | melhor `sw-lr-06` (lote 32, lr 0,0006) val 5,1310 |
| GPT-2 vs moderna (dim 256, 6 camadas) | 5.847.040 vs **5.205.248** parâmetros (~11% menos com a mesma largura) |
| Aritmética do laboratório (G5) | direta 2,6% · CoT 7,9% · ToT 7,9% em 152 itens |

## 4. Correções registradas (trilha de auditoria)

- **MoE:** a perda auxiliar não tinha gradiente — o `coef_auxiliar` não fazia nada.
  Corrigido; evidência: colapso de 94,5% → **26,6%** com a auxiliar ativa, contra 81,6% sem.
- **Retomada:** morrer entre gravar o último checkpoint e fechar o estado deixava o run
  "não concluído" para sempre. Corrigido com teste determinístico.
- **Estimador de tempo:** só FLOPs errava 2×; virou modelo de dois termos (erro médio 4,8%).
- **Especialistas do MoE ignoravam a opção `mlp`** — corrigido (MoE com SwiGLU agora é possível).
- **Orçamento de geração do CoT cortava a cadeia no meio** (`n_max=64`): as famílias de
  lógica que precisam de uma tabela-verdade inteira nunca escreviam "Resposta:" e
  marcavam 0%. Com parada antecipada e orçamento de 256 tokens, a mesma medição subiu de
  **38,3% para 80,0%** no split de teste e de 31,7% para 71,7% no difícil. É a mesma família
  de erro da perda auxiliar do MoE: artefato de medição disfarçado de falha do modelo.

## 5. O que falta

| frente | item | depende de |
|---|---|---|
| **B14 — prioridade** | cadernos livres + sessão Python + núcleo como biblioteca + aceite do experimento autoral | L0: avaliar base open source |
| B7 | A.4 benchmark na 3070, A.5 lições 11–13, A.6 seletor na janela | nada |
| B13 | resolver bloqueios dos fluxos existentes conforme necessário à transição; papel auxiliar | direção B14 |
| B8 | modelos pré-treinados, GGUF e serving; biblioteca llama.cpp já carrega | backlog; verificar rede e escolher pesos antes de baixar |
| B9 | currículo completo (matemática, ML clássico, estatística, método) | nada |
| B10 | múltiplas sementes, incerteza, caderno de hipóteses | nada |
| B11 | DS em camadas, acessibilidade, pacote instalável | **sua decisão** |

## 6. Ambiente — registros históricos a revalidar em L0

| limite | consequência |
|---|---|
| Falha histórica de SSL no PyPI | revalidar acesso na prova de conceito; não presumir bloqueio atual |
| Jupyter não validado nesta revisão | verificar instalação e execução de cadernos em L0 |
| llama.cpp carrega; catálogo sem GGUF em 15/09 | instalação da biblioteca não equivale a modelo externo integrado |
| VRAM 8 GB (RTX 3070) | nada acima de ~6 GB de pico |
| Disco ~89 GB livres | pacote CUDA ocupa 4,84 GB |
| Gauntlet Hub ausente | execução em modo direto, lacuna registrada |

## 7. Decisões para etapas posteriores

Estas decisões não bloqueiam L0 nem o início da experiência interativa. A abertura
a soluções open source já foi confirmada; a escolha da base será justificada na
prova de conceito.

1. **Sabor do pacote portátil**: CUDA 4,84 GB · CPU ~500 MB · arquivo único.
2. **Pesos pré-treinados**: quais licenças/origens você aceita baixar.
3. **Ritmo do currículo**: quanto tempo por semana, para dimensionar o Épico C.

## 8. Como usar hoje (5 minutos)

```bat
:: 1. seus dados (ou use o corpus do laboratório)
.venv\Scripts\python -m labia.cli dados --de data\corpus_ptbr.txt --id meus-textos

:: 2. config explicada, com estimativa de tempo e VRAM
.venv\Scripts\python -m labia.cli novo --nome meu-run --dados meus-textos --preset rapido

:: 3. treinar (22 s de relógio medidos, 14 s de laço)
.venv\Scripts\python -m labia.cli train --config configs/meu-run.yaml

:: 4. ver o que aconteceu
.venv\Scripts\python -m labia.cli comparar
.venv\Scripts\python -m labia.cli gerar --run meu-run --prompt "Uma noite"
```

Pela janela: `pnpm --dir ui electron` → aba **Bancada** (dados → config → treinar),
**Comparar** (diagnóstico e curvas), **Trilha** (as lições, com botão de rodar experimento).

## 9. Onde está o quê

| caminho | papel |
|---|---|
| `core/labia/` | núcleo: modelos, treino, ajuste, quantização, lógica, trilha, bancada, API |
| `docs/bancada.md` | guia de uso do laboratório |
| `docs/COMO-ESCREVER-TESTES.md` | como escrever teste nesta suíte |
| `trilha/` | lições, experimentos e notebooks |
| `specs/` | uma spec por meta + `PLANO.md` (estado vivo) + `PROMPT-PLATAFORMA.md` |
| `ui/` | Electron + React + Design System |
| `runs/` | experimentos executados (não versionado) |
