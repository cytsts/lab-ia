# Estado do projeto — avaliação de 13/09/2026

Avaliação honesta do que existe, do que falta e do que ainda precisa de decisão sua.
**Todo número aqui foi medido nesta máquina**, não estimado. Onde não foi verificado,
está dito que não foi.

---

## 1. Verificação em um comando

```bat
.venv\Scripts\python -m pytest -q -m "not slow"    :: suíte do núcleo
pnpm --dir ui test                                  :: suíte da interface
lab-ia trilha --conferir                            :: lições citam código existente?
.venv\Scripts\python scripts\gera_notebooks.py --checar
```

## 2. O que existe e está verde

| área | o que é | estado |
|---|---|---|
| **Núcleo do zero (G1–G10)** | GPT decoder-only, BPE, treino retomável, LoRA/QLoRA, quantização int8/NF4, MoE, CoT/ToT, agentes — 100% PyTorch escrito aqui | pronto e aprovado |
| **Bancada (B1–B5)** | trazer corpus próprio, config comentada com estimativas, comparar runs com diagnóstico, varredura de hiperparâmetros, curvas SVG/PNG | pronto |
| **Trilha (B6)** | 10 lições em pt-BR mapeadas ao código, 10 experimentos executáveis, 10 notebooks gerados, aba na janela | pronto |
| **Janela** | Electron + React + DS próprio, 7 abas, ~20 endpoints de API, **sobe o núcleo sozinha** | pronto |
| **Arquiteturas modernas (B7)** | RMSNorm, RoPE, SwiGLU, GQA integrados; preset `moderna`; contagem de parâmetros ciente da arquitetura | A.1–A.3 prontos |
| **Lógica proposicional (B12)** | gerador de datasets com resposta verificada por tabela-verdade (5 famílias, 3 splits, botão de dificuldade) | gerador + CLI + testes prontos |

Suíte: **273 testes pytest** (1 pulado: matplotlib) e **54 testes de UI**, verdes.
Runs treinados: 16. Datasets: corpus pt-BR + `logica-pq`.

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

## 5. O que falta

| frente | item | depende de |
|---|---|---|
| B7 | A.4 benchmark na 3070, A.5 lições 11–13, A.6 seletor na janela | nada |
| B12 | CLI de treino pela interface, visualizador de dataset, painel de teste do modelo | nada |
| B8 | modelos pré-treinados, llama.cpp, GGUF, serving | **rede** |
| B9 | currículo completo (matemática, ML clássico, estatística, método) | nada |
| B10 | múltiplas sementes, incerteza, caderno de hipóteses | nada |
| B11 | DS em camadas, acessibilidade, pacote instalável | **sua decisão** |

## 6. Limites do ambiente (medidos)

| limite | consequência |
|---|---|
| **Sem rede** (PyPI falha por SSL) | llama.cpp, pesos pré-treinados, jupyter e NSIS ficam parados |
| Sem jupyter | notebooks verificados por estrutura, não executados |
| Sem llama.cpp | o Épico B começa por uma instalação; hoje não há LLM local |
| VRAM 8 GB (RTX 3070) | nada acima de ~6 GB de pico |
| Disco ~89 GB livres | pacote CUDA ocupa 4,84 GB |
| Gauntlet Hub ausente | execução em modo direto, lacuna registrada |

## 7. Decisões que só você pode tomar

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
