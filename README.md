# Lab-IA

Laboratório de IA em português brasileiro: treinamento do zero, fine-tuning
(LoRA/QLoRA), quantização, MoE e modelos de raciocínio — com retomada após queda
e interface desktop. Projeto guiado por specs (`specs/`) e pelo loop Gauntlet
(builder constrói, critic valida contra a barra).

## Requisitos

- Windows/Linux/macOS com **Python 3.13+** e **Node 20+** (recomendado 24) + **pnpm**
- GPU NVIDIA (CUDA 12.x) opcional — sem GPU tudo roda em CPU (mais lento)
- ~5 GB livres para ambiente + modelos de teste

## Instalação (do zero)

```bat
:: 1. Ambiente Python
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e "core[dev,finetune]"
:: (opcional, quantização GPU) .venv\Scripts\pip install bitsandbytes

:: 2. Corpus pt-BR (domínio público; já versionado em data/, re-executável)
.venv\Scripts\python scripts\prepara_corpus.py

:: 3. Interface (ver G6)
pnpm install --dir ui
```

## Verificação

```bat
.venv\Scripts\python -m pytest        :: suíte completa (não precisa de GPU)
```

## Uso rápido (G1 — treinamento do zero)

```bat
:: Treinar nano-GPT pt-BR (~15M parâmetros, 2500 passos)
.venv\Scripts\python -m labia.cli train --config configs/g1_treino_zero.yaml

:: Retomar após queda de energia (usa último checkpoint)
.venv\Scripts\python -m labia.cli train --config configs/g1_treino_zero.yaml --retomar

:: Gerar texto do último checkpoint
.venv\Scripts\python -m labia.cli gerar --run g1-treino-zero --prompt "Escritor de memória"

:: API interna (para a camada visual)
.venv\Scripts\python -m labia.cli servir --porta 8765
```

## Uso rápido (G2 — fine-tuning / G3 — quantização)

```bat
:: LoRA sobre a base treinada (tarefa: estilo técnico-científico)
.venv\Scripts\python scripts\prepara_tarefa_ciencia.py
.venv\Scripts\python -m labia.cli ajustar --config configs/g2_ajuste_ciencia.yaml
:: QLoRA: mude "tipo: qlora" (e bits: 8 ou 4) na config

:: Gerar com o adaptador (reconstrói base+LoRA automaticamente)
.venv\Scripts\python -m labia.cli gerar --run g2-ajuste-ciencia --prompt "A amostra foi"

:: Quantizar a base (int8 ou NF4) e medir tamanho/perda
.venv\Scripts\python -m labia.cli quantizar --run g1-treino-zero --saida g3-g1-int8 --modo int8
.venv\Scripts\python -m labia.cli quantizar --run g1-treino-zero --saida g3-g1-nf4  --modo nf4
.venv\Scripts\python -m labia.cli gerar --run g3-g1-nf4 --prompt "Uma noite destas"
```

Artefatos de um experimento ficam em `runs/<run-id>/`: `estado.json` (progresso),
`metricas.jsonl` (append-only), `ckpt/` (checkpoints atômicos), `tokens/` (BPE).
Log global append-only em `.lab-ia/eventos.jsonl`.

## Estrutura

| Caminho | Papel |
|---|---|
| `core/labia/` | Núcleo Python: modelos, treino, tokens, runner de experimentos, API |
| `ui/` | Camada visual TypeScript (Electron + Design System) — G6 |
| `specs/` | Uma spec por meta (G1…G10); `PLANO.md` = estado vivo do loop |
| `tests/` | Testes derivados das specs (rodam antes da implementação existir) |
| `configs/` | Configurações YAML de experimentos |
| `data/` | Corpus pt-BR versionado (domínio público) |

## Status das metas

Ver `specs/PLANO.md` — cada meta só muda de status com evidência re-executável
e veredito do Critic.
