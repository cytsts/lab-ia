# PLANO — Lab-IA (vivo, auditável)

Regra: item só muda de status com evidência re-executável (comando + saída) registrada em `.lab-ia/eventos.jsonl` ou neste arquivo.

## Ambiente (E0) — validado 2026-09-11
| Checagem | Resultado | Evidência |
|---|---|---|
| Python | 3.13 disponível (`py -3.13`) | `py -0p` |
| Node/pnpm | v24.12.0 / 10.32.1 | `node --version` |
| GPU | RTX 3070 8GB, driver 591.86 | `nvidia-smi` |
| Torch sistema | 2.11.0+cpu (cuda_ok False) | `python -c "import torch; ..."` |
| Disco K: | 48GB livres (⚠ <50GB) | `shutil.disk_usage('K:')` |
| Gauntlet Hub CLI/MCP | AUSENTE → modo direto, lacuna registrada | `gauntlet work status` |

## Atividades
| ID | Atividade | Status | Veredito do Critic |
|----|-----------|--------|--------------------|
| E0 | Validar ambiente | FEITO | — |
| S0 | Estrutura de diretórios + specs/ | FEITO | — |
| S1 | `.venv` (py3.13) + deps CUDA | FEITO | torch 2.11.0+cu128, cuda_ok True (medido) |
| S2 | Corpus pt-BR domínio público em `data/` | FEITO | 886KB (Dom Casmurro + O Cortiço), manifesto com licença; IDs verificados por busca, não suposição |
| G1 | Treinamento do zero (nano-GPT pt-BR) | FEITO (Aprovada) | v1 reprovada: overfit real (val 4.99→6.04). v2: CA1 reescrita + stride deslizante. Evidências: 21/21 testes; mutação detectada (0.0095 > 1e-6); treino real 72s CUDA @281k tok/s — min val 4.673 ≤ unigram 6.529−1.5 (CA1a ✅), drift final−min 0.032 ≤ 0.20 (CA1b ✅); gerar produz prosa pt-BR coerente (CA3 ✅) |
| G2 | LoRA/QLoRA | FEITO (Aprovada) | 15/15 testes CPU (incl. slow no run real); mutação do otimizador já detectada na G1 vale aqui (CA5 <1e-6). Run real GPU: perda na tarefa 8.84→0.40 (−95%) em 48s; CA2: sobreposição n-gramas 4 base 0.000 → ajustado 1.000; adaptador r=8 = 1,3% dos parâmetros; QLoRA int8 no micro: −22,5% com economia ≥2× por módulo |
| G3 | Quantização 4/8-bit | FEITO (Aprovada) | 4/4 testes CPU; run real G1: int8 fator 3.95× (+0.003 nats), nf4 fator 7.11× / 4.50 bits efetivos (+1.7% perda, ≤10%); gerar funcional nos dois; RF5: comparativo com bitsandbytes 0.50.2 registrado — nativo 0.00701 vs bnb 0.01312 (erro relativo) |
| G4 | MoE | AGUARDANDO | — |
| G5 | Raciocínio (CoT/ToT + benchmark) | AGUARDANDO | — |
| G6 | UI completa (Electron + DS) | AGUARDANDO | — |
| G7 | Empacotamento desktop | AGUARDANDO | — |
| G8 | Retomada pós-queda (kill -9) | AGUARDANDO | — |
| G9 | Agentes (3+ com skills) | AGUARDANDO | — |
| G10 | Suite de testes ≥80% cobertura | AGUARDANDO | — |

## Fora de escopo declarado
- G7 por último: depende da UI (G6) estável.
- Modelos grandes (>1B): inviável em 8GB VRAM + 48GB disco; o lab demonstra as técnicas em escala pequena, reproduzível e rápida.
- CI GitHub Actions: opcional no prompt; entra só se sobrar tempo.
