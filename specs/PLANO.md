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
| G4 | MoE | FEITO (Aprovada, spec v2) | 7/7 testes CPU; run real GPU 2500 passos: top-1 min val 4.703 (paridade c/ dense 4.673, Δ+0.030), uso [0.19–0.35] sem colapso, aux 1.048 (piso), drift 0.033, ativos = 1/4 dos totais; top-2 medido e documentado como achado (não melhora em corpus pequeno — CA7 v2) |
| G5 | Raciocínio (CoT/ToT + benchmark) | FEITO (Aprovada, spec v2) | 6/6 testes CPU; run real GPU: direta 2,6% · CoT 7,9% · ToT 7,9% (CoT−direta +5,3 p.p. = CA1 ✅; ToT−CoT 0,0 = CA2 ✅; taxa formato CoT 1,00 = CA3 v2 ✅). Achado documentado: aritmética com carry não generaliza em 15M (Passo 1 correto, soma errada) — harness pronto para escalar. Determinismo byte a byte ✅ |
| G6 | UI completa (Electron + DS) | FEITO (Aprovada) | vitest 7/7; tsc+vite build limpo; janela Electron REAL com dados do núcleo ao vivo (screenshot: 7 runs, badges corretos, progresso, tema escuro do sistema); DS catálogo vivo renderizando (screenshot); API RF5 /execucao+/configs+/tamanhos+/comparativo testada (53/53 pytest). Screenshot da janela em `.qwen/captura-app/ui-*.png` (fora do repo, reprodutível) |
| G7 | Empacotamento desktop | FEITO (Aprovada, spec v2) | **A "lacuna PyInstaller" era diagnóstico errado, não limite do ambiente.** CA1 ✅ `ui/release/Lab-IA 0.1.0.exe` portable (71,2 MiB = 74,7 MB). CA2 ✅ núcleo congelado executa: `.qwen/pyi-dist3/lab-ia/lab-ia.exe agentes` responde rc=0 e uma geração real do exe aparece como processo CUDA no `nvidia-smi` (PID 28732; `agente_acao ok=true`, `duracao_s=2.29`, `min=4.6731` = mínimo da G1). CA3 ✅ `lab-ia.spec` + `scripts/verifica_pacote.py` versionados. Causa medida: o PyInstaller puxava `msvcp140.dll` **14.16.27033 (2018) do JDK 11** e `vcruntime140.dll` do **Python 3.14 do sistema** para `_internal/`, que sombreia o CRT do sistema; `c10.dll` precisa de CRT mais nova → WinError 1114 (não é CUDA: as 37 DLLs de `torch/lib` estavam todas lá e byte-idênticas). Prova causal: T1/T2 0 falhas, T3 (raiz `_internal` no caminho) 1114, T4 (sem as 4 CRTs) 0 falhas; e no bundle bom, trocar um arquivo pela CRT do JDK ⇒ rc=1 com 1114, restaurar ⇒ rc=0 (sha `7c26614e1d733892` reconferido). Build 282 s. Verificação: `scripts/verifica_pacote.py` + `tests/g7/test_pacote.py` |
| G8 | Retomada pós-queda (kill -9) | FEITO (Aprovada) | 2/2 testes com KILL REAL de subprocesso (TerminateProcess, sem cleanup) em 2 janelas de passos diferentes: todo passo-*.pt visível carrega inteiro, estado.json íntegro com concluido=false, retomada produz métricas idênticas (<1e-6) ao contínuo e conclui ✅ |
| G9 | Agentes (3+ com skills) | FEITO (Aprovada) | 6/6 testes CPU; deviação de processo registrada: spec escrita DEPOIS de testes+código (contrato formalizado a posteriori em specs/G9.md). Evidência viva: `lab-ia agentes` → 3 agentes / 8 skills com assinaturas; log `agente_acao` ok=true/false auditado |
| G10 | Suite de testes ≥80% cobertura | FEITO (Aprovada) | 71 testes pytest (69 CPU-verdes + 2 `slow`), re-medido 2026-09-11 após a G7 v2: **69 passed, 2 deselected, 381s**, cobertura núcleo **90%** (cli incluída in-process) + 16 vitest (UI **96,7%** linhas); mutações documentadas (G1 otimizador, G3 códigos zerados, G7 CRT do JDK reintroduzida no bundle ⇒ 2 falhas); g8 anti-flaky (poll antes do assert de sinal) |

## Correções de veredito (trilha de auditoria — nada se apaga)
- 2026-09-11 · G7/CA2: o veredito anterior dizia *"`c10.dll WinError 1114` na
  inicialização CUDA congelada — rota oficial do núcleo é o venv"*. Falso: a
  causa era sombreamento de CRT (`msvcp140.dll` 14.16 do JDK 11 na raiz do
  bundle), e o exe funciona após re-pinchá-la. O texto antigo da CA2 ("OU a
  lacuna é documentada") admitia documentar em vez de corrigir — barra frouxa,
  agora CA2 exige verificador verde. Detalhado em `specs/G7.md` (Revisão v2).
- 2026-09-11 · README dizia "cobertura do núcleo 92%"; medido 90% no início
  desta sessão, antes de qualquer mudança (`pytest -m "not slow"
  --cov=core/labia`, 67 passed + 1 deselected). Alinhado ao PLANO.

## Fora de escopo declarado
- G7 por último: depende da UI (G6) estável.
- Modelos grandes (>1B): inviável em 8GB VRAM + 48GB disco; o lab demonstra as técnicas em escala pequena, reproduzível e rápida.
- CI GitHub Actions: opcional no prompt; entra só se sobrar tempo.
