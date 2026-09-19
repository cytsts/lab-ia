# PLANO — Lab-IA (vivo, auditável)

Regra: item só muda de status com evidência re-executável (comando + saída) registrada em `.lab-ia/eventos.jsonl` ou neste arquivo.

## Direção vigente — 15/09/2026

**O produto é um laboratório para os experimentos do usuário, com cadernos
interativos e execução livre de Python.** Testes de engenharia e benchmarks
predefinidos apoiam a plataforma, mas não são sua entrega principal.

O usuário confirmou a experiência semelhante a Jupyter e a abertura a soluções
open source. JupyterLab foi escolhido e integrado na primeira entrega; não há
compromisso de construir editor, protocolo de execução ou kernel próprio.

Plano vigente: [B14 — Laboratório de experimentação livre](B14.md).

| Prioridade | Entrega | Estado |
|---|---|---|
| L0 | JupyterLab avaliado e integrado em execução local | Feito: 4.6.3, servidor local com token |
| L1 | Cadernos autorais, células Python/Markdown e sessão persistente entre células | Feito via JupyterLab; falta aceite manual |
| L2 | Dados, modelos, treino e métricas livres usando o núcleo como biblioteca | Feito no exemplo mínimo; ampliar exemplos em L4 |
| L3 | Checkpoints, resultados salvos e reprodução em kernel limpo | Feito no exemplo mínimo |
| L4 | Exemplos editáveis e aceite com um experimento escolhido pelo usuário | Parcial |
| L5 | Instalação/abertura e distribuição verificadas em ambiente limpo | Pendente |

**Próximo passo:** L4, abra a aba **Cadernos**, copie o exemplo e conduza um
experimento escolhido por você. Esse aceite prático determina os próximos exemplos
e correções; não será substituído por uma suíte verde.

B13 passa a ser implementação parcial e auxiliar: seus passos visuais não oferecem
um kernel interativo nem código livre. Os bloqueios de configuração registrados
abaixo continuam abertos. B7–B11 permanecem como backlog, subordinados à entrega do
ciclo livre; ampliar formulários, suíte de testes ou currículo não substitui B14.

As tabelas e medições abaixo são histórico das entregas anteriores. “FEITO” em
G1–G10 ou em uma bancada não significa plataforma B14 entregue. A nova direção
está registrada; L0–L3 foram implementadas em 15/09/2026. L4 e L5 continuam abertas.

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

## Bancada (uso próprio) — B1/B2
| ID | Atividade | Status | Veredito do Critic |
|----|-----------|--------|--------------------|
| B1 | Dados próprios + config assistida | FEITO | 64 testes CPU verdes (`pytest tests/b1`); ponta a ponta real na GPU: corpus de 866 KB → `data/livros-ptbr/` (3319 parágrafos) → `configs/livros-rapido.yaml` → run `livros-rapido` val 8,33→5,07 (unigram 6,55) em 14,48 s de laço CUDA. **Achado:** estimador só-FLOPs errou 2,0× (previu 7,2 s; real 14,5 s) porque com 2–6 M parâmetros o gargalo é o custo fixo por passo, não a GPU → modelo de 2 termos (9,3 ms fixos + FLOPs/16,7 TFLOP/s), erro máx. 6,8% e médio 4,8% em validação cruzada deixando-um-de-fora sobre 4 medições. **Dois bugs achados por execução:** `utf-8-sig` rotulava todo utf-8 comum (ordem da detecção); `minimo_lr: 3e-05` vira string no YAML 1.1 e quebrava o treino no passo 1000 — agora falha na leitura dizendo o conserto |
| B2 | Comparar runs, diagnosticar e desenhar curvas | FEITO | 37 testes CPU (1 pulado por ausência de matplotlib); evidência nos 9 runs reais: `g4-moe-top2` → **overfit** (deriva +0,2691), coerente com o achado do top-2 da G4; `g1-treino-zero` aponta mínimo no passo 1500 com val 4,6731, igual ao PLANO; `g3-g1-int8`/`g3-g1-nf4` → **quantizacao** com fator 3,95×/7,11×. **Dois falsos positivos corrigidos:** queda do passo 0 (modelo aleatório) marcada como instabilidade; orçamento comum zerado por runs de quantização. Curvas saem em SVG sem dependência nenhuma (PyPI inacessível neste ambiente) |

| B3 | Bancada na API e na janela | FEITO | 14 testes de API CPU-verdes (`tests/b3/`) + 44 testes de UI (`pnpm --dir ui test`) + `tsc --noEmit` e vite limpos. Endpoints `/saude`, `/datasets`, `POST /dados`, `/presets`, `POST /novo`, `/comparar`, `/comparar/relatorio`, `/comparar/curvas.svg`, `/guia/{nome}`, `/execucoes/{chave}/log`. Páginas **Bancada** (dados → config → treinar) e **Comparar** (tabela, diagnóstico, veredito, curvas). Verificado: a config gerada pela API é aceita por `ConfigTreino.de_arquivo` sem ajuste manual; curvas saem em SVG sem matplotlib; nenhuma rota aceita caminho arbitrário |
| B4 | Laboratório portátil: o app sobe o próprio núcleo | FEITO (empacotamento não executado) | Diagnóstico com evidência: `ui/package.json` levava só `dist + electron` (app.asar de 4,88 MB) e o `main.mjs` calculava a raiz como `join(aqui,'..','..')`, que dentro do asar não aponta para lugar nenhum — os dois artefatos da G7 nunca foram unidos. Agora: resolvedor testável em `ui/electron/nucleo.mjs` (19 testes em ambiente node, com dublês injetados), semente do workspace sem sobrescrever nada, espera de `/saude` distinguindo morte de timeout, encerramento junto com a janela, `scripts/monta_portatil.py` e `extraResources`. **Peso medido: 4,84 GB, dos quais 4,08 GB são torch CUDA** — é o número que decide o sabor do pacote (specs/B4.md). `pnpm --dir ui package` não rodou aqui: baixa NSIS/winCodeSign e não há rede neste ambiente |

| B5 | Varredura de hiperparâmetros | FEITO | 19 testes CPU-verdes (`tests/b5/`: 14 do módulo + 5 da API) e 49 testes de UI. **Varredura real na GPU**: base `livros-rapido`, grade `lr` ∈ {0,00015; 0,0003; 0,0006} × `lote` ∈ {16, 32}, 500 passos por variante (orçamento igual) → melhor `sw-lr-06` (lote 32, lr 0,0006) val **5,1310**; efeito medido: `lr` 0,00015→5,7336 · 0,0003→5,4672 · 0,0006→5,2160 e `lote` 16→5,5378 · 32→5,4067 — tendência monótona nas duas chaves. Leitura honesta: as seis saíram `ainda_caindo` (orçamento curto), então o achado é a **direção**, não o valor final. **Dois bugs achados rodando de verdade:** o diretório do run não era criado (as seis variantes morreram com "Parent directory ckpt does not exist" — os testes de unidade passavam porque usavam diretório pronto) e `n or 8` engolia `--n 0` em silêncio |

| B13 | Interface de Treinamento v2 (área Laboratório) | PARCIAL / AUXILIAR — bloqueios funcionais; direção sucedida por B14 em 2026-09-15 | **9 testes pytest** (`tests/b13/test_api_b13.py`, 9/9 verdes em 5,28 s) + **6 testes Vitest** (`ui/test/laboratorio.test.tsx`, 6/6 verdes) + **60/60 testes de UI** totais. Os testes escritos cobrem CA1, CA4–CA17; **a verificação independente de 2026-09-14 refutou a cobertura de CA2/CA3** — os dois modos da sub-aba Configurar falham contra o núcleo real — e achou o build da UI quebrado (detalhes no fim desta célula e no log). **Novas rotas API:** `GET /datasets/{id}`, `GET /datasets/{id}/itens` (paginado, filtros — CA1), `GET /corre/{run_id}/progresso` (RF3.3/RF3.4), `GET /corre/{run_id}/benchmarks`, `POST /execucoes/{chave}/parar` (CA8 — terminate+kill, concluido:false), `GET /execucoes/{chave}/log` (cauda reversa <200 ms — CA12), `POST /testar` (CoT+teacher forcing — CA6/CA15, grava em `exploracao.jsonl`), `POST /benchmark`, `GET /modelos` (catálogo unificado runs+GGUF — CA13). **Novos componentes UI:** `PassoCaderno` (célula de caderno c/ CLI copiável — CA17), `SubAbaDados` (manifesto + navegador itens — CA1), `SubAbaConfigurar` (modo zero/refinar, aba runs+GGUF c/ rótulo de honestidade — CA13, validação síncrona dim%cabeças — CA4), `SubAbaExecutar` (log ao vivo 2 s — RF3.2, parada/retomada — CA8), `SubAbaCurvas` (gráfico perda, diagnóstico, sparkline), `SubAbaTestar` (CoT/direta, teacher forcing — CA15, diff "Onde ele errou" — CA16), `SubAbaMedir` (benchmark c/ progresso parcial — CA14, acurácia por família, comparativo delta p.p.). Aba `Laboratório` adicionada ao `App.tsx`. `llama-cpp-python 0.3.35` instalado em-processo via wheel local (confirmado: `from llama_cpp import Llama` carrega; `modelos/` está vazio, então CA13 só é verificável com um GGUF de verdade). **Três defeitos medidos na conferência independente (2026-09-14):** (a) **o payload que a própria UI envia era rejeitado** por `POST /novo` — a whitelist `SOBRESCRITAS_*` não tem `janela_ctx` nem `arquitetura` (modo zero) nem `lora_r`/`lora_alpha`/`lora_tipo` (modo refinar, que é o **padrão** da aba): medido `STATUS=400 {"detail":"sobrescrita desconhecida: 'janela_ctx'"}` e `STATUS=400 {"detail":"sobrescrita desconhecida: 'lora_r'"}`. Os 60 testes de UI passavam porque **mockam** `api.novo` — nenhum toca a whitelist. (b) `pnpm --dir ui build` estava **quebrado** (`tsc` exit 2: `test/laboratorio.test.tsx(147,106)` usa `total`, campo que a rota `/execucoes/{chave}/log` devolve sempre, mas que o tipo de `logExecucao` em `ui/src/api.ts` não declarava). (c) `GET /modelos` devolvia `parametros: null` em 16 de 18 runs e `treinado_em_raciocinio: false` para **todos**, inclusive `logica-1`: lia `runs/<id>/config.yaml`, arquivo que nenhum run grava (medido: 0 ocorrências em `runs/`). **(b) e (c) corrigidos** nesta conferência; **(a) segue aberto**, e não é só whitelist: o config de ajuste canônico (`configs/g5_ajuste_cot.yaml`) usa outro schema (`base`, `corpus_tarefa`, `tipo`, `r`, `alpha`) e **nada no núcleo gera esse arquivo** — falta o gerador do modo refinar, não uma chave. |
| B6 | Trilha de estudo (fase 2) | FEITO (10 lições) | 23 testes CPU-verdes (`tests/trilha/`). **10 lições** em pt-BR mapeadas ao núcleo — parte 1 (criar): tensor/autograd · tokenização BPE · atenção causal · bloco e resíduo · laço de treino; parte 2 (otimizar): **LoRA · quantização · MoE · raciocínio · agentes** — com 10 experimentos executáveis e 10 notebooks gerados do próprio experimento. Cada afirmação tem teste com número real: autograd exato (erro 0,0); acúmulo `[3,6,9]` sem `zero_grad`; 4096 → 3,43 chars/token contra 0,98 com 128; causalidade exata (0,000e+00) e vazamento de 1,95 sem `is_causal`; crescimento 1,71x com escala `1/sqrt(2·camadas)` contra 5,40x sem ela; lr=1,0 a 92,3 de perda contra 6,39 com lr=3e-3; LoRA no-op exato e posto de B·A = r; int8 3,94×/erro 0,034 contra NF4 7,53×/erro 0,130; MoE ativos 38,2% e colapso 94,5% → 26,6% com a auxiliar; CoT 6,4× mais tokens e +5,3 p.p. no run real; 3 agentes com 8 skills e log `ok=true/false`. `lab-ia trilha --conferir` acusa citação a código inexistente (0 quebradas) e o teste anti-apodrecimento falha se um notebook ficar desatualizado — **ele falhou de verdade nesta sessão** ao editar um experimento sem regerar o caderno. A trilha também está **dentro da janela** (aba Trilha + `GET /trilha`), para o app portátil levar o material didático junto — 5 testes de API e 5 de UI **Não verificado:** execução dos notebooks — instalar jupyter exige rede, indisponível neste ambiente (mesma causa do matplotlib na B2) |

- 2026-09-14 · **B13 — conferência independente da implementação (3 defeitos, 2 corrigidos).**
  Baseline re-medido antes de mexer: **304 pytest** (1 pulado, exit 0) e **60 testes de UI** verdes.
  O que está de pé e foi conferido contra o núcleo rodando de verdade, não contra dublê:
  `/datasets/logica-pq` (3500 itens, 3 splits), `/corre/logica-1/progresso` (val 0,1804 em 2500 passos),
  `/corre/logica-1/benchmarks` (CoT 80,0% no teste e 71,7% no difícil, com `acuracia_por_familia` e `amostra_erros`),
  `POST /testar` (CoT guloso devolveu 3 passos + `"Resposta: 0"`, correta para p=1, q=0 em (p E q)) e o teacher forcing do RF7
  (`exploracao: true` e gravação em `.lab-ia/exploracao.jsonl` com o prefixo — guarda CA15 funcionando).
  **Achado de qualidade do modelo, não da interface:** no item acima o CoT gerado é **infiel** — o Passo 1 diz
  `(p OU-EX q) = 0` quando 1 XOR 0 = 1 — e ainda assim a resposta final sai certa. O benchmark só julga a resposta
  final, então "80% de acurácia" não diz nada sobre a fidelidade do raciocínio; é exatamente o tipo de erro que o painel
  "Onde ele errou" (CA16) existe para mostrar.
  **Defeitos:** (a) whitelist x UI (HTTP 400 nos dois modos — evidência abaixo), (b) build da UI quebrado pelo campo
  `total` ausente no tipo, (c) `/modelos` sem parâmetros e sem marca de raciocínio.
  **Comandos re-executáveis:** `curl -s -X POST localhost:8766/novo -H "Content-Type: application/json" -d @{"nome":"t","dados":"logica-pq","preset":"equilibrado","sobrescritas":{"dim":256,"camadas":6,"cabecas":8,"janela_ctx":256,"lote":32,"passos":100,"lr":0.0003,"abandono":0.1,"arquitetura":"classica"},"forcar":true}@ -> 400`;
  `pnpm --dir ui exec tsc --noEmit` (antes: exit 2; depois: exit 0) e `pnpm --dir ui build` (antes parava no tsc; depois "✓ built in 1.28s").
  **Impacto medido do conserto (c):** `logica-1` passou a reportar `parametros: 4913152` e `treinado_em_raciocinio: true`;
  a contagem bate exatamente com a aritmética à mão (448·256 + 256·256 = 180.224 de embedding; 6 camadas × (2·256² + 2·256² + 8·256² + 5·256) = 4.732.416; + 2·256 = **4.913.152**), o que confirma que o fallback usa a mesma conta verificada de `bancada/presets.contar_parametros` (comparada com `GPT.contar_parametros` em `tests/b7`).
  Restam 9 runs antigos (g1–g5, `livros-rapido`, `valida-custo`) sem parâmetros: o `estado.json` deles é do schema antigo (sem `config_modelo`) e os dois `tamanhos.json` que existem são o **relatório de quantização** da G3, que não tem chave `total` — não inventei número para eles.
- 2026-09-11 · **G4: a perda auxiliar do MoE não tinha gradiente.** A Lição 8 da trilha
  foi escrita para medir o mecanismo de balanceamento e o `backward()` da auxiliar
  estourou com *"element 0 of tensors does not require grad"*. Causa: o produto
  `n · Σ fᵢ·P̄ᵢ` estava inteiro dentro de `torch.no_grad()`, então `aux` entrava na perda
  como **constante** — o `coef_auxiliar` não tinha efeito nenhum no treino, contra o que
  a RF3 da G4 exige ("somada à perda de linguagem com `coef_auxiliar`"). O `aux_router`
  registrado em `metricas.jsonl` era diagnóstico, não sinal de treino — o balanceamento
  observado nas runs da G4 (uso 0,19–0,35, aux 1,048) veio da dinâmica natural.
  **Conserto:** `fracao` continua sob `no_grad` (contagem dura, como no Switch) e o
  produto saiu de lá, deixando `P̄ᵢ` carregar o gradiente. **Evidência do efeito:** partindo
  do mesmo colapso de 94,5%, sem auxiliar o roteador fica em 81,6% e com auxiliar vai a
  **26,6%** (uniforme, aux 1,0056 = piso); antes do conserto os dois davam resultado
  idêntico. **Regressão:** dois testes novos em `tests/g4/` (a auxiliar precisa ter gradiente;
  o coeficiente precisa mudar o gradiente do roteador).
- 2026-09-11 · G4: `test_gerar_no_moe` exigia que o texto gerado tivesse letras. Com a
  auxiliar agora ativa, a decodificação gulosa de um modelo micro de 300 passos passou a
  produzir 12 espaços (medido: `'            '` com auxiliar, `' a    a a a '` sem ela). A CA6
  pede que a geração funcione, não que seja literata — a asserção passou a verificar o
  mecanismo, com os valores medidos registrados no comentário.

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

- 2026-09-11 · B2: a primeira versão classificava `livros-rapido` como
  *instável* por causa da queda de 2,01 nats do passo 0 (modelo aleatório) para o
  passo 100. Falso: queda de linha de base não é instabilidade — a checagem de
  salto passou a começar depois do passo 0.
- 2026-09-11 · B2: a coluna *val@comum* saía "?" para todos os runs porque runs de
  quantização não têm campo `passo`, o `min()` do orçamento virava 0 e 0 era
  tratado como falso. Quantização saiu do ranking e do orçamento comum.
- 2026-09-11 · G8: o teste de queda real falhava ~1 em 6 execuções, por **duas causas
  medidas**, nenhuma delas aleatória. (a) **Bug de produto:** morrer entre gravar o
  último checkpoint e fechar o `estado.json` deixava o run com `concluido=false`
  para sempre — na retomada o laço não rodava (`range` vazio), nada era salvo e
  toda retomada futura virava no-op silencioso. Corrigido com fechamento explícito
  quando `passo_atual >= passos`, coberto por teste determinístico novo em
  `tests/g8`. (b) Corrida do próprio teste: a espera de 0,15 s entre pollings dava
  tempo de o modelo micro terminar os passos restantes antes do kill (o comentário
  do teste já admitia isso, mas a asserção seguinte não tratava). Espera reduzida
  para 0,02 s. Verificação: 10/10 execuções do caso `[35]` depois da correção
  (antes: 5/6, falha reproduzida com a saída completa).
- 2026-09-11 · Ambiente: o PyPI está inacessível nesta máquina (falha de SSL), então
  matplotlib não pôde ser instalado. A bancada não depende dele: as curvas saem em
  SVG gerado no próprio projeto; PNG é caminho opcional.

## Fora de escopo declarado
- G7 por último: depende da UI (G6) estável.
- Modelos grandes (>1B): inviável em 8GB VRAM + 48GB disco; o lab demonstra as técnicas em escala pequena, reproduzível e rápida.
- CI GitHub Actions: opcional no prompt; entra só se sobrar tempo.
