# Lab-IA

Laboratório de IA em português brasileiro: treinamento do zero, fine-tuning
(LoRA/QLoRA), quantização (int8/NF4), MoE, raciocínio (CoT/ToT) e agentes —
com interface desktop amistosa, retomada após queda e specs + testes para
tudo. Projeto guiado pelo loop Gauntlet: spec → teste → build → crítico.

Para **entender** o que os números dizem, há a trilha de estudo:
`.venv\Scripts\python -m labia.cli trilha` lista as lições; cada uma tem ideia curta,
o trecho exato do núcleo que a implementa, um experimento que você roda e exercícios
com resultado esperado. Comece por `trilha/LEIA.md`.

Além das metas fechadas (G1…G10), há a **bancada (B1/B5)** para uso próprio:
trazer o seu corpus, gerar a configuração explicada com estimativa de tempo e
VRAM, treinar, comparar runs lado a lado com diagnóstico (overfit, platô, deriva)
e desenhar as curvas. Comece por `docs/bancada.md`.

## Requisitos

- Windows/Linux/macOS com **Python 3.13+** e **Node 20+** + **pnpm**
- GPU NVIDIA (CUDA 12.x) opcional — tudo roda em CPU (mais lento)
- ~5 GB livres para ambiente + artefatos

## Instalação (do zero)

```bat
:: 1. Ambiente Python
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e "core[dev,finetune]"
:: (opcional, comparativo de quantização) .venv\Scripts\pip install bitsandbytes

:: 2. Dados (domínio público / sintéticos determinísticos, versionados)
.venv\Scripts\python scripts\prepara_corpus.py
.venv\Scripts\python scripts\prepara_tarefa_ciencia.py
.venv\Scripts\python scripts\prepara_benchmark.py

:: 3. Interface
pnpm install --dir ui
cd ui && node node_modules\.pnpm\electron@*\node_modules\electron\install.js  :: binário Electron (pnpm 10 bloqueia postinstall)
```

## Verificação

```bat
.venv\Scripts\python -m pytest -q            :: suíte completa (CPU, sem GPU/rede)
.venv\Scripts\python -m pytest -q -m "not slow"  :: ciclo rápido
pnpm --dir ui test                           :: testes da UI
pnpm --dir ui coverage                       :: cobertura TS (v8)
.venv\Scripts\python scripts\verifica_pacote.py .qwen\pyi-dist\lab-ia --cuda  :: G7: exe do núcleo (após o build abaixo)
```

## Metas e comandos

| Meta | Comando | Artefato |
|------|---------|----------|
| B1 dados próprios | `... dados --de C:\meus\textos --id meus-textos` | `data/meus-textos/{trem,val}.txt` + `manifesto.json` |
| B1 config assistida | `... novo --nome meu-run --dados meus-textos --preset rapido` | `configs/meu-run.yaml` comentado + estimativas |
| B2 comparar runs | `... comparar [runs...] [--relatorio rel.md]` | tabela, ranking e veredito por run |
| B2 curvas | `... curva a b --saida curvas.svg` | SVG (sem dependências) ou PNG (matplotlib) |
| B3 API da bancada | `... servir` + @BT@/saude /datasets /dados /novo /comparar /comparar/curvas.svg /guia@BT@ | contrato consumido pela janela |
| B3 janela da bancada | `pnpm --dir ui dev` → abas **Bancada** e **Comparar** | ciclo completo sem terminal |
| B5 API da varredura | `POST /varrer/seco` · `POST /varrer` · `GET /varreduras` | grade conferida e varredura disparada pela janela |
| B5 varredura | `... varrer --base configs/x.yaml --grade lr=1e-4,3e-4 --passos 500` | `runs/_varredura/<prefixo>/varredura.json` + runs `<prefixo>-NN` |
| B5 conferir grade | `... varrer --base configs/x.yaml --grade dim=64,100 --seco` | lista as combinações sem treinar |
| B6 trilha de estudo | `... trilha` · `--licao 3` · `--licao 3 --rodar` · `--conferir` | 5 lições + experimentos + notebooks |
| B6 notebooks | `.venv\Scripts\python scripts\gera_notebooks.py` | `trilha/notebooks/*.ipynb` (gerados do experimento) |
| B6 trilha na janela | aba **Trilha** (`pnpm --dir ui dev`) ou `GET /trilha` · `GET /trilha/{n}` | lição lida e experimento rodado dentro do app |
| B4 pacote portátil | `.venv\Scripts\python scripts\monta_portatil.py` | `pacote/nucleo/` + `pacote/semente/`; mede o peso |
| B1/B2 guia | `docs/bancada.md` | ciclo completo: dados → config → treino → comparação |
| G1 treino do zero | `.venv\Scripts\python -m labia.cli train --config configs/g1_treino_zero.yaml` | `runs/g1-treino-zero/` |
| G1 retomar após queda | `... train --config ... --retomar` | continua do último ckpt |
| G1 gerar | `... gerar --run g1-treino-zero --prompt "Uma noite destas"` | prosa pt-BR |
| G2 LoRA | `... ajustar --config configs/g2_ajuste_ciencia.yaml` | `runs/g2-ajuste-ciencia/adaptador/` |
| G2 QLoRA | mude `tipo: qlora` (bits 8 ou 4) na config | idem + base quantizada |
| G3 quantizar | `... quantizar --run g1-treino-zero --saida g3-g1-nf4 --modo nf4` | `tamanhos.json` |
| G4 MoE | `... train --config configs/g4_moe.yaml` | métricas de rota no `metricas.jsonl` |
| G5 raciocínio | `... raciocinio --run g5-ajuste-cot --comparar` | `comparativo.json` |
| G9 agentes | `... agentes` / `... agente treinador train_model --json "{\"config\": ...}"` | eventos `agente_acao` |
| API p/ UI | `... servir --porta 8765` | REST local |
| G6 desktop (dev) | `pnpm --dir ui dev` + `pnpm --dir ui electron` | SPA + shell |
| G7 empacotar UI | `pnpm --dir ui package` | `ui/release/Lab-IA 0.1.0.exe` (portable) |
| G7 empacotar núcleo | `.venv\Scripts\pyinstaller --noconfirm --distpath .qwen\pyi-dist --workpath .qwen\pyi-work lab-ia.spec` | `.qwen\pyi-dist\lab-ia\lab-ia.exe` (headless, sem venv) |

Artefatos de um experimento em `runs/<run-id>/`: `estado.json` (progresso),
`metricas.jsonl` (append-only), `ckpt/` (checkpoints atômicos), `tokens/`
(BPE), `adaptador/` (LoRA), `tamanhos.json` (quantização), `benchmark-*.json`
(raciocínio). Log global: `.lab-ia/eventos.jsonl`.

## Estrutura

| Caminho | Papel |
|---|---|
| `core/labia/` | Núcleo Python: modelos, treino, ajuste, quantização, MoE, raciocínio, agentes, API |
| `lab-ia.spec` | Empacotamento do núcleo (PyInstaller); fixa as CRTs do `System32` — ver `specs/G7.md` |
| `scripts/` | Preparação de dados, ponto de entrada congelado, verificador do pacote |
| `ui/` | Electron + React/TS + Design System próprio (`ui/src/ds/`) |
| `specs/` | Uma spec por meta (G1…G10, B1…B4) + `PLANO.md` = estado vivo do loop |
| `docs/` | Guias de uso em pt-BR (`bancada.md` = trazer dados, gerar config, treinar, comparar) |
| `trilha/` | Trilha de estudo: lições em pt-BR mapeadas ao código, experimentos executáveis e notebooks gerados |
| `ui/electron/nucleo.mjs` | Resolvedor do lab portátil: onde ficam os dados, qual núcleo subir, como esperar o /saude |
| `tests/` | Testes derivados das specs (g1…g10; sufixo `slow` = exige runs reais) |
| `configs/` | Configurações YAML reproduzíveis (sementes fixas) |
| `data/` | Corpus literário pt-BR (domínio público) + tarefa + benchmark (geráveis) |

## Laboratório portátil (em andamento, com números medidos)

O objetivo é instalar em outra máquina, abrir a janela e usar. Hoje:

- o app Electron **sobe o núcleo sozinho** (pacote congelado, ou o venv em dev),
  resolve onde ficam os dados (`%APPDATA%\Lab-IA\lab` quando empacotado), semeia
  `configs/`, `docs/` e `data/` na primeira execução e derruba o núcleo ao fechar;
- `.venv\Scripts\python scripts\monta_portatil.py` junta `pacote/nucleo/` (núcleo
  congelado) e `pacote/semente/` (material inicial) e **mede** o resultado.

Medição que decide o formato: o núcleo congelado tem **4,84 GB**, dos quais
**4,08 GB são o torch com CUDA**. O alvo `portable` do electron-builder se auto-extrai
para `%TEMP%` a cada execução — com 4,84 GB isso é espera toda vez que o app abre.
Daí a decisão pendente entre três sabores (medidos em `specs/B4.md`):

| sabor | tamanho | quando serve |
|---|---|---|
| instalador (NSIS) + CUDA | ~4,9 GB | máquina com GPU NVIDIA, treino de verdade |
| instalador + CPU | ~500 MB | qualquer máquina, treino menor e mais lento |
| portátil de arquivo único | ~4,9 GB de auto-extração | só se não puder instalar nada |

## O que este lab demonstra honestamente

Os achados medidos estão nas specs/PLANO (ex.: MoE ≈ denso em corpus pequeno;
aritmética com carry não generaliza em 15M; CoT +5,3 p.p. vs direta). Cobertura
medida: núcleo 90%, UI 96,7%.
