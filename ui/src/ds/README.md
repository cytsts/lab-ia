# Design System do Lab-IA — "Coruja" 🦉

Identidade amistosa para um laboratório de ML: creme quente, âmbar, cantos
arredondados e microinterações discretas. Tudo em tokens CSS puros — zero
dependência de UI libs.

## Fundamentos

| Token | Claro | Escuro | Uso |
|---|---|---|---|
| `--fundo` | `#fff9ef` | `#191c1a` | fundo geral |
| `--primaria` | `#b45309` | `#f59e0b` | ações, marca (AA s/ creme) |
| `--secundaria` | `#0f766e` | `#2dd4bf` | progresso, links |
| `--ok / --aviso / --perigo / --info` | verdes/âmbar/vermelhos/azuis escuros | versões claras | status |
| `--fonte` | system-ui amigável (Segoe UI Variable / Nunito / Open Sans) | idem | corpo |
| `--raio` | 14px / 10px | — | cartões / controles |

Contraste: todos os pares texto/fundo passam WCAG AA (≥4.5:1 texto normal).
Tema segue `prefers-color-scheme`, alternável e persistido (`localStorage` +
`.lab-ia/ui-estado.json` via IPC no Electron).

## Componentes (`src/ds/`)

- `Cartao` — contêiner básico com sombra suave.
- `Botão` (`.botao`, `.botao--primario`) — feedback `:active` com scale.
- `Distintivo` — badges de status (`ok/aviso/perigo/info`) mapeados do
  `estado.json` dos runs.
- `BarraProgresso` — passo atual do experimento (role=progressbar + aria).
- `GraficoDeMetricas` — SVG de linhas (perda treino/validação), eixo
  dimensionado, estado vazio amigável.
- `BarrasDeUso` — tráfego por especialista (MoE).
- `Avisos` (toasts) — `useAvisos().avisar(texto, erro?)`, auto-dismiss 4,2 s,
  `aria-live=polite`.
- `AlternadorDeTema` — 🌙/☀️.

## Catálogo vivo

Rota **“Design System”** no app (`src/paginas/CatalogoDS.tsx`) renderiza todos
os componentes com exemplos interativos — abra o app e clique na aba.

## Testes

`pnpm test` cobre: gráfico (2 séries + vazio), badges, progresso (aria),
alternância de tema (persistência) e Painel com API mockada (lista/erro).
