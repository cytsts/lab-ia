/** Cliente da API interna do núcleo (specs G6 RF5 e B3). */
export interface EstadoRun {
  run_id: string
  tipo?: string
  passo?: number
  passos_totais?: number
  concluido?: boolean
  dispositivo?: string
  atualizado_em?: string
  [chave: string]: unknown
}

export interface RegistroMetrica {
  passo: number | null
  loss_trem?: number | null
  loss_val?: number | null
  lr?: number
  tokens_por_s?: number
  tempo_s?: number
  aux_router?: number | null
  uso_especialistas?: number[] | null
  origem?: string
  tipo?: string
  estrategia?: string
  acuracia_global?: number
  [chave: string]: unknown
}

export interface Saude {
  ok: boolean
  versao_api: string
  raiz: string
  cuda: boolean
  gpu: string | null
  runs: number
  datasets: number
  configs: number
  guia: string[]
}

export interface Dataset {
  id: string
  gerado_em?: string
  chars_trem: number
  chars_val: number
  idioma: string
  tokens_aprox_trem: number
  fontes: string[]
}

export interface Preset {
  nome: string
  descricao: string
  modelo: Record<string, number>
  passos: number
  lote: number
}

export interface ResumoRun {
  run_id: string
  familia: string
  passos: number
  parametros: number | null
  melhor_val: number | null
  passo_melhor_val: number | null
  val_final: number | null
  drift: number | null
  melhor_val_orcamento_comum: number | null
  tokens_por_s: number | null
  diagnostico: string
  comentario: string
}

export interface ResultadoComparacao {
  runs: ResumoRun[]
  orcamento_comum: number | null
  familias: string[]
  ranking: string[]
  melhor: string | null
  veredito: string
  limiares: Record<string, number>
}

export interface ResultadoNovo {
  caminho: string
  caminho_relativo: string
  arquivo_config: string
  resumo: string
  config: Record<string, unknown>
  parametros: { total: number; ativos: number }
  desempenho: { segundos_previstos: number; tokens_por_s_previsto: number; segundos_por_passo: number }
  vram: { mb_total: number }
  epocas: number | null
  yaml: string
}

export interface VarianteVarredura {
  indice: number
  run_id: string
  sobrecritas: Record<string, number | string>
  config?: string
  melhor_val?: number | null
  passo_melhor_val?: number | null
  diagnostico?: string
  erro?: string | null
}

export interface EfeitoChave {
  base: number | string | null
  por_valor: Record<string, { n: number; media_melhor_val: number | null; melhor_val: number | null }>
  melhor_valor: string | null
}

export interface RelatorioVarredura {
  prefixo: string
  base: string
  modo: string
  passos_por_variante: number
  semente: number
  seco?: boolean
  grades: Record<string, string[]>
  variantes: VarianteVarredura[]
  ranking?: string[]
  melhor?: { run_id: string; melhor_val: number; sobrecritas: Record<string, number | string> } | null
  efeito_por_chave?: Record<string, EfeitoChave>
  falhas?: string[]
  tabela?: string
}

export interface ResumoVarredura {
  prefixo: string
  base: string
  criado_em?: string
  melhor: { run_id: string; melhor_val: number } | null
  variantes: number
  falhas: string[]
}

export interface CorpoVarredura {
  base: string
  grades: string[]
  modo?: string
  n?: number
  passos?: number
  prefixo?: string
}

export interface ResumoLicao {
  numero: number
  titulo: string
  arquivo: string
  experimento: string | null
  resumo: string
  tem_experimento: boolean
}

export interface CitacaoQuebrada {
  licao: number
  arquivo: string
  simbolo?: string
  motivo: string
}

export interface IndiceTrilha {
  total: number
  licoes: ResumoLicao[]
  citacoes_quebradas: CitacaoQuebrada[]
}

// B13 — Laboratório
export interface ItemBenchmark {
  id: string
  enunciado: string
  cot: string
  resposta: number | string
  familia: string
  split: string
  [chave: string]: unknown
}

export interface ManifestoDataset {
  id: string
  manifesto: Record<string, unknown>
  familias: string[]
  splits: Record<string, number>
  itens_treino?: number
  itens_benchmark?: number
  sha256?: string
  idioma: string
  tokens_aprox_trem?: number
  tem_benchmark: boolean
}

export interface RespostaItensDataset {
  id: string
  total: number
  filtrados: number
  desde: number
  limite: number
  itens: ItemBenchmark[]
  tem_benchmark: boolean
}

export interface ProgressoRun {
  run_id: string
  passo: number
  passos_totais: number
  concluido: boolean
  loss_trem: number | null
  loss_val: number | null
  lr: number | null
  tokens_por_s: number | null
  tempo_s: number | null
  tempo_restante_s: number | null
  sparkline: { passo: number; loss_trem?: number | null; loss_val?: number | null }[]
  benchmark_progresso?: {
    item_atual: number
    total_itens: number
    acertos: number
    acuracia_parcial: number
    concluido: boolean
  } | null
}

export interface RelatorioBenchmark {
  run: string
  estrategia: string
  split: string
  semente: number
  itens: number
  acuracia_global: number
  acuracia_por_familia: Record<string, number>
  taxa_resposta_valida: number
  amostra_erros: { id: string; esperado: number | string; obtido: number | string | null; texto: string }[]
  arquivo?: string
}

export interface ModeloBaseRun {
  id: string
  origem: 'run'
  parametros: number | null
  dispositivo: string
  medido: boolean
  treinado_em_raciocinio: boolean
  benchmarks: { estrategia: string; split: string; acuracia_global: number; itens: number }[]
}

export interface ModeloGGUF {
  arquivo: string
  origem: 'gguf'
  tamanho_mb: number
  quantizacao: string
  reasoning_declarado: boolean
  rotulo_honestidade: string
  medido: boolean
}

export interface CatalogoModelos {
  runs: ModeloBaseRun[]
  gguf: ModeloGGUF[]
  llama_cpp_instalado: boolean
  comando_instalacao: string
}

export interface CorpoTestar {
  run: string
  enunciado: string
  estrategia?: string
  temperatura?: number
  guloso?: boolean
  max_tokens?: number
  prefixo?: string | null
  semente?: number
}

export interface RespostaTestar {
  run: string
  enunciado: string
  estrategia: string
  texto_gerado: string
  resposta_extraida: number | string | null
  exploracao: boolean
  tempo_s: number
}

export interface CorpoBenchmark {
  run: string
  estrategia?: string
  benchmark?: string
  split?: string
  limite?: number | null
  semente?: number
}

export const API_BASE_PADRAO = 'http://127.0.0.1:8765'

export function baseDaApi(): string {
  return localStorage.getItem('labia-api') ?? API_BASE_PADRAO
}

async function mensagemDeErro(resposta: Response, caminho: string): Promise<string> {
  try {
    const corpo = await resposta.json()
    if (corpo && typeof corpo.detail === 'string') return corpo.detail
  } catch {
    /* resposta sem JSON: cai na mensagem genérica */
  }
  return `HTTP ${resposta.status} em ${caminho}`
}

async function buscar<T>(caminho: string): Promise<T> {
  const resposta = await fetch(`${baseDaApi()}${caminho}`)
  if (!resposta.ok) throw new Error(await mensagemDeErro(resposta, caminho))
  return (await resposta.json()) as T
}

async function buscarTexto(caminho: string): Promise<string> {
  const resposta = await fetch(`${baseDaApi()}${caminho}`)
  if (!resposta.ok) throw new Error(await mensagemDeErro(resposta, caminho))
  return resposta.text()
}

async function enviar<T>(caminho: string, corpo: unknown): Promise<T> {
  const resposta = await fetch(`${baseDaApi()}${caminho}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(corpo),
  })
  if (!resposta.ok) throw new Error(await mensagemDeErro(resposta, caminho))
  return (await resposta.json()) as T
}

export const api = {
  corre: () => buscar<EstadoRun[]>('/corre'),
  metricas: (id: string) => buscar<RegistroMetrica[]>(`/corre/${encodeURIComponent(id)}/metricas`),
  estado: (id: string) => buscar<EstadoRun>(`/corre/${encodeURIComponent(id)}/estado`),
  tamanhos: (id: string) => buscar<Record<string, unknown>>(`/corre/${encodeURIComponent(id)}/tamanhos`),
  comparativo: (id: string) => buscar<Record<string, unknown>>(`/corre/${encodeURIComponent(id)}/comparativo`),
  configs: () => buscar<string[]>('/configs'),
  eventos: (desde = 0) => buscar<Record<string, unknown>[]>(`/eventos?desde=${desde}`),
  execucoes: () => buscar<{ chave: string; pid: number; vivo: boolean; codigo_saida: number | null }[]>('/execucoes'),
  executar: (acao: string, config: string) => enviar<{ chave: string; pid: number }>('/execucao', { acao, config }),
  logExecucao: (chave: string, linhas = 60) =>
    buscar<{ chave: string; linhas: string[]; existe: boolean; total: number }>(
      `/execucoes/${encodeURIComponent(chave)}/log?linhas=${linhas}`,
    ),

  // bancada B3 — o ciclo completo pela janela
  saude: () => buscar<Saude>('/saude'),
  datasets: () => buscar<Dataset[]>('/datasets'),
  presets: () => buscar<Preset[]>('/presets'),
  prepararDados: (corpo: {
    fontes: string[]
    id: string
    frac_val?: number
    min_chars?: number
    quase_duplicata?: boolean
    campo?: string | null
    forcar?: boolean
  }) => enviar<Record<string, unknown>>('/dados', corpo),
  novo: (corpo: {
    nome: string
    dados: string
    preset: string
    sobrescritas?: Record<string, number>
    forcar?: boolean
  }) => enviar<ResultadoNovo>('/novo', corpo),
  comparar: (runs?: string[]) =>
    buscar<ResultadoComparacao>(
      runs && runs.length
        ? `/comparar?runs=${encodeURIComponent(runs.join(','))}`
        : '/comparar',
    ),
  urlCurvas: (runs: string[], metricas = 'loss_val,loss_trem') =>
    `${baseDaApi()}/comparar/curvas.svg?runs=${encodeURIComponent(runs.join(','))}&metricas=${encodeURIComponent(metricas)}`,
  urlRelatorio: (runs: string[]) =>
    `${baseDaApi()}/comparar/relatorio?runs=${encodeURIComponent(runs.join(','))}`,
  urlGuia: (nome: string) => `${baseDaApi()}/guia/${encodeURIComponent(nome)}`,

  // bancada B5 — varredura de hiperparâmetros
  varrerSeco: (corpo: CorpoVarredura) => enviar<RelatorioVarredura>('/varrer/seco', corpo),
  varrer: (corpo: CorpoVarredura) => enviar<{ chave: string; pid: number; prefixo: string }>('/varrer', corpo),
  varreduras: () => buscar<ResumoVarredura[]>('/varreduras'),
  varredura: (prefixo: string) => buscar<RelatorioVarredura>(`/varreduras/${encodeURIComponent(prefixo)}`),

  // trilha de estudo (B6) — o material didático dentro da janela
  trilha: () => buscar<IndiceTrilha>('/trilha'),
  licao: (numero: number) => buscarTexto(`/trilha/${numero}`),
  rodarLicao: (numero: number) =>
    enviar<{ chave: string; pid: number; licao: number; experimento: string }>(
      `/trilha/${numero}/rodar`,
      {},
    ),

  // Laboratório (B13) — interface estilo caderno e ciclo integrado
  dataset: (id: string) => buscar<ManifestoDataset>(`/datasets/${encodeURIComponent(id)}`),
  itensDataset: (
    id: string,
    params: { desde?: number; limite?: number; split?: string; familia?: string; busca?: string } = {},
  ) => {
    const q = new URLSearchParams()
    if (params.desde !== undefined) q.set('desde', String(params.desde))
    if (params.limite !== undefined) q.set('limite', String(params.limite))
    if (params.split) q.set('split', params.split)
    if (params.familia) q.set('familia', params.familia)
    if (params.busca) q.set('busca', params.busca)
    const qs = q.toString()
    return buscar<RespostaItensDataset>(`/datasets/${encodeURIComponent(id)}/itens${qs ? `?${qs}` : ''}`)
  },
  progresso: (id: string) => buscar<ProgressoRun>(`/corre/${encodeURIComponent(id)}/progresso`),
  benchmarksDoRun: (id: string) => buscar<RelatorioBenchmark[]>(`/corre/${encodeURIComponent(id)}/benchmarks`),
  pararExecucao: (chave: string) =>
    enviar<{ chave: string; pid: number; parado: boolean }>(`/execucoes/${encodeURIComponent(chave)}/parar`, {}),
  testar: (corpo: CorpoTestar) => enviar<RespostaTestar>('/testar', corpo),
  dispararBenchmark: (corpo: CorpoBenchmark) =>
    enviar<{ chave: string; pid: number; comando: string[] }>('/benchmark', corpo),
  modelos: () => buscar<CatalogoModelos>('/modelos'),
}
