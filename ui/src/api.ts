/** Cliente da API interna do núcleo (spec G6 RF5). */
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

export const API_BASE_PADRAO = 'http://127.0.0.1:8765'

export function baseDaApi(): string {
  return localStorage.getItem('labia-api') ?? API_BASE_PADRAO
}

async function buscar<T>(caminho: string): Promise<T> {
  const resposta = await fetch(`${baseDaApi()}${caminho}`)
  if (!resposta.ok) throw new Error(`HTTP ${resposta.status} em ${caminho}`)
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
  executar: (acao: string, config: string) =>
    fetch(`${baseDaApi()}/execucao`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ acao, config }),
    }).then(async (r) => {
      const corpo = await r.json()
      if (!r.ok) throw new Error(String(corpo.detail ?? r.status))
      return corpo as { chave: string; pid: number }
    }),
}
