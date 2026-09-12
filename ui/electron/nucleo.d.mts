/** Contrato do resolvedor do laboratório portátil (spec B4). */
export declare const PORTA_PADRAO: number
export declare const PASTAS_SEMENTE: string[]

export interface OpcoesRaiz {
  ambiente?: Record<string, string | undefined>
  empacotado?: boolean
  pastaUsuario?: string
  pastaApp?: string
}

export interface ComandoNucleo {
  exe: string
  args: string[]
  origem: string
}

export interface OpcoesComando {
  ambiente?: Record<string, string | undefined>
  empacotado?: boolean
  raiz?: string
  recursos?: string
  existe?: (caminho: string) => boolean
}

export interface OpcoesSemear {
  raiz: string
  semente?: string
  existe?: (caminho: string) => boolean
  copiar?: (origem: string, destino: string, opcoes?: { recursive?: boolean }) => void
  criar?: (caminho: string, opcoes?: { recursive?: boolean }) => unknown
}

export interface SaudeNucleo {
  ok?: boolean
  [chave: string]: unknown
}

export declare function escolherRaiz(opcoes?: OpcoesRaiz): string
export declare function escolherComandoNucleo(opcoes?: OpcoesComando): ComandoNucleo | null
export declare function semearWorkspace(opcoes?: OpcoesSemear): string[]
/** Resposta mínima que o resolvedor usa — aceita tanto fetch quanto um dublê de teste. */
export interface RespostaHttp {
  ok: boolean
  json: () => Promise<unknown>
}
export type Buscador = (url: string, init?: { signal?: AbortSignal }) => Promise<RespostaHttp>

export declare function consultarSaude(porta?: number, buscar?: Buscador): Promise<SaudeNucleo | null>
export declare function esperarSaude(opcoes?: {
  porta?: number
  limiteMs?: number
  intervaloMs?: number
  buscar?: Buscador
  agora?: () => number
  dormir?: (ms: number) => Promise<unknown>
  vivo?: () => boolean
}): Promise<{ ok: boolean; saude?: SaudeNucleo; erro?: string }>
export declare function garantirNucleo(opcoes?: {
  raiz: string
  porta?: number
  comando?: ComandoNucleo | null
  buscar?: Buscador
  lancar?: (...args: unknown[]) => unknown
  criar?: (...args: unknown[]) => unknown
  abrirLog?: (...args: unknown[]) => unknown
  esperar?: (opcoes: Record<string, unknown>) => Promise<{ ok: boolean; saude?: SaudeNucleo; erro?: string }>
}): Promise<{ estado: string; erro?: string; pid?: number; origem?: string; saude?: SaudeNucleo; processo?: unknown }>
