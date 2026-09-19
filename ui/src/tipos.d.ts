/** Ponte exposta pelo preload do Electron (spec B4: laboratório portátil). */
export {}

export interface EstadoNucleo {
  estado: 'pendente' | 'iniciado' | 'reutilizado' | 'erro'
  erro?: string | null
  pid?: number | null
  comando?: string | null
  origem?: string | null
  raiz: string
  porta: number
}

declare global {
  interface Window {
    labia?: {
      lerEstado?: () => Promise<Record<string, unknown>>
      salvarEstado?: (parcial: Record<string, unknown>) => Promise<Record<string, unknown>>
      nucleo?: () => Promise<EstadoNucleo>
      raiz?: () => Promise<string>
      abrirCadernos?: () => Promise<{ ok: boolean; url?: string; erro?: string }>
    }
  }
}
