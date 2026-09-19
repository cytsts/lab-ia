export const PORTA_CADERNOS: number
export function urlCadernos(opcoes: { porta?: number; token: string }): string
export function argumentosJupyter(opcoes: { raiz: string; porta?: number; token: string }): string[]
export function esperarCadernos(opcoes?: Record<string, unknown>): Promise<{ ok: boolean; erro?: string }>
