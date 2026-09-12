// Resolução do laboratório portátil: onde ficam os dados, qual núcleo subir e como.
//
// Fica separado de main.mjs de propósito: assim a lógica que decide caminhos e
// comandos é testável sem abrir janela nenhuma (test/nucleo.test.ts).
import { existsSync, mkdirSync, cpSync, createWriteStream } from 'node:fs'
import { join } from 'node:path'
import { spawn } from 'node:child_process'

export const PORTA_PADRAO = 8765
export const PASTAS_SEMENTE = ['configs', 'docs', 'data']

/** Onde vive o laboratório: variável explícita > pasta do usuário (empacotado) > repositório. */
export function escolherRaiz({ ambiente = {}, empacotado = false, pastaUsuario = '', pastaApp = '' } = {}) {
  if (ambiente.LABIA_RAIZ) return ambiente.LABIA_RAIZ
  if (empacotado) return join(pastaUsuario, 'lab')
  return join(pastaApp, '..', '..')
}

/** Qual comando sobe o núcleo: exe congelado, comando explícito ou o Python do venv. */
export function escolherComandoNucleo({ ambiente = {}, empacotado = false, raiz = '', recursos = '', existe = existsSync } = {}) {
  if (ambiente.LABIA_NUCLEO) return { exe: ambiente.LABIA_NUCLEO, args: [], origem: 'LABIA_NUCLEO' }
  if (empacotado) {
    const exe = join(recursos, 'nucleo', process.platform === 'win32' ? 'lab-ia.exe' : 'lab-ia')
    if (existe(exe)) return { exe, args: [], origem: 'nucleo empacotado' }
    return null
  }
  const python =
    process.platform === 'win32' ? join(raiz, '.venv', 'Scripts', 'python.exe') : join(raiz, '.venv', 'bin', 'python')
  if (existe(python)) return { exe: python, args: ['-m', 'labia.cli'], origem: 'venv do repositório' }
  return { exe: process.platform === 'win32' ? 'python' : 'python3', args: ['-m', 'labia.cli'], origem: 'python do PATH' }
}

/** Copia o material inicial (configs, docs, data) para o workspace do usuário, uma vez. */
export function semearWorkspace({ raiz, semente = '', existe = existsSync, copiar = cpSync, criar = mkdirSync } = {}) {
  const semeadas = []
  if (!semente || !existe(semente)) return semeadas
  criar(raiz, { recursive: true })
  for (const pasta of PASTAS_SEMENTE) {
    const origem = join(semente, pasta)
    const destino = join(raiz, pasta)
    if (existe(destino) || !existe(origem)) continue
    copiar(origem, destino, { recursive: true })
    semeadas.push(pasta)
  }
  return semeadas
}

/** Consulta /saude; devolve o corpo quando o núcleo responde, null quando não. */
export async function consultarSaude(porta = PORTA_PADRAO, buscar = globalThis.fetch) {
  try {
    const resposta = await buscar(`http://127.0.0.1:${porta}/saude`, { signal: AbortSignal.timeout(1500) })
    return resposta.ok ? await resposta.json() : null
  } catch {
    return null
  }
}

/** Espera o núcleo ficar saudável; devolve {ok} ou {erro} com o motivo. */
export async function esperarSaude({
  porta = PORTA_PADRAO,
  limiteMs = 180000,
  intervaloMs = 700,
  buscar = globalThis.fetch,
  agora = Date.now,
  dormir = (ms) => new Promise((r) => setTimeout(r, ms)),
  vivo = () => true,
} = {}) {
  const fim = agora() + limiteMs
  while (agora() < fim) {
    const saude = await consultarSaude(porta, buscar)
    if (saude) return { ok: true, saude }
    if (!vivo()) return { ok: false, erro: 'o processo do núcleo morreu antes de responder' }
    await dormir(intervaloMs)
  }
  return { ok: false, erro: `o núcleo não respondeu em ${Math.round(limiteMs / 1000)} s` }
}

/** Sobe o núcleo do laboratório; reaproveita um que já esteja saudável na porta. */
export async function garantirNucleo({
  raiz,
  porta = PORTA_PADRAO,
  comando,
  buscar = globalThis.fetch,
  lancar = spawn,
  criar = mkdirSync,
  abrirLog = createWriteStream,
  esperar = esperarSaude,
} = {}) {
  const jaNoAr = await consultarSaude(porta, buscar)
  if (jaNoAr) return { estado: 'reutilizado', saude: jaNoAr }

  if (!comando) return { estado: 'erro', erro: 'núcleo não encontrado no pacote nem no venv' }

  const pastaLogs = join(raiz, '.lab-ia', 'logs')
  criar(pastaLogs, { recursive: true })
  const log = abrirLog(join(pastaLogs, 'nucleo.log'), { flags: 'a' })
  const argumentos = [...comando.args, 'servir', '--porta', String(porta), '--raiz', raiz]
  const processo = lancar(comando.exe, argumentos, { cwd: raiz, windowsHide: true })
  processo.stdout?.pipe(log)
  processo.stderr?.pipe(log)

  const resultado = await esperar({
    porta,
    buscar,
    vivo: () => processo.exitCode === null && !processo.killed,
  })
  if (!resultado.ok) return { estado: 'erro', erro: resultado.erro, pid: processo.pid, comando: comando.exe }
  return { estado: 'iniciado', pid: processo.pid, comando: comando.exe, origem: comando.origem, saude: resultado.saude, processo }
}
