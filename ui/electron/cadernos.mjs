// Gerencia a sessão JupyterLab do laboratório. O Jupyter entrega editor,
// células, kernel e resultados; o Electron só prepara, inicia e abre a janela.
import { randomBytes } from 'node:crypto'
import { createWriteStream, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { spawn } from 'node:child_process'

export const PORTA_CADERNOS = 8889

export function urlCadernos({ porta = PORTA_CADERNOS, token }) {
  return `http://127.0.0.1:${porta}/lab/tree/cadernos?token=${encodeURIComponent(token)}`
}

export function argumentosJupyter({ raiz, porta = PORTA_CADERNOS, token }) {
  return [
    '-m', 'jupyterlab',
    `--ServerApp.root_dir=${raiz}`,
    '--ServerApp.ip=127.0.0.1',
    `--ServerApp.port=${porta}`,
    '--ServerApp.port_retries=0',
    '--ServerApp.open_browser=False',
    `--ServerApp.token=${token}`,
  ]
}

export async function esperarCadernos({
  url,
  buscar = globalThis.fetch,
  limiteMs = 30000,
  intervaloMs = 300,
  agora = Date.now,
  dormir = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  vivo = () => true,
} = {}) {
  const fim = agora() + limiteMs
  while (agora() < fim) {
    try {
      const resposta = await buscar(url, { signal: AbortSignal.timeout(1500) })
      if (resposta.ok) return { ok: true }
    } catch {
      // O servidor ainda está subindo; a próxima tentativa resolve ou o processo morre.
    }
    if (!vivo()) return { ok: false, erro: 'o processo do JupyterLab terminou antes de responder' }
    await dormir(intervaloMs)
  }
  return { ok: false, erro: 'o JupyterLab não respondeu em 30 s' }
}

/** Prepara o caderno por meio do núcleo e inicia um JupyterLab local autenticado. */
export async function iniciarCadernos({
  python,
  argsNucleo = [],
  raiz,
  porta = PORTA_CADERNOS,
  token = randomBytes(24).toString('hex'),
  lancar = spawn,
  criar = mkdirSync,
  abrirLog = createWriteStream,
  esperar = esperarCadernos,
} = {}) {
  if (!python) return { ok: false, erro: 'o pacote atual não traz Python/Jupyter; use o ambiente de desenvolvimento por enquanto' }

  const preparar = lancar(python, [...argsNucleo, 'laboratorio', '--sem-abrir', '--raiz', raiz], {
    cwd: raiz, windowsHide: true,
  })
  const preparo = await new Promise((resolve) => {
    preparar.once('error', (erro) => resolve({ codigo: -1, erro: String(erro) }))
    preparar.once('close', (codigo) => resolve({ codigo: codigo ?? -1 }))
  })
  if (preparo.codigo !== 0) return { ok: false, erro: `não foi possível preparar os cadernos (código ${preparo.codigo})` }

  const pastaLogs = join(raiz, '.lab-ia', 'logs')
  criar(pastaLogs, { recursive: true })
  const log = abrirLog(join(pastaLogs, 'jupyterlab.log'), { flags: 'a' })
  const processo = lancar(python, argumentosJupyter({ raiz, porta, token }), { cwd: raiz, windowsHide: true })
  processo.stdout?.pipe(log)
  processo.stderr?.pipe(log)
  const url = urlCadernos({ porta, token })
  const pronto = await esperar({ url, vivo: () => processo.exitCode === null && !processo.killed })
  if (!pronto.ok) return { ok: false, erro: pronto.erro, processo }
  return { ok: true, url, processo, porta }
}
