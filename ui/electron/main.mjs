// Processo principal do Electron — shell da SPA do Lab-IA (specs G6 RF2 e B4).
//
// Papel novo a partir da B4: o laboratório portátil precisa se virar sozinho numa
// máquina sem Python instalado. Então este processo resolve onde ficam os dados,
// semeia o material inicial, sobe o núcleo (exe congelado no pacote, venv em dev),
// espera o /saude e derruba tudo ao fechar a janela.
import { app, BrowserWindow, ipcMain } from 'electron'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  PORTA_PADRAO,
  escolherRaiz,
  escolherComandoNucleo,
  semearWorkspace,
  garantirNucleo,
} from './nucleo.mjs'

const aqui = dirname(fileURLToPath(import.meta.url))
const empacotado = app.isPackaged
const porta = Number(process.env.LABIA_PORTA ?? PORTA_PADRAO)
const raizLab = escolherRaiz({
  ambiente: process.env,
  empacotado,
  pastaUsuario: app.getPath('userData'),
  pastaApp: aqui,
})
const pastaEstado = join(raizLab, '.lab-ia')
const arquivoEstado = join(pastaEstado, 'ui-estado.json')

let estadoNucleo = { estado: 'pendente', raiz: raizLab, porta }
let processoNucleo = null

function lerEstado() {
  try {
    return JSON.parse(readFileSync(arquivoEstado, 'utf-8'))
  } catch {
    return {}
  }
}

function salvarEstado(parcial) {
  mkdirSync(pastaEstado, { recursive: true })
  const atual = { ...lerEstado(), ...parcial, salvo_em: new Date().toISOString() }
  writeFileSync(arquivoEstado, JSON.stringify(atual, null, 1), 'utf-8')
  return atual
}

ipcMain.handle('lab-ia:ler-estado', () => lerEstado())
ipcMain.handle('lab-ia:salvar-estado', (_e, parcial) => salvarEstado(parcial))
ipcMain.handle('lab-ia:nucleo', () => estadoNucleo)
ipcMain.handle('lab-ia:raiz', () => raizLab)

async function prepararNucleo() {
  const semente = empacotado ? join(process.resourcesPath, 'semente') : ''
  const semeadas = semearWorkspace({ raiz: raizLab, semente })
  const comando = escolherComandoNucleo({
    ambiente: process.env,
    empacotado,
    raiz: raizLab,
    recursos: process.resourcesPath,
  })
  const resultado = await garantirNucleo({ raiz: raizLab, porta, comando })
  processoNucleo = resultado.processo ?? null
  estadoNucleo = {
    estado: resultado.estado,
    erro: resultado.erro ?? null,
    pid: resultado.pid ?? null,
    origem: resultado.origem ?? null,
    saude: resultado.saude ?? null,
    raiz: raizLab,
    porta,
    semeadas,
    comando: comando ? comando.exe : null,
  }
  return estadoNucleo
}

function criarJanela() {
  const estado = lerEstado()
  const janela = new BrowserWindow({
    width: estado.largura ?? 1180,
    height: estado.altura ?? 780,
    x: estado.x,
    y: estado.y,
    backgroundColor: '#fff9ef',
    title: 'Lab-IA',
    webPreferences: {
      preload: join(aqui, 'preload.mjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const urlDev = process.env.VITE_DEV_URL
  if (urlDev) {
    janela.loadURL(urlDev)
  } else {
    janela.loadFile(join(aqui, '..', 'dist', 'index.html'))
  }

  const persistir = () => {
    const [x, y] = janela.getPosition()
    const [largura, altura] = janela.getSize()
    salvarEstado({ x, y, largura, altura })
  }
  janela.on('close', persistir)
  return janela
}

function encerrarNucleo() {
  if (processoNucleo && processoNucleo.exitCode === null) {
    try {
      processoNucleo.kill()
    } catch {
      /* processo já morto: nada a fazer */
    }
  }
  processoNucleo = null
}

app.whenReady().then(() => {
  criarJanela()
  // sobe o núcleo em paralelo: a janela abre na hora e a página mostra o estado
  prepararNucleo().catch((e) => {
    estadoNucleo = { estado: 'erro', erro: String(e), raiz: raizLab, porta }
  })
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) criarJanela()
  })
})
app.on('window-all-closed', () => {
  encerrarNucleo()
  if (process.platform !== 'darwin') app.quit()
})
app.on('before-quit', encerrarNucleo)
