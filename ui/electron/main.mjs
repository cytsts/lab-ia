// Processo principal do Electron — shell da SPA do Lab-IA (spec G6 RF2).
import { app, BrowserWindow, ipcMain } from 'electron'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const aqui = dirname(fileURLToPath(import.meta.url))
const raizLab = process.env.LABIA_RAIZ ?? join(aqui, '..', '..')
const pastaEstado = join(raizLab, '.lab-ia')
const arquivoEstado = join(pastaEstado, 'ui-estado.json')

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

app.whenReady().then(() => {
  criarJanela()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) criarJanela()
  })
})
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
