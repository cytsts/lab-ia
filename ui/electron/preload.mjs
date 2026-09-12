import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('labia', {
  lerEstado: () => ipcRenderer.invoke('lab-ia:ler-estado'),
  salvarEstado: (parcial) => ipcRenderer.invoke('lab-ia:salvar-estado', parcial),
  // estado do núcleo: a janela mostra se o laboratório portátil conseguiu subir o serviço
  nucleo: () => ipcRenderer.invoke('lab-ia:nucleo'),
  raiz: () => ipcRenderer.invoke('lab-ia:raiz'),
})
