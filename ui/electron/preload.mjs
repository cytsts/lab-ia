import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('labia', {
  lerEstado: () => ipcRenderer.invoke('lab-ia:ler-estado'),
  salvarEstado: (parcial) => ipcRenderer.invoke('lab-ia:salvar-estado', parcial),
})
