import React from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { ProvedorDeAvisos } from './ds/avisos'
import './ds/base.css'

const raiz = document.getElementById('raiz')
if (!raiz) throw new Error('elemento #raiz ausente')
createRoot(raiz).render(
  <React.StrictMode>
    <ProvedorDeAvisos>
      <App />
    </ProvedorDeAvisos>
  </React.StrictMode>,
)
