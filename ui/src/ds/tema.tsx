import { useEffect, useState } from 'react'

export type Tema = 'claro' | 'escuro'

const CHAVE = 'labia-tema'

export function temaAtual(): Tema {
  const salvo = localStorage.getItem(CHAVE)
  if (salvo === 'claro' || salvo === 'escuro') return salvo
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'escuro' : 'claro'
}

export function aplicarTema(tema: Tema) {
  document.documentElement.dataset.tema = tema
  localStorage.setItem(CHAVE, tema)
}

export function useTema(): [Tema, () => void] {
  const [tema, setTema] = useState<Tema>(() => temaAtual())
  useEffect(() => aplicarTema(tema), [tema])
  return [tema, () => setTema((t) => (t === 'claro' ? 'escuro' : 'claro'))]
}

export function AlternadorDeTema() {
  const [tema, alternar] = useTema()
  return (
    <button className="botao" onClick={alternar} aria-label="alternar tema">
      {tema === 'claro' ? '🌙 Escuro' : '☀️ Claro'}
    </button>
  )
}
