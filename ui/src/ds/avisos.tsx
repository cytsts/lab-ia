import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

interface AvisoItem {
  id: number
  texto: string
  erro?: boolean
}

const contexto = createContext<{ avisar(texto: string, erro?: boolean): void }>({ avisar: () => {} })

export function useAvisos() {
  return useContext(contexto)
}

export function ProvedorDeAvisos(props: { children: ReactNode }) {
  const [lista, setLista] = useState<AvisoItem[]>([])
  const avisar = useCallback((texto: string, erro = false) => {
    const id = Date.now() + Math.random()
    setLista((atual) => [...atual, { id, texto, erro }])
    setTimeout(() => setLista((atual) => atual.filter((a) => a.id !== id)), 4200)
  }, [])
  return (
    <contexto.Provider value={{ avisar }}>
      {props.children}
      <div className="avisos" aria-live="polite">
        {lista.map((a) => (
          <div key={a.id} className={`aviso${a.erro ? ' aviso--erro' : ''}`} role="status">
            {a.texto}
          </div>
        ))}
      </div>
    </contexto.Provider>
  )
}
