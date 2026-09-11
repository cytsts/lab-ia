import { useEffect, useState } from 'react'
import { api } from '../api'
import { Cartao } from '../ds/componentes'

export function Eventos() {
  const [lista, setLista] = useState<Record<string, unknown>[]>([])
  const [erro, setErro] = useState<string | null>(null)

  const recarregar = () =>
    api
      .eventos(0)
      .then((e) => {
        setLista(e)
        setErro(null)
      })
      .catch((e: Error) => setErro(e.message))

  useEffect(() => {
    recarregar()
    const t = setInterval(recarregar, 5000)
    return () => clearInterval(t)
  }, [])

  return (
    <Cartao
      titulo={
        <span className="linha" style={{ justifyContent: 'space-between', width: '100%' }}>
          Log de eventos (append-only)
          <button className="botao" onClick={recarregar}>↻</button>
        </span>
      }
    >
      {erro ? <p style={{ color: 'var(--perigo)' }}>{erro}</p> : null}
      <table className="tabela">
        <thead>
          <tr><th>tempo</th><th>tipo</th><th>detalhes</th></tr>
        </thead>
        <tbody>
          {[...lista].reverse().slice(0, 200).map((e, i) => {
            const { tempo, tipo, ...resto } = e as { tempo?: string; tipo?: string }
            return (
              <tr key={i}>
                <td className="mono">{tempo}</td>
                <td><strong>{tipo}</strong></td>
                <td className="mono">{JSON.stringify(resto)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </Cartao>
  )
}
