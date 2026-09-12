import { useEffect, useState } from 'react'
import { api, type EstadoRun, type ResultadoComparacao, type ResumoRun } from '../api'
import { Cartao, Distintivo, type StatusTipo } from '../ds/componentes'
import { useAvisos } from '../ds/avisos'

const ROTULO_DIAGNOSTICO: Record<string, StatusTipo> = {
  estavel: 'ok',
  ainda_caindo: 'info',
  estagnado: 'aviso',
  deteriorando: 'aviso',
  overfit: 'perigo',
  instavel: 'perigo',
  divergiu: 'perigo',
  quantizacao: 'info',
  sem_metricas: 'info',
  ausente: 'info',
}

function numero(valor: number | null, casas = 4): string {
  return valor === null || valor === undefined ? '—' : valor.toFixed(casas)
}

function comSinal(valor: number | null): string {
  return valor === null || valor === undefined ? '—' : (valor >= 0 ? '+' : '') + valor.toFixed(4)
}

function inteiro(valor: number | null): string {
  return valor === null || valor === undefined ? '—' : valor.toLocaleString('pt-BR')
}

/** Página de comparação (B2/B3): tabela, veredito e curvas dos runs escolhidos. */
export function Comparar() {
  const { avisar } = useAvisos()
  const [runs, setRuns] = useState<EstadoRun[]>([])
  const [escolhidos, setEscolhidos] = useState<string[]>([])
  const [resultado, setResultado] = useState<ResultadoComparacao | null>(null)
  const [ocupado, setOcupado] = useState(false)

  useEffect(() => {
    api
      .corre()
      .then((lista) => {
        setRuns(lista)
        setEscolhidos(lista.map((r) => r.run_id))
      })
      .catch((e: Error) => avisar(e.message, true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const alternar = (id: string) => {
    setEscolhidos((atual) => (atual.includes(id) ? atual.filter((x) => x !== id) : [...atual, id]))
  }

  const comparar = async () => {
    setOcupado(true)
    try {
      setResultado(await api.comparar(escolhidos))
    } catch (e) {
      avisar(String((e as Error).message), true)
      setResultado(null)
    } finally {
      setOcupado(false)
    }
  }

  const linhas: ResumoRun[] = resultado?.runs ?? []

  return (
    <>
      <Cartao titulo="Escolher runs">
        {runs.length === 0 ? <p className="suave">nenhum run registrado ainda.</p> : null}
        <div className="linha" style={{ flexWrap: 'wrap', gap: 12 }}>
          {runs.map((r) => (
            <label key={r.run_id} className="linha" style={{ gap: 6 }}>
              <input
                type="checkbox"
                aria-label={`incluir ${r.run_id}`}
                checked={escolhidos.includes(r.run_id)}
                onChange={() => alternar(r.run_id)}
              />
              <span className="mono">{r.run_id}</span>
            </label>
          ))}
        </div>
        <button className="botao botao--primario" disabled={ocupado || escolhidos.length === 0} onClick={comparar}>
          {ocupado ? '…' : 'Comparar'}
        </button>
      </Cartao>

      {resultado ? (
        <>
          <Cartao titulo="Comparação">
            <table className="tabela">
              <thead>
                <tr>
                  <th>run</th>
                  <th>família</th>
                  <th>passos</th>
                  <th>params</th>
                  <th>melhor val</th>
                  <th>passo</th>
                  <th>final</th>
                  <th>deriva</th>
                  <th>val@comum</th>
                  <th>veredito</th>
                </tr>
              </thead>
              <tbody>
                {linhas.map((r) => (
                  <tr key={r.run_id}>
                    <td className="mono">{r.run_id}</td>
                    <td className="mono">{r.familia}</td>
                    <td className="mono">{r.passos || '—'}</td>
                    <td className="mono">{inteiro(r.parametros)}</td>
                    <td className="mono">{numero(r.melhor_val)}</td>
                    <td className="mono">{r.passo_melhor_val ?? '—'}</td>
                    <td className="mono">{numero(r.val_final)}</td>
                    <td className="mono">{comSinal(r.drift)}</td>
                    <td className="mono">{numero(r.melhor_val_orcamento_comum)}</td>
                    <td>
                      <Distintivo status={ROTULO_DIAGNOSTICO[r.diagnostico] ?? 'info'}>{r.diagnostico}</Distintivo>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {resultado.orcamento_comum ? (
              <p className="suave">
                val@comum compara todos no orçamento de {resultado.orcamento_comum} passos — sem confundir
                "treinou mais" com "treinou melhor".
              </p>
            ) : null}
            <p>{resultado.veredito}</p>
          </Cartao>

          <Cartao titulo="Diagnóstico por run">
            <ul>
              {linhas
                .filter((r) => r.comentario)
                .map((r) => (
                  <li key={r.run_id}>
                    <strong className="mono">{r.run_id}</strong> ({r.diagnostico}): {r.comentario}
                  </li>
                ))}
            </ul>
          </Cartao>

          <Cartao titulo="Curvas">
            <img
              src={api.urlCurvas(escolhidos)}
              alt="curvas de perda e learning rate dos runs comparados"
              style={{ width: '100%', background: '#fff', borderRadius: 'var(--raio-suave)' }}
            />
          </Cartao>
        </>
      ) : null}
    </>
  )
}
