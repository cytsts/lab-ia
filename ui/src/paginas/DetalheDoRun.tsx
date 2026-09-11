import { useEffect, useState } from 'react'
import { api, type EstadoRun, type RegistroMetrica } from '../api'
import { Cartao, Distintivo } from '../ds/componentes'
import { BarrasDeUso, GraficoDeMetricas } from '../ds/graficos'
import { etiquetaEstado } from './Painel'

function bytes(n: number): string {
  return n > 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : `${(n / 2 ** 10).toFixed(1)} kB`
}

export function DetalheDoRun(props: { id: string }) {
  const [estado, setEstado] = useState<EstadoRun | null>(null)
  const [metricas, setMetricas] = useState<RegistroMetrica[]>([])
  const [tamanhos, setTamanhos] = useState<Record<string, unknown> | null>(null)
  const [comparativo, setComparativo] = useState<Record<string, unknown> | null>(null)
  const [erro, setErro] = useState<string | null>(null)

  useEffect(() => {
    setErro(null)
    api
      .estado(props.id)
      .then(setEstado)
      .catch((e: Error) => setErro(e.message))
    api.metricas(props.id).then(setMetricas).catch(() => setMetricas([]))
    api.tamanhos(props.id).then(setTamanhos).catch(() => setTamanhos(null))
    api.comparativo(props.id).then(setComparativo).catch(() => setComparativo(null))
  }, [props.id])

  if (erro) return <Cartao titulo={props.id}><p style={{ color: 'var(--perigo)' }}>{erro}</p></Cartao>

  const treino = metricas.filter((m) => typeof m.passo === 'number' && m.loss_val != null)
  const serie = (nome: string, chave: 'loss_trem' | 'loss_val', cor: string) => ({
    nome,
    cor,
    pontos: treino
      .filter((m) => m[chave] != null)
      .map((m) => ({ x: m.passo as number, y: m[chave] as number })),
  })
  const series = [
    serie('perda (treino)', 'loss_trem', 'var(--serie-a)'),
    serie('perda (validação)', 'loss_val', 'var(--serie-b)'),
  ].filter((s) => s.pontos.length > 0)

  const uso = [...treino].reverse().find((m) => m.uso_especialistas)?.uso_especialistas

  return (
    <>
      <Cartao
        titulo={
          <span className="linha">
            {props.id}{' '}
            {estado ? (
              <Distintivo status={etiquetaEstado(estado).status}>{etiquetaEstado(estado).texto}</Distintivo>
            ) : null}
          </span>
        }
      >
        {series.length ? (
          <GraficoDeMetricas titulo="Curva de perda" series={series} />
        ) : (
          <p className="suave">sem métricas de treino neste run.</p>
        )}
        {uso ? <BarrasDeUso uso={uso} /> : null}
      </Cartao>

      {tamanhos ? (
        <Cartao titulo="Quantização">
          <table className="tabela">
            <tbody>
              <tr><td>modo</td><td className="mono">{String(tamanhos['modo'])}</td></tr>
              <tr><td>fator (alvos)</td><td className="mono">{Number(tamanhos['fator_alvos']).toFixed(2)}×</td></tr>
              <tr><td>bits efetivos</td><td className="mono">{String(tamanhos['bits_efetivos_por_parametro'])}</td></tr>
              <tr><td>fp32 → quantizado</td><td className="mono">{bytes(Number(tamanhos['bytes_fp32_alvos']))} → {bytes(Number(tamanhos['bytes_quantizados_alvos']))}</td></tr>
              <tr><td>perda antes → depois</td><td className="mono">{Number(tamanhos['perda_val_antes']).toFixed(3)} → {Number(tamanhos['perda_val_depois']).toFixed(3)}</td></tr>
            </tbody>
          </table>
        </Cartao>
      ) : null}

      {comparativo ? (
        <Cartao titulo="Benchmark de raciocínio">
          <pre className="mono">{JSON.stringify(comparativo, null, 1)}</pre>
        </Cartao>
      ) : null}
    </>
  )
}
