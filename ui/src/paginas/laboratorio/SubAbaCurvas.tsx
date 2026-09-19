import { useEffect, useState } from 'react'
import {
  api,
  type EstadoRun,
  type RegistroMetrica,
  type ResumoRun,
} from '../../api'
import { Distintivo } from '../../ds/componentes'
import { BarrasDeUso, GraficoDeMetricas, type Serie } from '../../ds/graficos'
import { PassoCaderno } from './PassoCaderno'

export function SubAbaCurvas() {
  const [runs, setRuns] = useState<EstadoRun[]>([])
  const [runId, setRunId] = useState<string>('logica-1')
  const [metricas, setMetricas] = useState<RegistroMetrica[]>([])
  const [metricasBase, setMetricasBase] = useState<RegistroMetrica[]>([])
  const [resumoRun, setResumoRun] = useState<ResumoRun | null>(null)
  const [carregando, setCarregando] = useState<boolean>(false)

  // Carregar lista de runs
  useEffect(() => {
    api.corre().then((r) => {
      setRuns(r)
      if (r.length > 0 && !r.some((x) => x.run_id === runId)) {
        setRunId(r[r.length - 1].run_id)
      }
    })
  }, [])

  // Carregar métricas do run e diagnóstico comparativo
  useEffect(() => {
    if (!runId) return
    setCarregando(true)
    Promise.all([
      api.metricas(runId).catch(() => []),
      api.comparar([runId]).catch(() => null),
    ]).then(([m, comp]) => {
      setMetricas(m)
      if (comp && comp.runs && comp.runs.length > 0) {
        setResumoRun(comp.runs[0])
      } else {
        setResumoRun(null)
      }

      // Se for run de ajuste, tenta carregar run base esmaecido (RF4.5)
      const estadoAtual = runs.find((r) => r.run_id === runId)
      const baseId = (estadoAtual as Record<string, unknown>)?.base as string | undefined
      if (baseId) {
        api.metricas(baseId).then(setMetricasBase).catch(() => setMetricasBase([]))
      } else {
        setMetricasBase([])
      }
    }).finally(() => setCarregando(false))
  }, [runId, runs])

  // Preparação das séries para o gráfico SVG (RF4.1 e RF4.5)
  const seriesPerda: Serie[] = []

  // Se houver base esmaecida (RF4.5)
  if (metricasBase.length > 0) {
    const ptsBase = metricasBase
      .filter((m) => m.passo !== null && m.loss_val !== null && m.loss_val !== undefined)
      .map((m) => ({ x: m.passo as number, y: m.loss_val as number }))
    if (ptsBase.length > 0) {
      seriesPerda.push({
        nome: 'validação (base)',
        cor: 'rgba(156, 163, 175, 0.4)',
        pontos: ptsBase,
      })
    }
  }

  // Série treino
  const ptsTrem = metricas
    .filter((m) => m.passo !== null && m.loss_trem !== null && m.loss_trem !== undefined)
    .map((m) => ({ x: m.passo as number, y: m.loss_trem as number }))
  if (ptsTrem.length > 0) {
    seriesPerda.push({
      nome: 'treino',
      cor: 'var(--serie-b, #f59e0b)',
      pontos: ptsTrem,
    })
  }

  // Série validação
  const ptsVal = metricas
    .filter((m) => m.passo !== null && m.loss_val !== null && m.loss_val !== undefined)
    .map((m) => ({ x: m.passo as number, y: m.loss_val as number }))
  if (ptsVal.length > 0) {
    seriesPerda.push({
      nome: 'validação',
      cor: 'var(--primaria, #38bdf8)',
      pontos: ptsVal,
    })
  }

  // Séries secundárias: lr e tokens/s (RF4.3)
  const ptsLr = metricas
    .filter((m) => m.passo !== null && m.lr !== undefined)
    .map((m) => ({ x: m.passo as number, y: (m.lr as number) * 1000 })) // em 1e-3

  const ptsVel = metricas
    .filter((m) => m.passo !== null && m.tokens_por_s !== undefined)
    .map((m) => ({ x: m.passo as number, y: m.tokens_por_s as number }))

  // Último uso de especialistas MoE se houver (RF4.3)
  const ultimoMoE = [...metricas].reverse().find((m) => m.uso_especialistas && m.uso_especialistas.length > 0)

  // Marcações de melhor validação, final e deriva (RF4.2)
  let melhorValPonto: { passo: number; val: number } | null = null
  let finalVal: number | null = null
  let drift: number | null = null

  if (ptsVal.length > 0) {
    finalVal = ptsVal[ptsVal.length - 1].y
    let minVal = ptsVal[0].y
    let minPasso = ptsVal[0].x
    for (const p of ptsVal) {
      if (p.y < minVal) {
        minVal = p.y
        minPasso = p.x
      }
    }
    melhorValPonto = { passo: minPasso, val: minVal }
    drift = finalVal - minVal
  }

  const comandoCli = `lab-ia comparar --runs ${runId}`

  return (
    <div className="sub-aba-curvas">
      <PassoCaderno
        numero={1}
        titulo="Escolher o run para análise das curvas de convergência"
        comandoCli={comandoCli}
        estado="concluido"
        mensagemEstado={`analisando run '${runId}'`}
      >
        <div className="linha" style={{ gap: 10, alignItems: 'center', marginBottom: 12 }}>
          <label htmlFor="select-run-curva" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
            Run:
          </label>
          <select
            id="select-run-curva"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            style={{ padding: '6px 10px', minWidth: 220 }}
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.tipo || 'treino'})
              </option>
            ))}
          </select>
        </div>

        {/* Resumo e Diagnóstico Textual (RF4.2 e RF4.4) */}
        {melhorValPonto ? (
          <div
            style={{
              background: 'var(--superficie, #1e293b)',
              padding: 12,
              borderRadius: 6,
              marginBottom: 14,
            }}
          >
            <div className="linha" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <div>
                <strong>Ponto de Melhor Validação:</strong>{' '}
                <span className="mono" style={{ color: 'var(--primaria)' }}>
                  {melhorValPonto.val.toFixed(4)}
                </span>{' '}
                (no passo {melhorValPonto.passo})
              </div>

              <div>
                <strong>Valor Final:</strong>{' '}
                <span className="mono">{finalVal?.toFixed(4) ?? '—'}</span>
              </div>

              <div>
                <strong>Deriva (Drift):</strong>{' '}
                <span
                  className="mono"
                  style={{
                    color: drift !== null && drift > 0.1 ? 'var(--perigo, #ef4444)' : 'inherit',
                  }}
                >
                  {drift !== null ? (drift >= 0 ? `+${drift.toFixed(4)}` : drift.toFixed(4)) : '—'}
                </span>
              </div>

              {resumoRun?.diagnostico ? (
                <div>
                  <strong>Diagnóstico:</strong>{' '}
                  <Distintivo
                    status={
                      resumoRun.diagnostico === 'overfit' || resumoRun.diagnostico === 'divergiu'
                        ? 'perigo'
                        : resumoRun.diagnostico === 'estavel'
                        ? 'ok'
                        : 'aviso'
                    }
                  >
                    {resumoRun.diagnostico}
                  </Distintivo>
                </div>
              ) : null}
            </div>

            {resumoRun?.comentario ? (
              <p className="suave" style={{ fontSize: '0.8rem', marginTop: 6, marginBottom: 0 }}>
                {resumoRun.comentario}
              </p>
            ) : null}
          </div>
        ) : null}

        {/* Gráfico Principal de Perda (RF4.1) */}
        {carregando ? (
          <p className="suave">Carregando métricas...</p>
        ) : seriesPerda.length > 0 ? (
          <div style={{ marginBottom: 18 }}>
            <GraficoDeMetricas
              titulo={`Curva de Perda por Passo — ${runId}`}
              series={seriesPerda}
              altura={240}
            />
          </div>
        ) : (
          <p className="suave" style={{ padding: 14 }}>
            Nenhuma métrica de perda encontrada para o run '{runId}'.
          </p>
        )}

        {/* Painéis Secundários (RF4.3): lr e tokens/s */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
          {ptsLr.length > 0 ? (
            <div style={{ border: '1px solid var(--borda)', padding: 10, borderRadius: 6 }}>
              <GraficoDeMetricas
                titulo="Taxa de Aprendizado (lr × 1e-3)"
                series={[{ nome: 'lr', cor: 'var(--serie-d, #a855f7)', pontos: ptsLr }]}
                altura={140}
              />
            </div>
          ) : null}

          {ptsVel.length > 0 ? (
            <div style={{ border: '1px solid var(--borda)', padding: 10, borderRadius: 6 }}>
              <GraficoDeMetricas
                titulo="Velocidade (tokens/s)"
                series={[{ nome: 'tok/s', cor: 'var(--serie-c, #10b981)', pontos: ptsVel }]}
                altura={140}
              />
            </div>
          ) : null}
        </div>

        {/* Painel de Especialistas MoE se houver (RF4.3) */}
        {ultimoMoE?.uso_especialistas ? (
          <div style={{ marginTop: 14, border: '1px solid var(--borda)', padding: 12, borderRadius: 6 }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: 8 }}>
              Distribuição de Uso dos Especialistas (MoE)
            </h3>
            <BarrasDeUso uso={ultimoMoE.uso_especialistas} />
          </div>
        ) : null}
      </PassoCaderno>
    </div>
  )
}
