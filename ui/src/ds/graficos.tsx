export interface Serie {
  nome: string
  cor: string
  pontos: { x: number; y: number }[]
}

/** Gráfico de linhas em SVG puro (sem libs): curvas de perda e métricas. */
export function GraficoDeMetricas(props: {
  titulo: string
  series: Serie[]
  largura?: number
  altura?: number
}) {
  const largura = props.largura ?? 560
  const altura = props.altura ?? 200
  const margem = { c: 38, b: 24, t: 8, e: 8 }
  const todas = props.series.flatMap((s) => s.pontos)
  if (todas.length === 0) return <p className="suave">sem dados ainda</p>

  const xMin = Math.min(...todas.map((p) => p.x))
  const xMax = Math.max(...todas.map((p) => p.x))
  const yMin = Math.min(...todas.map((p) => p.y))
  const yMax = Math.max(...todas.map((p) => p.y))
  const ySpan = yMax - yMin || 1
  const xSpan = xMax - xMin || 1
  const px = (x: number) => margem.c + ((x - xMin) / xSpan) * (largura - margem.c - margem.e)
  const py = (y: number) => altura - margem.b - ((y - yMin) / ySpan) * (altura - margem.b - margem.t)

  return (
    <figure style={{ margin: 0 }}>
      <svg
        viewBox={`0 0 ${largura} ${altura}`}
        role="img"
        aria-label={props.titulo}
        style={{ width: '100%', height: 'auto' }}
      >
        {[0, 0.5, 1].map((f) => {
          const y = yMin + f * ySpan
          return (
            <g key={f}>
              <line x1={margem.c} x2={largura - margem.e} y1={py(y)} y2={py(y)} stroke="var(--borda)" />
              <text x={2} y={py(y) + 4} fontSize={10} fill="var(--tinta-suave)">
                {y.toFixed(ySpan < 1 ? 2 : 1)}
              </text>
            </g>
          )
        })}
        {props.series.map((s) => (
          <polyline
            key={s.nome}
            fill="none"
            stroke={s.cor}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            points={s.pontos.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')}
          />
        ))}
      </svg>
      <figcaption className="linha" style={{ fontSize: 'var(--miudo)' }}>
        <strong>{props.titulo}</strong>
        {props.series.map((s) => (
          <span key={s.nome} className="linha" style={{ gap: 4 }}>
            <svg width={14} height={8} aria-hidden>
              <line x1={0} x2={14} y1={4} y2={4} stroke={s.cor} strokeWidth={3} />
            </svg>
            {s.nome}
          </span>
        ))}
      </figcaption>
    </figure>
  )
}

/** Barras horizontais de uso por especialista (MoE). */
export function BarrasDeUso(props: { uso: number[] }) {
  return (
    <div aria-label="uso por especialista">
      {props.uso.map((v, i) => (
        <div key={i} className="linha" style={{ gap: 8, marginBottom: 4 }}>
          <span className="mono" style={{ width: 64 }}>
            esp. {i}
          </span>
          <div className="barra-progresso" style={{ flex: 1 }}>
            <div style={{ width: `${Math.min(100, v * 100)}%`, background: 'var(--serie-c)' }} />
          </div>
          <span className="mono">{(v * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
