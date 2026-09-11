import type { CSSProperties, ReactNode } from 'react'

export function Cartao(props: { titulo?: ReactNode; children: ReactNode; estilo?: CSSProperties }) {
  return (
    <section className="cartao" style={props.estilo}>
      {props.titulo ? <h2>{props.titulo}</h2> : null}
      {props.children}
    </section>
  )
}

export type StatusTipo = 'ok' | 'aviso' | 'perigo' | 'info'

export function Distintivo(props: { status: StatusTipo; children: ReactNode }) {
  return <span className={`distintivo distintivo--${props.status}`}>{props.children}</span>
}

export function BarraProgresso(props: { atual: number; total: number }) {
  const pct = props.total > 0 ? Math.min(100, (props.atual / props.total) * 100) : 0
  return (
    <div
      className="barra-progresso"
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="progresso do experimento"
    >
      <div style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Campo(props: { rotulo: string; children: ReactNode }) {
  return (
    <div className="campo">
      <label>{props.rotulo}</label>
      {props.children}
    </div>
  )
}
