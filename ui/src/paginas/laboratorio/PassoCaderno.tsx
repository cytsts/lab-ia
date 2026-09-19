import { useState, type ReactNode } from 'react'
import { Distintivo, type StatusTipo } from '../../ds/componentes'

export type EstadoPasso = 'parado' | 'rodando' | 'concluido' | 'falhou'

interface PassoCadernoProps {
  numero: number
  titulo: string
  comandoCli: string
  estado?: EstadoPasso
  mensagemEstado?: string
  concluido?: boolean
  bloqueado?: boolean
  mensagemBloqueio?: string
  children: ReactNode
  resultado?: ReactNode
  acaoRotulo?: string
  acaoDesabilitada?: boolean
  aoExecutar?: () => void
}

export function PassoCaderno({
  numero,
  titulo,
  comandoCli,
  estado = 'parado',
  mensagemEstado,
  bloqueado = false,
  mensagemBloqueio,
  children,
  resultado,
  acaoRotulo,
  acaoDesabilitada = false,
  aoExecutar,
}: PassoCadernoProps) {
  const [copiado, setCopiado] = useState(false)

  const copiar = async () => {
    try {
      await navigator.clipboard.writeText(comandoCli)
      setCopiado(true)
      setTimeout(() => setCopiado(false), 2000)
    } catch {
      /* fallback se permissão for restrita */
    }
  }

  const mapaDistintivo: Record<EstadoPasso, { status: StatusTipo; texto: string }> = {
    parado: { status: 'info', texto: 'não executado' },
    rodando: { status: 'aviso', texto: 'rodando' },
    concluido: { status: 'ok', texto: 'concluído' },
    falhou: { status: 'perigo', texto: 'falhou' },
  }

  const badge = mapaDistintivo[estado]

  return (
    <section className="cartao passo-caderno" style={{ marginBottom: 18, borderLeft: '4px solid var(--primaria)' }}>
      <header className="linha" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
        <h2 style={{ fontSize: '1.05rem', margin: 0 }}>
          <span className="mono" style={{ color: 'var(--primaria)', marginRight: 6 }}>
            Passo {numero}.
          </span>
          {titulo}
        </h2>
        <div className="linha" style={{ gap: 8 }}>
          {mensagemEstado ? <span className="suave" style={{ fontSize: 'var(--miudo)' }}>{mensagemEstado}</span> : null}
          <Distintivo status={badge.status}>{badge.texto}</Distintivo>
        </div>
      </header>

      {/* Linha do comando CLI equivalente (RF1.2 e CA17) */}
      <div
        className="linha"
        style={{
          background: 'var(--superficie-2, #181c24)',
          border: '1px solid var(--borda)',
          borderRadius: 6,
          padding: '4px 8px',
          marginBottom: 12,
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <code className="mono" style={{ fontSize: '0.8rem', overflowX: 'auto', flex: 1, marginRight: 8 }}>
          <span style={{ color: 'var(--serie-c, #38bdf8)', userSelect: 'none' }}>$ </span>
          {comandoCli}
        </code>
        <button
          type="button"
          className="botao"
          onClick={copiar}
          style={{ padding: '2px 8px', fontSize: '0.75rem', minWidth: 64 }}
          title="Copiar comando para reproduzir no terminal"
        >
          {copiado ? '✓ Copiado' : 'Copiar'}
        </button>
      </div>

      {bloqueado ? (
        <div
          className="aviso aviso--erro"
          role="alert"
          style={{ marginBottom: 12, padding: '8px 12px', fontSize: 'var(--miudo)' }}
        >
          ⚠ {mensagemBloqueio || 'Execute o passo anterior para liberar esta etapa.'}
        </div>
      ) : null}

      <div className="passo-conteudo" style={{ opacity: bloqueado ? 0.6 : 1, pointerEvents: bloqueado ? 'none' : 'auto' }}>
        {children}

        {aoExecutar && acaoRotulo ? (
          <div style={{ marginTop: 12 }}>
            <button
              type="button"
              className="botao botao--primario"
              onClick={aoExecutar}
              disabled={bloqueado || acaoDesabilitada || estado === 'rodando'}
            >
              {estado === 'rodando' ? '… Rodando' : acaoRotulo}
            </button>
          </div>
        ) : null}

        {resultado ? (
          <div
            className="passo-resultado"
            style={{
              marginTop: 14,
              paddingTop: 12,
              borderTop: '1px dashed var(--borda)',
            }}
          >
            <div className="linha" style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: '0.85rem', color: 'var(--tinta-suave)' }}>Resultado da etapa:</strong>
            </div>
            {resultado}
          </div>
        ) : null}
      </div>
    </section>
  )
}
