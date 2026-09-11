import { useEffect, useMemo, useState } from 'react'
import { api, type EstadoRun } from '../api'
import { BarraProgresso, Cartao, Distintivo, type StatusTipo } from '../ds/componentes'

export function etiquetaEstado(run: EstadoRun): { texto: string; status: StatusTipo } {
  if (run.tipo === 'quantizacao') return { texto: 'quantizado', status: 'info' }
  if (run.concluido === true) return { texto: 'concluído', status: 'ok' }
  if (run.passo) return { texto: 'em andamento', status: 'aviso' }
  return { texto: 'registrado', status: 'info' }
}

const TIPO_BONITO: Record<string, string> = {
  undefined: 'treino',
  quantizacao: 'quantização',
  lora: 'ajuste LoRA',
  qlora: 'ajuste QLoRA',
}

export function Painel(props: { abrirRun(id: string): void }) {
  const [runs, setRuns] = useState<EstadoRun[] | null>(null)
  const [erro, setErro] = useState<string | null>(null)
  const [busca, setBusca] = useState('')

  const carregar = () => {
    api
      .corre()
      .then((r) => {
        setRuns(r)
        setErro(null)
      })
      .catch((e: Error) => setErro(e.message))
  }
  useEffect(carregar, [])

  const filtrados = useMemo(() => {
    if (!runs) return []
    const termo = busca.trim().toLowerCase()
    return termo ? runs.filter((r) => r.run_id.toLowerCase().includes(termo)) : runs
  }, [runs, busca])

  return (
    <Cartao
      titulo={
        <span className="linha" style={{ justifyContent: 'space-between', width: '100%' }}>
          Experimentos
          <button className="botao" onClick={carregar}>
            ↻ Atualizar
          </button>
        </span>
      }
    >
      <input
        className="campo"
        style={{
          width: '100%',
          font: 'inherit',
          padding: '8px 12px',
          borderRadius: 'var(--raio-suave)',
          border: '1px solid var(--borda)',
          background: 'var(--fundo-cartao)',
          color: 'var(--tinta)',
          marginBottom: 16,
        }}
        placeholder="buscar por nome…"
        aria-label="buscar experimento"
        value={busca}
        onChange={(e) => setBusca(e.target.value)}
      />
      {erro ? <p style={{ color: 'var(--perigo)' }}>núcleo indisponível: {erro}</p> : null}
      {!runs && !erro ? <p className="suave">carregando…</p> : null}
      {runs && filtrados.length === 0 ? <p className="suave">nenhum experimento encontrado.</p> : null}
      <div className="grade">
        {filtrados.map((r) => {
          const e = etiquetaEstado(r)
          return (
            <article
              key={r.run_id}
              className="cartao"
              style={{ margin: 0, cursor: 'pointer' }}
              onClick={() => props.abrirRun(r.run_id)}
              onKeyDown={(ev) => ev.key === 'Enter' && props.abrirRun(r.run_id)}
              tabIndex={0}
              role="button"
              aria-label={`abrir run ${r.run_id}`}
            >
              <h2>{r.run_id}</h2>
              <div className="linha" style={{ marginBottom: 8 }}>
                <Distintivo status={e.status}>{e.texto}</Distintivo>
                <span className="mono">{TIPO_BONITO[String(r.tipo)] ?? String(r.tipo)}</span>
                {r.dispositivo ? <span className="mono">{String(r.dispositivo)}</span> : null}
              </div>
              {typeof r.passo === 'number' && typeof r.passos_totais === 'number' ? (
                <>
                  <BarraProgresso atual={r.passo} total={r.passos_totais} />
                  <p className="mono">
                    {r.passo} / {r.passos_totais} passos
                  </p>
                </>
              ) : null}
            </article>
          )
        })}
      </div>
    </Cartao>
  )
}
