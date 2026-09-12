import { useEffect, useState } from 'react'
import { api, type IndiceTrilha, type ResumoLicao } from '../api'
import { Cartao, Distintivo } from '../ds/componentes'
import { useAvisos } from '../ds/avisos'

/** Página da trilha (B6): as lições e os experimentos, dentro da janela do laboratório. */
export function Trilha() {
  const { avisar } = useAvisos()
  const [indice, setIndice] = useState<IndiceTrilha | null>(null)
  const [aberta, setAberta] = useState<ResumoLicao | null>(null)
  const [texto, setTexto] = useState('')
  const [log, setLog] = useState<string[]>([])
  const [chave, setChave] = useState<string | null>(null)
  const [ocupado, setOcupado] = useState(false)

  const recarregar = () => {
    api
      .trilha()
      .then(setIndice)
      .catch((e: Error) => avisar(e.message, true))
  }
  useEffect(recarregar, [])
  // eslint-disable-next-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!aberta) return
    setTexto('')
    api
      .licao(aberta.numero)
      .then(setTexto)
      .catch((e: Error) => avisar(e.message, true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aberta])

  // acompanha o experimento em execução pelo mesmo log das outras execuções
  useEffect(() => {
    if (!chave) return
    const buscarLog = () => {
      api
        .logExecucao(chave, 200)
        .then((r) => setLog(r.linhas))
        .catch(() => {})
    }
    buscarLog() // primeira leitura imediata: sem isso a tela fica vazia por 1,5 s
    const relogio = setInterval(buscarLog, 1500)
    return () => clearInterval(relogio)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chave])

  const rodar = async () => {
    if (!aberta?.tem_experimento) return
    setOcupado(true)
    setLog([])
    try {
      const r = await api.rodarLicao(aberta.numero)
      setChave(r.chave)
      avisar(`rodando ${r.experimento} (pid ${r.pid})`)
    } catch (e) {
      avisar(String((e as Error).message), true)
    } finally {
      setOcupado(false)
    }
  }

  return (
    <>
      <Cartao
        titulo={
          <span className="linha" style={{ justifyContent: 'space-between', width: '100%' }}>
            Trilha de estudo
            {indice && indice.citacoes_quebradas.length === 0 ? (
              <Distintivo status="ok">material em dia com o código</Distintivo>
            ) : null}
            {indice && indice.citacoes_quebradas.length > 0 ? (
              <Distintivo status="aviso">
                {indice.citacoes_quebradas.length} citação(ões) quebrada(s)
              </Distintivo>
            ) : null}
          </span>
        }
      >
        <p className="suave">
          Cada lição tem a ideia, o trecho exato do núcleo que a implementa, um experimento que
          você roda e exercícios com resultado esperado.
        </p>
        {!indice ? <p className="suave">carregando…</p> : null}
        <div className="grade">
          {(indice?.licoes ?? []).map((licao) => (
            <article
              key={licao.numero}
              className="cartao"
              style={{ margin: 0, cursor: 'pointer' }}
              onClick={() => setAberta(licao)}
              onKeyDown={(ev) => ev.key === 'Enter' && setAberta(licao)}
              tabIndex={0}
              role="button"
              aria-label={`abrir lição ${licao.numero}`}
            >
              <h2>
                {licao.numero}. {licao.titulo.replace(/^Lição \d+ — /, '')}
              </h2>
              <p className="suave">{licao.resumo}</p>
              {licao.experimento ? <span className="mono">{licao.experimento.split(/[\\/]/).pop()}</span> : null}
            </article>
          ))}
        </div>
      </Cartao>

      {aberta ? (
        <Cartao
          titulo={
            <span className="linha" style={{ justifyContent: 'space-between', width: '100%' }}>
              {aberta.titulo}
              <span className="linha" style={{ gap: 8 }}>
                {aberta.tem_experimento ? (
                  <button className="botao botao--primario" disabled={ocupado} onClick={rodar}>
                    ▶ Rodar experimento
                  </button>
                ) : null}
                <button className="botao" onClick={() => setAberta(null)}>
                  fechar
                </button>
              </span>
            </span>
          }
        >
          {log.length ? (
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', maxHeight: 320, overflow: 'auto' }}>
              {log.join('\n')}
            </pre>
          ) : null}
          <pre className="mono" style={{ whiteSpace: 'pre-wrap' }} aria-label="texto da lição">
            {texto || 'carregando…'}
          </pre>
        </Cartao>
      ) : null}
    </>
  )
}
