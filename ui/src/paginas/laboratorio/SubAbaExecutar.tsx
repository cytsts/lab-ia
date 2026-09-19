import { useEffect, useRef, useState } from 'react'
import {
  api,
  type ProgressoRun,
} from '../../api'
import { BarraProgresso, Distintivo } from '../../ds/componentes'
import { PassoCaderno } from './PassoCaderno'

interface SubAbaExecutarProps {
  configPreSelecionada?: string
}

export function SubAbaExecutar({ configPreSelecionada }: SubAbaExecutarProps) {
  const [configs, setConfigs] = useState<string[]>([])
  const [config, setConfig] = useState<string>(configPreSelecionada || '')
  const [acao, setAcao] = useState<'train' | 'ajustar'>('train')
  const [chaveAtiva, setChaveAtiva] = useState<string>('')
  const [execucoes, setExecucoes] = useState<
    { chave: string; pid: number; vivo: boolean; codigo_saida: number | null }[]
  >([])
  const [linhasLog, setLinhasLog] = useState<string[]>([])
  const [progresso, setProgresso] = useState<ProgressoRun | null>(null)
  const [ocupado, setOcupado] = useState<boolean>(false)
  const [confirmandoParar, setConfirmandoParar] = useState<boolean>(false)
  const [avisoFila, setAvisoFila] = useState<boolean>(false)

  const fimDoLogRef = useRef<HTMLDivElement | null>(null)

  // Atualiza config se vier via prop
  useEffect(() => {
    if (configPreSelecionada) {
      setConfig(configPreSelecionada)
    }
  }, [configPreSelecionada])

  // Carregar configs disponíveis e execuções ativas
  const recarregar = async () => {
    try {
      const [cfgs, execs] = await Promise.all([api.configs(), api.execucoes()])
      setConfigs(cfgs)
      setExecucoes(execs)
      if (!config && cfgs.length > 0) {
        setConfig(cfgs[0])
      }
      // Se houver alguma execução viva, foca nela
      const viva = execs.find((x) => x.vivo)
      if (viva && !chaveAtiva) {
        setChaveAtiva(viva.chave)
      } else if (!viva && !chaveAtiva && execs.length > 0) {
        setChaveAtiva(execs[execs.length - 1].chave)
      }
    } catch {
      /* núcleo pode estar subindo */
    }
  }

  useEffect(() => {
    recarregar()
    const t = setInterval(recarregar, 3000)
    return () => clearInterval(t)
  }, [])

  // Deduz o run_id da chave de execução ou config
  const runIdAtual = chaveAtiva
    ? chaveAtiva.replace(/^(train|ajustar|benchmark)-/, '').replace(/\.yaml$/, '')
    : config.replace(/\.yaml$/, '')

  // Polling de 2 s do log e do progresso (RF3.2, RF3.3, RF3.4, RNF5)
  useEffect(() => {
    if (!chaveAtiva) return

    const atualizarLogEProgresso = () => {
      // Log ao vivo (200 linhas)
      api.logExecucao(chaveAtiva, 200)
        .then((res) => {
          if (res.existe) {
            setLinhasLog(res.linhas)
          }
        })
        .catch(() => {})

      // Progresso do run
      if (runIdAtual) {
        api.progresso(runIdAtual)
          .then(setProgresso)
          .catch(() => {})
      }
    }

    atualizarLogEProgresso()
    const t = setInterval(atualizarLogEProgresso, 2000)
    return () => clearInterval(t)
  }, [chaveAtiva, runIdAtual])

  // Auto-scroll do log
  useEffect(() => {
    fimDoLogRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [linhasLog])

  // Disparar execução
  const disparar = async (forcar = false) => {
    const temExecucaoViva = execucoes.some((x) => x.vivo)
    if (temExecucaoViva && !forcar) {
      setAvisoFila(true)
      return
    }
    setAvisoFila(false)
    setOcupado(true)
    try {
      const res = await api.executar(acao, config)
      setChaveAtiva(res.chave)
      setLinhasLog([])
      await recarregar()
    } finally {
      setOcupado(false)
    }
  }

  // Parar execução (RF3.7, CA8)
  const parar = async () => {
    if (!chaveAtiva) return
    setOcupado(true)
    try {
      await api.pararExecucao(chaveAtiva)
      setConfirmandoParar(false)
      await recarregar()
    } finally {
      setOcupado(false)
    }
  }

  // Retomar execução (RF3.7, CA8)
  const retomar = async () => {
    // Para retomar, executa o comando com flag retomar via CLI ou API
    await disparar(true)
  }

  const execucaoAtual = execucoes.find((x) => x.chave === chaveAtiva)
  const estaVivo = execucaoAtual ? execucaoAtual.vivo : false

  const comandoCli = `lab-ia ${acao} --config configs/${config || '<config>.yaml'}`

  return (
    <div className="sub-aba-executar">
      {/* Passo 1: Configurar e disparar execução */}
      <PassoCaderno
        numero={1}
        titulo="Lançar ou retomar treinamento"
        comandoCli={comandoCli}
        estado={estaVivo ? 'rodando' : progresso?.concluido ? 'concluido' : 'parado'}
        mensagemEstado={
          estaVivo
            ? `PID ${execucaoAtual?.pid} ativo`
            : execucaoAtual?.codigo_saida === 0
            ? 'execução finalizada com sucesso'
            : execucaoAtual
            ? `código de saída: ${execucaoAtual.codigo_saida}`
            : 'pronto para disparar'
        }
      >
        <div className="linha" style={{ gap: 14, flexWrap: 'wrap', marginBottom: 12 }}>
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-acao" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>
              Tipo:
            </label>
            <select
              id="select-acao"
              value={acao}
              onChange={(e) => setAcao(e.target.value as 'train' | 'ajustar')}
              style={{ padding: '6px 10px' }}
            >
              <option value="train">treinar do zero (train)</option>
              <option value="ajustar">refinar LoRA (ajustar)</option>
            </select>
          </div>

          <div className="linha" style={{ gap: 6, flex: 1 }}>
            <label htmlFor="select-config" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>
              Configuração:
            </label>
            <select
              id="select-config"
              value={config}
              onChange={(e) => setConfig(e.target.value)}
              style={{ padding: '6px 10px', width: '100%' }}
            >
              {configs.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Aviso de concorrência de GPU (RF3.8) */}
        {avisoFila ? (
          <div
            className="aviso aviso--erro"
            style={{ marginBottom: 12, padding: 10, borderRadius: 6 }}
          >
            ⚠ <strong>Aviso de GPU Única:</strong> Já existe uma execução em andamento (PID{' '}
            {execucoes.find((x) => x.vivo)?.pid}). Se você disparar outra agora, as duas irão competir
            pela VRAM e tempo de processamento.
            <div className="linha" style={{ gap: 8, marginTop: 8 }}>
              <button
                type="button"
                className="botao botao--primario"
                onClick={() => disparar(true)}
              >
                Prosseguir mesmo assim
              </button>
              <button
                type="button"
                className="botao"
                onClick={() => setAvisoFila(false)}
              >
                Cancelar
              </button>
            </div>
          </div>
        ) : null}

        {/* Botões de controle */}
        <div className="linha" style={{ gap: 10, alignItems: 'center' }}>
          {!estaVivo ? (
            <button
              type="button"
              className="botao botao--primario"
              onClick={() => disparar(false)}
              disabled={ocupado || !config}
            >
              ▶ Iniciar Treino
            </button>
          ) : (
            <>
              {!confirmandoParar ? (
                <button
                  type="button"
                  className="botao"
                  style={{ background: 'var(--perigo, #ef4444)', color: '#fff' }}
                  onClick={() => setConfirmandoParar(true)}
                  disabled={ocupado}
                >
                  ⏹ PARAR Execução
                </button>
              ) : (
                <div className="linha" style={{ gap: 8, alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--perigo)' }}>
                    Confirmar parada? O run ficará pausado.
                  </span>
                  <button
                    type="button"
                    className="botao"
                    style={{ background: 'var(--perigo, #ef4444)', color: '#fff' }}
                    onClick={parar}
                    disabled={ocupado}
                  >
                    Sim, Parar Agora
                  </button>
                  <button
                    type="button"
                    className="botao"
                    onClick={() => setConfirmandoParar(false)}
                  >
                    Voltar
                  </button>
                </div>
              )}
            </>
          )}

          {!estaVivo && progresso && !progresso.concluido && progresso.passo > 0 ? (
            <button
              type="button"
              className="botao"
              onClick={retomar}
              disabled={ocupado}
              title="Continuar do último checkpoint gravado"
            >
              ↻ Retomar do checkpoint ({progresso.passo}/{progresso.passos_totais})
            </button>
          ) : null}
        </div>
      </PassoCaderno>

      {/* Passo 2: Monitoramento ao vivo (Barra, Perda, Métricas e Log) */}
      <PassoCaderno
        numero={2}
        titulo="Acompanhamento ao vivo (métrica, progresso e console)"
        comandoCli={`lab-ia comparar --runs ${runIdAtual || 'run'}`}
        estado={estaVivo ? 'rodando' : progresso?.concluido ? 'concluido' : 'parado'}
      >
        {/* Barra de Progresso e Métricas Rápidas */}
        {progresso ? (
          <div
            style={{
              background: 'var(--superficie, #1e293b)',
              padding: 12,
              borderRadius: 6,
              marginBottom: 14,
            }}
          >
            <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 6 }}>
              <span className="mono" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>
                Passo {progresso.passo} de {progresso.passos_totais}{' '}
                {progresso.passos_totais > 0
                  ? `(${((progresso.passo / progresso.passos_totais) * 100).toFixed(1)}%)`
                  : ''}
              </span>
              <div className="linha" style={{ gap: 8 }}>
                {progresso.tempo_s ? (
                  <span className="suave" style={{ fontSize: '0.8rem' }}>
                    decorrido: {Math.round(progresso.tempo_s)}s
                  </span>
                ) : null}
                {progresso.tempo_restante_s ? (
                  <span className="suave" style={{ fontSize: '0.8rem' }}>
                    restante: ~{Math.round(progresso.tempo_restante_s)}s
                  </span>
                ) : null}
              </div>
            </div>

            <BarraProgresso atual={progresso.passo} total={progresso.passos_totais} />

            <div
              className="linha"
              style={{
                justifyContent: 'space-between',
                marginTop: 10,
                fontSize: '0.85rem',
                flexWrap: 'wrap',
              }}
            >
              <span>
                <strong>Perda Treino:</strong>{' '}
                <span className="mono">{progresso.loss_trem?.toFixed(4) ?? '—'}</span>
              </span>
              <span>
                <strong>Perda Validação:</strong>{' '}
                <span className="mono" style={{ color: 'var(--primaria)' }}>
                  {progresso.loss_val?.toFixed(4) ?? '—'}
                </span>
              </span>
              <span>
                <strong>lr:</strong> <span className="mono">{progresso.lr ?? '—'}</span>
              </span>
              <span>
                <strong>Velocidade:</strong>{' '}
                <span className="mono">
                  {progresso.tokens_por_s ? `${Math.round(progresso.tokens_por_s)} tok/s` : '—'}
                </span>
              </span>
            </div>

            {/* Mini Sparkline SVG de Perda Recente (RF3.4) */}
            {progresso.sparkline && progresso.sparkline.length > 2 ? (
              <div style={{ marginTop: 10, height: 35 }}>
                <svg viewBox="0 0 300 35" style={{ width: '100%', height: '100%' }}>
                  {(() => {
                    const pts = progresso.sparkline.filter((p) => p.loss_val !== null && p.loss_val !== undefined)
                    if (pts.length < 2) return null
                    const minVal = Math.min(...pts.map((p) => p.loss_val as number))
                    const maxVal = Math.max(...pts.map((p) => p.loss_val as number))
                    const span = maxVal - minVal || 1
                    const coords = pts.map((p, i) => {
                      const x = (i / (pts.length - 1)) * 300
                      const y = 32 - (((p.loss_val as number) - minVal) / span) * 28
                      return `${x},${y}`
                    })
                    return (
                      <polyline
                        fill="none"
                        stroke="var(--primaria)"
                        strokeWidth={2}
                        points={coords.join(' ')}
                      />
                    )
                  })()}
                </svg>
              </div>
            ) : null}

            {/* Progresso de Benchmark Parcial se houver (RF3.5, CA14) */}
            {progresso.benchmark_progresso ? (
              <div
                style={{
                  marginTop: 10,
                  padding: 8,
                  background: 'var(--superficie-2, #0f172a)',
                  borderRadius: 4,
                  fontSize: '0.85rem',
                }}
              >
                <div className="linha" style={{ justifyContent: 'space-between' }}>
                  <span>
                    Benchmark: item {progresso.benchmark_progresso.item_atual} de{' '}
                    {progresso.benchmark_progresso.total_itens}
                  </span>
                  <Distintivo status="ok">
                    acurácia parcial:{' '}
                    {(progresso.benchmark_progresso.acuracia_parcial * 100).toFixed(1)}%
                  </Distintivo>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Console de Log ao Vivo (RF3.2) */}
        <div>
          <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>
              Console de saída em tempo real:
            </span>
            <span className="suave" style={{ fontSize: '0.75rem' }}>
              {linhasLog.length} linhas lidas · atualizado a cada 2 s
            </span>
          </div>

          <div
            className="mono"
            style={{
              height: 260,
              overflowY: 'auto',
              background: '#090d13',
              color: '#d1d5db',
              padding: '10px 12px',
              borderRadius: 6,
              fontSize: '0.8rem',
              lineHeight: 1.45,
              border: '1px solid var(--borda)',
            }}
          >
            {linhasLog.length === 0 ? (
              <span className="suave">Nenhum log registrado para a execução atual.</span>
            ) : (
              linhasLog.map((linha, idx) => (
                <div key={idx} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {linha}
                </div>
              ))
            )}
            <div ref={fimDoLogRef} />
          </div>
        </div>
      </PassoCaderno>
    </div>
  )
}
