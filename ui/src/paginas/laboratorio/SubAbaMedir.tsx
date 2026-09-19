import { useEffect, useState } from 'react'
import {
  api,
  type EstadoRun,
  type RelatorioBenchmark,
} from '../../api'
import { Distintivo } from '../../ds/componentes'
import { PassoCaderno } from './PassoCaderno'

export function SubAbaMedir() {
  const [runs, setRuns] = useState<EstadoRun[]>([])
  const [runId, setRunId] = useState<string>('logica-1')
  const [split, setSplit] = useState<'teste' | 'dificil'>('teste')
  const [limite, setLimite] = useState<number>(60)
  const [estrategia, setEstrategia] = useState<'cot' | 'direta' | 'tot'>('cot')

  // Estado do benchmark ativo
  const [chaveBenchmark, setChaveBenchmark] = useState<string>('')
  const [ocupado, setOcupado] = useState<boolean>(false)
  const [progressoParcial, setProgressoParcial] = useState<{
    item_atual: number
    total_itens: number
    acertos: number
    acuracia_parcial: number
    concluido: boolean
  } | null>(null)
  const [relatorioAtual, setRelatorioAtual] = useState<RelatorioBenchmark | null>(null)

  // Comparação entre duas bases (RF6.3)
  const [runCompararA, setRunCompararA] = useState<string>('logica-1')
  const [runCompararB, setRunCompararB] = useState<string>('g5-ajuste-cot')
  const [relatoriosA, setRelatoriosA] = useState<RelatorioBenchmark[]>([])
  const [relatoriosB, setRelatoriosB] = useState<RelatorioBenchmark[]>([])

  // Carregar runs
  useEffect(() => {
    api.corre().then((r) => {
      setRuns(r)
      if (r.length > 0) {
        if (!r.some((x) => x.run_id === runId)) setRunId(r[r.length - 1].run_id)
        if (!r.some((x) => x.run_id === runCompararA)) setRunCompararA(r[0].run_id)
        if (r.length > 1 && !r.some((x) => x.run_id === runCompararB)) setRunCompararB(r[1].run_id)
      }
    })
  }, [])

  // Carregar benchmarks existentes do run
  const carregarBenchmarksSalvos = () => {
    if (!runId) return
    api.benchmarksDoRun(runId).then((rels) => {
      const match = rels.find((rel) => rel.split === split && rel.estrategia === estrategia)
      if (match) {
        setRelatorioAtual(match)
      } else if (rels.length > 0) {
        setRelatorioAtual(rels[0])
      }
    }).catch(() => {})
  }

  useEffect(() => {
    carregarBenchmarksSalvos()
  }, [runId, split, estrategia])

  // Polling de progresso parcial durante a execução (RF3.5 e CA14)
  useEffect(() => {
    if (!chaveBenchmark) return
    const t = setInterval(async () => {
      try {
        const prog = await api.progresso(runId)
        if (prog.benchmark_progresso) {
          setProgressoParcial(prog.benchmark_progresso)
          if (prog.benchmark_progresso.concluido) {
            setChaveBenchmark('')
            carregarBenchmarksSalvos()
          }
        }
      } catch {
        /* se ainda não gerou o progresso */
      }
    }, 2000)
    return () => clearInterval(t)
  }, [chaveBenchmark, runId])

  // Disparar benchmark (RF6.1 / RF6.5)
  const dispararBenchmark = async () => {
    setOcupado(true)
    setProgressoParcial(null)
    try {
      const res = await api.dispararBenchmark({
        run: runId,
        estrategia,
        benchmark: 'data/logica-pq/benchmark.jsonl',
        split,
        limite,
        semente: 1234,
      })
      setChaveBenchmark(res.chave)
    } finally {
      setOcupado(false)
    }
  }

  // Carregar dados de comparação entre A e B (RF6.3)
  useEffect(() => {
    if (runCompararA) {
      api.benchmarksDoRun(runCompararA).then(setRelatoriosA).catch(() => setRelatoriosA([]))
    }
    if (runCompararB) {
      api.benchmarksDoRun(runCompararB).then(setRelatoriosB).catch(() => setRelatoriosB([]))
    }
  }, [runCompararA, runCompararB])

  const relA = relatoriosA.find((r) => r.split === split && r.estrategia === estrategia) || relatoriosA[0]
  const relB = relatoriosB.find((r) => r.split === split && r.estrategia === estrategia) || relatoriosB[0]

  const comandoCli = `lab-ia raciocinio --run ${runId} --benchmark data/logica-pq/benchmark.jsonl --split ${split} --limite ${limite} --estrategia ${estrategia}`

  return (
    <div className="sub-aba-medir">
      {/* Passo 1: Configurar e rodar o benchmark */}
      <PassoCaderno
        numero={1}
        titulo="Configurar e executar medição padronizada de raciocínio"
        comandoCli={comandoCli}
        estado={chaveBenchmark ? 'rodando' : relatorioAtual ? 'concluido' : 'parado'}
        mensagemEstado={
          progressoParcial
            ? `item ${progressoParcial.item_atual} de ${progressoParcial.total_itens} (${(progressoParcial.acuracia_parcial * 100).toFixed(1)}%)`
            : relatorioAtual
            ? `acurácia global: ${(relatorioAtual.acuracia_global * 100).toFixed(1)}%`
            : undefined
        }
      >
        <div className="linha" style={{ gap: 14, flexWrap: 'wrap', marginBottom: 14 }}>
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-run-medir" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
              Base/Run:
            </label>
            <select
              id="select-run-medir"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              style={{ padding: '6px 10px', minWidth: 200 }}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id}
                </option>
              ))}
            </select>
          </div>

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-split-medir" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
              Split:
            </label>
            <select
              id="select-split-medir"
              value={split}
              onChange={(e) => setSplit(e.target.value as 'teste' | 'dificil')}
              style={{ padding: '6px 10px' }}
            >
              <option value="teste">teste (tamanho padrão)</option>
              <option value="dificil">difícil (generalização de comprimento)</option>
            </select>
          </div>

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-estrategia-medir" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
              Estratégia:
            </label>
            <select
              id="select-estrategia-medir"
              value={estrategia}
              onChange={(e) => setEstrategia(e.target.value as 'cot' | 'direta' | 'tot')}
              style={{ padding: '6px 10px' }}
            >
              <option value="cot">CoT (raciocínio passo a passo)</option>
              <option value="direta">Direta (resposta imediata)</option>
              <option value="tot">ToT (árvore de raciocínio)</option>
            </select>
          </div>

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="input-limite-medir" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
              Limite de Itens:
            </label>
            <input
              id="input-limite-medir"
              type="number"
              value={limite}
              onChange={(e) => setLimite(Number(e.target.value))}
              style={{ width: 80, padding: '4px 8px' }}
            />
          </div>
        </div>

        {/* Barra de Progresso Parcial em Tempo Real (RF3.5 e CA14) */}
        {progressoParcial && !progressoParcial.concluido ? (
          <div
            style={{
              background: 'var(--superficie, #1e293b)',
              padding: 12,
              borderRadius: 6,
              marginBottom: 12,
            }}
          >
            <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 6 }}>
              <span className="mono" style={{ fontWeight: 'bold' }}>
                item {progressoParcial.item_atual} de {progressoParcial.total_itens}
              </span>
              <Distintivo status="aviso">
                acurácia parcial: {(progressoParcial.acuracia_parcial * 100).toFixed(1)}% ({progressoParcial.acertos} acertos)
              </Distintivo>
            </div>
            <div
              className="barra-progresso"
              style={{ height: 8, background: 'var(--superficie-2, #0f172a)', borderRadius: 4 }}
            >
              <div
                style={{
                  width: `${(progressoParcial.item_atual / progressoParcial.total_itens) * 100}%`,
                  height: '100%',
                  background: 'var(--primaria)',
                  borderRadius: 4,
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        ) : null}

        <button
          type="button"
          className="botao botao--primario"
          onClick={dispararBenchmark}
          disabled={ocupado || Boolean(chaveBenchmark)}
        >
          {chaveBenchmark ? '… Medição em andamento' : '▶ Medir Agora (Executar Benchmark)'}
        </button>
      </PassoCaderno>

      {/* Passo 2: Relatório detalhado do benchmark (RF6.2) */}
      {relatorioAtual ? (
        <PassoCaderno
          numero={2}
          titulo={`Relatório de acurácia — ${relatorioAtual.run} (${relatorioAtual.estrategia} / ${relatorioAtual.split})`}
          comandoCli={`lab-ia raciocinio --run ${relatorioAtual.run} --estrategia ${relatorioAtual.estrategia} --split ${relatorioAtual.split}`}
          estado="concluido"
        >
          {/* Métricas Principais */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: 12,
              marginBottom: 16,
            }}
          >
            <div style={{ background: 'var(--superficie, #1e293b)', padding: 12, borderRadius: 6 }}>
              <span className="suave" style={{ fontSize: '0.8rem' }}>Acurácia Global</span>
              <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 'bold', color: 'var(--primaria)' }}>
                {(relatorioAtual.acuracia_global * 100).toFixed(1)}%
              </div>
              <span className="suave" style={{ fontSize: '0.75rem' }}>em {relatorioAtual.itens} itens avaliados</span>
            </div>

            <div style={{ background: 'var(--superficie, #1e293b)', padding: 12, borderRadius: 6 }}>
              <span className="suave" style={{ fontSize: '0.8rem' }}>Taxa de Resposta Válida</span>
              <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 'bold' }}>
                {(relatorioAtual.taxa_resposta_valida * 100).toFixed(1)}%
              </div>
              <span className="suave" style={{ fontSize: '0.75rem' }}>respostas no formato correto</span>
            </div>

            <div style={{ background: 'var(--superficie, #1e293b)', padding: 12, borderRadius: 6 }}>
              <span className="suave" style={{ fontSize: '0.8rem' }}>Erros na Amostra</span>
              <div className="mono" style={{ fontSize: '1.6rem', fontWeight: 'bold', color: relatorioAtual.amostra_erros?.length ? 'var(--perigo)' : 'var(--sucesso)' }}>
                {relatorioAtual.amostra_erros?.length ?? 0}
              </div>
              <span className="suave" style={{ fontSize: '0.75rem' }}>registrados para inspeção</span>
            </div>
          </div>

          {/* Acurácia por Família de Lógica (RF6.2) */}
          {relatorioAtual.acuracia_por_familia ? (
            <div style={{ background: 'var(--superficie, #1e293b)', padding: 14, borderRadius: 6, marginBottom: 16 }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem' }}>Acurácia por Família de Problema:</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {Object.entries(relatorioAtual.acuracia_por_familia).map(([familia, acuracia]) => {
                  const pct = (acuracia * 100).toFixed(1)
                  return (
                    <div key={familia} className="linha" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
                      <span className="mono" style={{ width: 140 }}>{familia}</span>
                      <div style={{ flex: 1, height: 8, background: 'var(--superficie-2, #0f172a)', borderRadius: 4, margin: '0 12px' }}>
                        <div
                          style={{
                            width: `${acuracia * 100}%`,
                            height: '100%',
                            background: acuracia >= 0.7 ? 'var(--sucesso, #22c55e)' : acuracia >= 0.4 ? 'var(--aviso, #eab308)' : 'var(--perigo, #ef4444)',
                            borderRadius: 4,
                          }}
                        />
                      </div>
                      <span className="mono" style={{ fontWeight: 'bold', width: 50, textAlign: 'right' }}>
                        {pct}%
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}

          {/* Amostra de até 10 erros (RF6.2) */}
          {relatorioAtual.amostra_erros && relatorioAtual.amostra_erros.length > 0 ? (
            <div style={{ background: 'var(--superficie-2, #181c24)', padding: 12, borderRadius: 6 }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem' }}>
                Amostra de Erros do Modelo ({relatorioAtual.amostra_erros.length} exemplos):
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {relatorioAtual.amostra_erros.map((erro, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '8px 10px',
                      background: 'var(--superficie, #1e293b)',
                      borderRadius: 4,
                      fontSize: '0.8rem',
                    }}
                  >
                    <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                      <strong className="mono">{erro.id}</strong>
                      <div>
                        Esperado: <span className="mono" style={{ color: 'var(--sucesso)' }}>{String(erro.esperado)}</span> · Obtido:{' '}
                        <span className="mono" style={{ color: 'var(--perigo)' }}>{String(erro.obtido ?? 'nenhum')}</span>
                      </div>
                    </div>
                    {erro.texto ? (
                      <div className="mono suave" style={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem' }}>
                        "{erro.texto.slice(0, 150)}…"
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </PassoCaderno>
      ) : null}

      {/* Passo 3: Comparação lado a lado entre duas bases (RF6.3) */}
      <PassoCaderno
        numero={3}
        titulo="Comparar duas bases no mesmo benchmark (diferença em pontos percentuais)"
        comandoCli={`lab-ia comparar --runs ${runCompararA},${runCompararB}`}
        estado={relA && relB ? 'concluido' : 'parado'}
      >
        <div className="linha" style={{ gap: 14, flexWrap: 'wrap', marginBottom: 14 }}>
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-comp-a" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>Base A:</label>
            <select
              id="select-comp-a"
              value={runCompararA}
              onChange={(e) => setRunCompararA(e.target.value)}
              style={{ padding: '4px 8px' }}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
              ))}
            </select>
          </div>

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-comp-b" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>Base B:</label>
            <select
              id="select-comp-b"
              value={runCompararB}
              onChange={(e) => setRunCompararB(e.target.value)}
              style={{ padding: '4px 8px' }}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>{r.run_id}</option>
              ))}
            </select>
          </div>
        </div>

        {relA && relB ? (
          <div style={{ background: 'var(--superficie, #1e293b)', padding: 14, borderRadius: 6 }}>
            <div className="linha" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <div>
                <strong>{runCompararA}:</strong>{' '}
                <span className="mono" style={{ fontSize: '1.2rem', color: 'var(--primaria)' }}>
                  {(relA.acuracia_global * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ fontSize: '1.3rem', fontWeight: 'bold' }}>vs</div>
              <div>
                <strong>{runCompararB}:</strong>{' '}
                <span className="mono" style={{ fontSize: '1.2rem', color: 'var(--serie-b)' }}>
                  {(relB.acuracia_global * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                {(() => {
                  const diff = Number(((relA.acuracia_global - relB.acuracia_global) * 100).toFixed(1))
                  return (
                    <Distintivo status={diff >= 0 ? 'ok' : 'aviso'}>
                      Diferença: {diff >= 0 ? `+${diff}` : diff} p.p.
                    </Distintivo>
                  )
                })()}
              </div>
            </div>
          </div>
        ) : (
          <p className="suave" style={{ padding: 10 }}>
            Selecione duas bases que possuam relatórios de benchmark salvos para ver o comparativo lado a lado.
          </p>
        )}
      </PassoCaderno>
    </div>
  )
}
