import { useEffect, useState } from 'react'
import {
  api,
  type EstadoRun,
  type ItemBenchmark,
  type RespostaTestar,
} from '../../api'
import { Distintivo } from '../../ds/componentes'
import { PassoCaderno } from './PassoCaderno'

interface PerguntaHistorico {
  id: string
  enunciado: string
  estrategia: string
  textoGerado: string
  respostaExtraida: number | string | null
  esperado?: number | string | null
  acerto?: boolean | null
  exploracao: boolean
  dataHora: string
}

export function SubAbaTestar() {
  const [runs, setRuns] = useState<EstadoRun[]>([])
  const [runId, setRunId] = useState<string>('logica-1')

  // Configuração da inferência
  const [origemEnunciado, setOrigemEnunciado] = useState<'digitado' | 'sorteado'>('digitado')
  const [enunciado, setEnunciado] = useState<string>(
    'p = 1, q = 0. Calcule: (p E q). Use 1 para verdadeiro e 0 para falso.',
  )
  const [itemSorteado, setItemSorteado] = useState<ItemBenchmark | null>(null)
  const [estrategia, setEstrategia] = useState<'cot' | 'direta'>('cot')
  const [guloso, setGuloso] = useState<boolean>(true)
  const [temperatura, setTemperatura] = useState<number>(0.7)
  const [maxTokens, setMaxTokens] = useState<number>(256)

  // Resultado atual
  const [respostaAtual, setRespostaAtual] = useState<RespostaTestar | null>(null)
  const [ocupado, setOcupado] = useState<boolean>(false)

  // Painel de Exploração CoT (RF7)
  const [textoEditavelCoT, setTextoEditavelCoT] = useState<string>('')
  const [linhaDivergente, setLinhaDivergente] = useState<number | null>(null)
  const [comparacaoPassos, setComparacaoPassos] = useState<
    { esperada: string; obtida: string; diverge: boolean }[] | null
  >(null)

  // Histórico da sessão (últimas 20) (RF5.5)
  const [historico, setHistorico] = useState<PerguntaHistorico[]>([])

  // Carregar lista de runs
  useEffect(() => {
    api.corre().then((r) => {
      setRuns(r)
      if (r.length > 0 && !r.some((x) => x.run_id === runId)) {
        setRunId(r[r.length - 1].run_id)
      }
    })
  }, [])

  // Sorteio de item do dataset (RF5.2)
  const sortearItem = async () => {
    try {
      // Sorteia um índice aleatório dentro dos primeiros 300 do split teste
      const indice = Math.floor(Math.random() * 60)
      const res = await api.itensDataset('logica-pq', { desde: indice, limite: 1, split: 'teste' })
      if (res.itens && res.itens.length > 0) {
        const item = res.itens[0]
        setItemSorteado(item)
        setEnunciado(item.enunciado)
        setRespostaAtual(null)
        setTextoEditavelCoT('')
        setComparacaoPassos(null)
        setLinhaDivergente(null)
      }
    } catch {
      /* se falhar sortear */
    }
  }

  // Envio de teste / inferência (RF5.4)
  const testar = async (prefixoCustomizado?: string) => {
    if (!enunciado.trim()) return
    setOcupado(true)
    setComparacaoPassos(null)
    setLinhaDivergente(null)

    try {
      const res = await api.testar({
        run: runId,
        enunciado: enunciado.trim(),
        estrategia,
        guloso,
        temperatura: guloso ? 0.0 : temperatura,
        max_tokens: maxTokens,
        prefixo: prefixoCustomizado || undefined,
        semente: 42,
      })

      setRespostaAtual(res)
      setTextoEditavelCoT(res.texto_gerado)

      // Calcula acerto se houver valor esperado
      const esperado = itemSorteado?.resposta ?? null
      let acerto: boolean | null = null
      if (esperado !== null && esperado !== undefined && res.resposta_extraida !== null) {
        acerto = String(res.resposta_extraida) === String(esperado)
      }

      // Adiciona ao histórico (RF5.5)
      setHistorico((antigo) => [
        {
          id: String(Date.now()),
          enunciado: res.enunciado,
          estrategia: res.estrategia,
          textoGerado: res.texto_gerado,
          respostaExtraida: res.resposta_extraida,
          esperado,
          acerto,
          exploracao: res.exploracao,
          dataHora: new Date().toLocaleTimeString('pt-BR'),
        },
        ...antigo.slice(0, 19), // Mantém no máximo 20
      ])
    } finally {
      setOcupado(false)
    }
  }

  // Continuar geração a partir do CoT editado (Teacher Forcing - RF7.1)
  const continuarGerecaoCoT = () => {
    testar(textoEditavelCoT)
  }

  // Botão "Onde ele errou" (RF7.4 e CA16)
  const detectarOndeErrou = () => {
    if (!itemSorteado?.cot || !respostaAtual?.texto_gerado) return

    const linhasEsperadas = itemSorteado.cot.split('\n').map((l) => l.trim()).filter(Boolean)
    const linhasObtidas = respostaAtual.texto_gerado.split('\n').map((l) => l.trim()).filter(Boolean)

    const maxLinhas = Math.max(linhasEsperadas.length, linhasObtidas.length)
    const comparacoes = []
    let primeiraDivergencia: number | null = null

    for (let i = 0; i < maxLinhas; i++) {
      const esp = linhasEsperadas[i] || '(fim da cadeia esperada)'
      const obt = linhasObtidas[i] || '(fim da cadeia gerada)'
      const diverge = esp !== obt
      if (diverge && primeiraDivergencia === null) {
        primeiraDivergencia = i
      }
      comparacoes.push({ esperada: esp, obtida: obt, diverge })
    }

    setComparacaoPassos(comparacoes)
    setLinhaDivergente(primeiraDivergencia)
  }

  const comandoCli = `lab-ia gerar --run ${runId} --prompt "${enunciado.slice(0, 50)}..."${guloso ? ' --guloso' : ''}`

  return (
    <div className="sub-aba-testar">
      {/* Passo 1: Seleção de run e formulação da pergunta */}
      <PassoCaderno
        numero={1}
        titulo="Escolher o modelo e formular o enunciado de teste"
        comandoCli={comandoCli}
        estado={respostaAtual ? 'concluido' : 'parado'}
        mensagemEstado={respostaAtual ? `resposta gerada em ${respostaAtual.tempo_s}s` : undefined}
      >
        <div className="linha" style={{ gap: 14, flexWrap: 'wrap', marginBottom: 12 }}>
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-run-testar" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
              Run ativo:
            </label>
            <select
              id="select-run-testar"
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

          <div className="linha" style={{ gap: 10 }}>
            <button
              type="button"
              className={`botao${origemEnunciado === 'digitado' ? ' botao--primario' : ''}`}
              onClick={() => {
                setOrigemEnunciado('digitado')
                setItemSorteado(null)
              }}
              style={{ fontSize: '0.8rem', padding: '4px 10px' }}
            >
              Digitar pergunta
            </button>
            <button
              type="button"
              className={`botao${origemEnunciado === 'sorteado' ? ' botao--primario' : ''}`}
              onClick={() => {
                setOrigemEnunciado('sorteado')
                sortearItem()
              }}
              style={{ fontSize: '0.8rem', padding: '4px 10px' }}
            >
              🎲 Sortear do dataset logica-pq
            </button>
          </div>
        </div>

        {/* Textarea do enunciado */}
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="input-enunciado" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', marginBottom: 4 }}>
            Enunciado da pergunta:
          </label>
          <textarea
            id="input-enunciado"
            rows={3}
            value={enunciado}
            onChange={(e) => setEnunciado(e.target.value)}
            placeholder="Digite aqui o problema de raciocínio lógico ou fórmula proposicional..."
            style={{
              width: '100%',
              padding: 8,
              fontSize: '0.9rem',
              borderRadius: 6,
              lineHeight: 1.4,
            }}
          />
        </div>

        {/* Detalhes do item sorteado se houver */}
        {itemSorteado ? (
          <div
            className="linha"
            style={{
              gap: 10,
              background: 'var(--superficie-2, #0f172a)',
              padding: '6px 10px',
              borderRadius: 4,
              marginBottom: 12,
              fontSize: 'var(--miudo)',
              justifyContent: 'space-between',
            }}
          >
            <div className="linha" style={{ gap: 6 }}>
              <span>Item sorteado:</span>
              <Distintivo status="info">{itemSorteado.familia}</Distintivo>
              <Distintivo status="ok">split: {itemSorteado.split}</Distintivo>
            </div>
            <div>
              <strong>Resposta esperada verificada:</strong>{' '}
              <span className="mono" style={{ color: 'var(--primaria)', fontWeight: 'bold' }}>
                {String(itemSorteado.resposta)}
              </span>
            </div>
          </div>
        ) : null}

        {/* Opções de inferência */}
        <div
          className="linha"
          style={{
            gap: 16,
            flexWrap: 'wrap',
            background: 'var(--superficie-2, #181c24)',
            padding: '8px 12px',
            borderRadius: 6,
            marginBottom: 14,
          }}
        >
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="select-estrategia" style={{ fontSize: '0.8rem' }}>Estratégia:</label>
            <select
              id="select-estrategia"
              value={estrategia}
              onChange={(e) => setEstrategia(e.target.value as 'cot' | 'direta')}
              style={{ padding: '3px 8px', fontSize: '0.85rem' }}
            >
              <option value="cot">CoT (Passo a passo com tabela/derivação)</option>
              <option value="direta">Direta (Só o resultado)</option>
            </select>
          </div>

          <label className="linha" style={{ gap: 6, cursor: 'pointer', fontSize: '0.85rem' }}>
            <input
              type="checkbox"
              checked={guloso}
              onChange={(e) => setGuloso(e.target.checked)}
            />
            <span>Decodificação gulosa (determinística)</span>
          </label>

          {!guloso ? (
            <div className="linha" style={{ gap: 6 }}>
              <label htmlFor="input-temperatura" style={{ fontSize: '0.8rem' }}>Temperatura:</label>
              <input
                id="input-temperatura"
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                value={temperatura}
                onChange={(e) => setTemperatura(Number(e.target.value))}
                style={{ width: 60, padding: '2px 6px' }}
              />
            </div>
          ) : null}

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="input-max-tokens" style={{ fontSize: '0.8rem' }}>Máx. Tokens:</label>
            <input
              id="input-max-tokens"
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              style={{ width: 70, padding: '2px 6px' }}
            />
          </div>
        </div>

        {/* Aviso de temperatura ignorada com guloso (RF5.6) */}
        {guloso && temperatura > 0 ? (
          <p className="suave" style={{ fontSize: '0.75rem', marginBottom: 10 }}>
            ℹ Decodificação gulosa ativada: a temperatura é desconsiderada para garantir determinismo byte a byte.
          </p>
        ) : null}

        {/* Botão de Enviar */}
        <button
          type="button"
          className="botao botao--primario"
          onClick={() => testar()}
          disabled={ocupado || !enunciado.trim()}
        >
          {ocupado ? '… Gerando raciocínio' : '▶ Gerar Resposta'}
        </button>
      </PassoCaderno>

      {/* Passo 2: Resultado em 3 blocos separados e rotulados (RF5.4) */}
      {respostaAtual ? (
        <PassoCaderno
          numero={2}
          titulo="Resultado da inferência (texto gerado, extração e veredito)"
          comandoCli={`lab-ia gerar --run ${runId} --prompt "${enunciado.slice(0, 40)}..."`}
          estado="concluido"
        >
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, marginBottom: 16 }}>
            {/* Bloco 1: Texto Gerado */}
            <div
              style={{
                gridColumn: '1 / -1',
                background: 'var(--superficie-2, #090d13)',
                border: '1px solid var(--borda)',
                borderRadius: 6,
                padding: 12,
              }}
            >
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--tinta-suave)', marginBottom: 6 }}>
                1. Texto Gerado Completo pelo Modelo:
              </strong>
              <pre
                className="mono"
                style={{
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                  fontSize: '0.85rem',
                  lineHeight: 1.45,
                  color: '#e2e8f0',
                }}
              >
                {respostaAtual.texto_gerado}
              </pre>
            </div>

            {/* Bloco 2: Resposta Extraída */}
            <div
              style={{
                background: 'var(--superficie, #1e293b)',
                border: '1px solid var(--borda)',
                borderRadius: 6,
                padding: 12,
              }}
            >
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--tinta-suave)', marginBottom: 6 }}>
                2. Resposta Extraída pelo Avaliador:
              </strong>
              <div className="linha" style={{ gap: 8, alignItems: 'center' }}>
                <span
                  className="mono"
                  style={{
                    fontSize: '1.2rem',
                    fontWeight: 'bold',
                    padding: '2px 10px',
                    background: 'var(--superficie-2, #0f172a)',
                    borderRadius: 4,
                  }}
                >
                  {respostaAtual.resposta_extraida !== null ? String(respostaAtual.resposta_extraida) : '(nenhuma)'}
                </span>
                <span className="suave" style={{ fontSize: '0.8rem' }}>
                  {respostaAtual.resposta_extraida === 1 ? 'Verdadeiro' : respostaAtual.resposta_extraida === 0 ? 'Falso' : 'Sem resposta no formato'}
                </span>
              </div>
            </div>

            {/* Bloco 3: Veredito e Gabarito */}
            <div
              style={{
                background: 'var(--superficie, #1e293b)',
                border: '1px solid var(--borda)',
                borderRadius: 6,
                padding: 12,
              }}
            >
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--tinta-suave)', marginBottom: 6 }}>
                3. Veredito e Comparação com Gabarito:
              </strong>
              {itemSorteado ? (
                <div className="linha" style={{ gap: 10, alignItems: 'center' }}>
                  <Distintivo
                    status={
                      String(respostaAtual.resposta_extraida) === String(itemSorteado.resposta)
                        ? 'ok'
                        : respostaAtual.resposta_extraida === null
                        ? 'aviso'
                        : 'perigo'
                    }
                  >
                    {String(respostaAtual.resposta_extraida) === String(itemSorteado.resposta)
                      ? '✓ Acertou'
                      : respostaAtual.resposta_extraida === null
                      ? '? Ilegível'
                      : '✕ Errou'}
                  </Distintivo>
                  <span style={{ fontSize: '0.85rem' }}>
                    Esperado: <strong>{String(itemSorteado.resposta)}</strong>
                  </span>
                </div>
              ) : (
                <span className="suave" style={{ fontSize: '0.85rem' }}>
                  Enunciado digitado manualmente (sem gabarito automático).
                </span>
              )}
            </div>
          </div>

          {/* ========================================================= */}
          {/* PASSO 3 / PAINEL DE EXPLORAÇÃO DO COT (RF7, CA15, CA16)   */}
          {/* ========================================================= */}
          <div
            style={{
              marginTop: 18,
              border: '2px solid var(--primaria)',
              borderRadius: 8,
              padding: 14,
              background: 'rgba(56, 189, 248, 0.03)',
            }}
          >
            {/* Rótulo Obrigatório de Exploração (RF7.3) */}
            <div
              className="aviso"
              style={{
                marginBottom: 12,
                borderLeft: '4px solid var(--serie-c, #0284c7)',
                fontSize: '0.85rem',
                lineHeight: 1.4,
              }}
            >
              🧭 <strong>Painel de Exploração (Teacher Forcing):</strong> aqui você está forçando o caminho
              (teacher forcing): o resultado não é a acurácia do modelo, é o que ele faz quando o começo já está certo.
              Os resultados desta área são salvos separadamente em <code className="mono">.lab-ia/exploracao.jsonl</code> para nunca contaminar relatórios de benchmark.
            </div>

            <label htmlFor="textarea-cot-editavel" style={{ display: 'block', fontSize: '0.8rem', fontWeight: 'bold', marginBottom: 4 }}>
              Edite os passos do CoT abaixo para testar correções intermediárias:
            </label>
            <textarea
              id="textarea-cot-editavel"
              className="mono"
              rows={6}
              value={textoEditavelCoT}
              onChange={(e) => setTextoEditavelCoT(e.target.value)}
              style={{
                width: '100%',
                background: 'var(--superficie-2, #090d13)',
                color: '#f8fafc',
                padding: 10,
                fontSize: '0.85rem',
                lineHeight: 1.45,
                borderRadius: 6,
                marginBottom: 10,
              }}
            />

            <div className="linha" style={{ gap: 10, flexWrap: 'wrap' }}>
              <button
                type="button"
                className="botao botao--primario"
                onClick={continuarGerecaoCoT}
                disabled={ocupado || !textoEditavelCoT.trim()}
              >
                ⏩ Continuar geração a partir daqui
              </button>

              {itemSorteado?.cot ? (
                <button
                  type="button"
                  className="botao"
                  onClick={detectarOndeErrou}
                  disabled={ocupado}
                  title="Compara passo a passo com o gabarito e localiza a primeira divergência"
                >
                  🔍 Onde ele errou? (comparar passos)
                </button>
              ) : null}
            </div>

            {/* Painel Comparativo Linha a Linha (RF7.4 e CA16) */}
            {comparacaoPassos ? (
              <div
                style={{
                  marginTop: 14,
                  background: 'var(--superficie, #1e293b)',
                  border: '1px solid var(--borda)',
                  borderRadius: 6,
                  padding: 12,
                }}
              >
                <h4 style={{ margin: '0 0 8px 0', fontSize: '0.85rem' }}>
                  Comparação Linha a Linha entre Esperado e Gerado:
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {comparacaoPassos.map((c, idx) => (
                    <div
                      key={idx}
                      className="linha"
                      style={{
                        padding: '6px 8px',
                        borderRadius: 4,
                        background: c.diverge ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
                        borderLeft: c.diverge ? '3px solid var(--perigo, #ef4444)' : '3px solid transparent',
                        fontSize: '0.8rem',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div className="mono" style={{ flex: 1 }}>
                        <span className="suave" style={{ marginRight: 6 }}>L{idx + 1}:</span>
                        <span style={{ color: c.diverge ? 'var(--perigo)' : 'inherit' }}>
                          {c.obtida}
                        </span>
                      </div>
                      {c.diverge ? (
                        <div className="mono suave" style={{ fontSize: '0.75rem' }}>
                          Esperado: {c.esperada}
                        </div>
                      ) : (
                        <span style={{ color: 'var(--sucesso, #22c55e)' }}>✓ coincide</span>
                      )}
                    </div>
                  ))}
                </div>

                {linhaDivergente !== null ? (
                  <p className="suave" style={{ fontSize: '0.75rem', marginTop: 8, marginBottom: 0 }}>
                    Nota: O primeiro passo divergente foi localizado na linha {linhaDivergente + 1}. A divergência de formato ou ordem dos passos não implica necessariamente em erro lógico da resposta final.
                  </p>
                ) : (
                  <p style={{ fontSize: '0.8rem', color: 'var(--sucesso)', marginTop: 8, marginBottom: 0 }}>
                    ✓ Todos os passos coincidem com o gabarito do dataset!
                  </p>
                )}
              </div>
            ) : null}
          </div>
        </PassoCaderno>
      ) : null}

      {/* Histórico da Sessão (RF5.5) */}
      {historico.length > 0 ? (
        <section className="cartao" style={{ marginTop: 20 }}>
          <header className="linha" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
            <h3 style={{ margin: 0, fontSize: '0.95rem' }}>
              Histórico da Sessão ({historico.length} perguntas)
            </h3>
            <span className="suave" style={{ fontSize: '0.75rem' }}>
              Histórico volátil mantido na sessão (fechar a janela limpa este histórico)
            </span>
          </header>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {historico.map((h) => (
              <div
                key={h.id}
                style={{
                  padding: '8px 12px',
                  background: 'var(--superficie-2, #181c24)',
                  borderRadius: 6,
                  border: '1px solid var(--borda)',
                  fontSize: '0.85rem',
                }}
              >
                <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                  <span className="mono" style={{ fontWeight: 'bold' }}>{h.enunciado.slice(0, 60)}…</span>
                  <div className="linha" style={{ gap: 6 }}>
                    {h.exploracao ? <Distintivo status="aviso">exploração</Distintivo> : null}
                    <Distintivo status={h.acerto === true ? 'ok' : h.acerto === false ? 'perigo' : 'info'}>
                      {h.acerto === true ? '✓ acertou' : h.acerto === false ? '✕ errou' : 'resposta gerada'}
                    </Distintivo>
                    <span className="suave" style={{ fontSize: '0.75rem' }}>{h.dataHora}</span>
                  </div>
                </div>
                <div className="linha" style={{ gap: 8, color: 'var(--tinta-suave)', fontSize: '0.8rem' }}>
                  <span>Resposta: <strong>{String(h.respostaExtraida ?? '—')}</strong></span>
                  {h.esperado !== null && h.esperado !== undefined ? (
                    <span>Esperado: <strong>{String(h.esperado)}</strong></span>
                  ) : null}
                  <span>Estratégia: {h.estrategia}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  )
}
