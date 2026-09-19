import { useEffect, useState } from 'react'
import {
  api,
  type Dataset,
  type ItemBenchmark,
  type ManifestoDataset,
  type RespostaItensDataset,
} from '../../api'
import { Distintivo } from '../../ds/componentes'
import { PassoCaderno } from './PassoCaderno'

export function SubAbaDados() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetSelecionado, setDatasetSelecionado] = useState<string>('logica-pq')
  const [manifestoAtual, setManifestoAtual] = useState<ManifestoDataset | null>(null)
  const [dadosItens, setDadosItens] = useState<RespostaItensDataset | null>(null)
  const [indiceItem, setIndiceItem] = useState<number>(0)
  const [splitFiltro, setSplitFiltro] = useState<string>('todos')
  const [familiaFiltro, setFamiliaFiltro] = useState<string>('todas')
  const [busca, setBusca] = useState<string>('')
  const [carregando, setCarregando] = useState<boolean>(false)

  // Carregar lista de datasets
  useEffect(() => {
    api.datasets().then((ds) => {
      setDatasets(ds)
      if (ds.length > 0 && !ds.some((d) => d.id === datasetSelecionado)) {
        setDatasetSelecionado(ds[0].id)
      }
    })
  }, [])

  // Carregar manifesto do dataset selecionado
  useEffect(() => {
    if (!datasetSelecionado) return
    api.dataset(datasetSelecionado)
      .then(setManifestoAtual)
      .catch(() => setManifestoAtual(null))
  }, [datasetSelecionado])

  // Carregar itens com filtros e paginação
  const carregarItens = (novoIndice = 0) => {
    if (!datasetSelecionado) return
    setCarregando(true)
    api.itensDataset(datasetSelecionado, {
      desde: novoIndice,
      limite: 1, // Exibição item a item no navegador
      split: splitFiltro === 'todos' ? undefined : splitFiltro,
      familia: familiaFiltro === 'todas' ? undefined : familiaFiltro,
      busca: busca.trim() || undefined,
    })
      .then((res) => {
        setDadosItens(res)
        setIndiceItem(novoIndice)
      })
      .catch(() => setDadosItens(null))
      .finally(() => setCarregando(false))
  }

  useEffect(() => {
    carregarItens(0)
  }, [datasetSelecionado, splitFiltro, familiaFiltro, busca])

  const itemAtual: ItemBenchmark | null = dadosItens?.itens?.[0] ?? null

  const comandoPasso1 = `lab-ia dados --listar`
  const comandoPasso2 = `lab-ia dados --id ${datasetSelecionado || 'logica-pq'} --json`

  return (
    <div className="sub-aba-dados">
      {/* Passo 1: Escolher os dados e inspecionar manifesto */}
      <PassoCaderno
        numero={1}
        titulo="Escolher o dataset e inspecionar procedência"
        comandoCli={comandoPasso1}
        estado={datasets.length > 0 ? 'concluido' : 'parado'}
        mensagemEstado={datasets.length > 0 ? `${datasets.length} datasets disponíveis` : 'nenhum dataset'}
      >
        <div className="linha" style={{ gap: 12, alignItems: 'center', marginBottom: 12 }}>
          <label htmlFor="select-dataset" style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>
            Dataset ativo:
          </label>
          <select
            id="select-dataset"
            value={datasetSelecionado}
            onChange={(e) => setDatasetSelecionado(e.target.value)}
            style={{ padding: '6px 10px', minWidth: 220 }}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id} ({d.tokens_aprox_trem ? `${Math.round(d.tokens_aprox_trem / 1000)}k tokens` : `${d.chars_trem} chars`})
              </option>
            ))}
          </select>
        </div>

        {manifestoAtual ? (
          <div
            className="manifesto-card"
            style={{
              background: 'var(--superficie, #1e293b)',
              padding: 12,
              borderRadius: 6,
              fontSize: 'var(--miudo)',
            }}
          >
            <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
              <span>
                <strong>ID:</strong> <code className="mono">{manifestoAtual.id}</code>
              </span>
              <span>
                <strong>Idioma:</strong> {manifestoAtual.idioma}
              </span>
              <span>
                <strong>Tokens aprox. treino:</strong>{' '}
                {manifestoAtual.tokens_aprox_trem?.toLocaleString('pt-BR') ?? '—'}
              </span>
              <span>
                <strong>SHA256 (trem):</strong>{' '}
                <code className="mono">{manifestoAtual.sha256?.slice(0, 12) ?? '—'}…</code>
              </span>
            </div>

            {manifestoAtual.splits && Object.keys(manifestoAtual.splits).length > 0 ? (
              <div className="linha" style={{ gap: 8, marginTop: 6 }}>
                <strong>Splits:</strong>
                {Object.entries(manifestoAtual.splits).map(([split, qtd]) => (
                  <Distintivo key={split} status="info">
                    {split}: {qtd} itens
                  </Distintivo>
                ))}
              </div>
            ) : null}

            {manifestoAtual.familias && manifestoAtual.familias.length > 0 ? (
              <div className="linha" style={{ gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                <strong>Famílias:</strong>
                {manifestoAtual.familias.map((fam) => (
                  <span
                    key={fam}
                    className="mono"
                    style={{
                      padding: '2px 6px',
                      background: 'var(--superficie-2, #0f172a)',
                      borderRadius: 4,
                      fontSize: '0.75rem',
                    }}
                  >
                    {fam}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </PassoCaderno>

      {/* Passo 2: Navegador de itens com filtros */}
      <PassoCaderno
        numero={2}
        titulo="Navegador de itens (enunciado, cadeia CoT e resposta verificada)"
        comandoCli={comandoPasso2}
        estado={itemAtual ? 'concluido' : 'parado'}
        mensagemEstado={
          dadosItens
            ? `${dadosItens.filtrados} itens no filtro (${dadosItens.total} totais)`
            : 'carregando...'
        }
      >
        {/* Barra de Filtros */}
        <div
          className="linha"
          style={{
            gap: 12,
            marginBottom: 12,
            flexWrap: 'wrap',
            background: 'var(--superficie-2, #181c24)',
            padding: 10,
            borderRadius: 6,
          }}
        >
          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="filtro-split" style={{ fontSize: '0.8rem' }}>Split:</label>
            <select
              id="filtro-split"
              value={splitFiltro}
              onChange={(e) => setSplitFiltro(e.target.value)}
              style={{ padding: '4px 8px', fontSize: '0.85rem' }}
            >
              <option value="todos">Todos os splits</option>
              {manifestoAtual?.splits
                ? Object.keys(manifestoAtual.splits).map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))
                : (
                  <>
                    <option value="treino">treino</option>
                    <option value="teste">teste</option>
                    <option value="dificil">dificil</option>
                  </>
                )}
            </select>
          </div>

          <div className="linha" style={{ gap: 6 }}>
            <label htmlFor="filtro-familia" style={{ fontSize: '0.8rem' }}>Família:</label>
            <select
              id="filtro-familia"
              value={familiaFiltro}
              onChange={(e) => setFamiliaFiltro(e.target.value)}
              style={{ padding: '4px 8px', fontSize: '0.85rem' }}
            >
              <option value="todas">Todas as famílias</option>
              {manifestoAtual?.familias?.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          <div className="linha" style={{ gap: 6, flex: 1, minWidth: 180 }}>
            <label htmlFor="filtro-busca" style={{ fontSize: '0.8rem' }}>Busca:</label>
            <input
              id="filtro-busca"
              type="search"
              placeholder="Buscar em enunciado, CoT ou resposta..."
              value={busca}
              onChange={(e) => setBusca(e.target.value)}
              style={{ width: '100%', padding: '4px 8px', fontSize: '0.85rem' }}
            />
          </div>
        </div>

        {/* Visualizador do Item Atual */}
        {carregando ? (
          <p className="suave" style={{ padding: 16 }}>Carregando item...</p>
        ) : itemAtual ? (
          <div
            className="item-visualizador"
            style={{
              border: '1px solid var(--borda)',
              borderRadius: 6,
              padding: 14,
              background: 'var(--superficie, #1e293b)',
            }}
          >
            <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
              <div className="linha" style={{ gap: 8 }}>
                <span className="mono" style={{ fontWeight: 'bold', color: 'var(--primaria)' }}>
                  Item {indiceItem + 1} de {dadosItens?.filtrados ?? 1}
                </span>
                <Distintivo status="info">{itemAtual.split}</Distintivo>
                <Distintivo status="ok">{itemAtual.familia}</Distintivo>
                {itemAtual.id ? <span className="suave mono" style={{ fontSize: '0.75rem' }}>{itemAtual.id}</span> : null}
              </div>

              {/* Controles de navegação rápida */}
              <div className="linha" style={{ gap: 6 }}>
                <button
                  type="button"
                  className="botao"
                  disabled={indiceItem <= 0}
                  onClick={() => carregarItens(Math.max(0, indiceItem - 1))}
                  style={{ padding: '3px 10px' }}
                >
                  ← Anterior
                </button>
                <button
                  type="button"
                  className="botao"
                  disabled={!dadosItens || indiceItem >= dadosItens.filtrados - 1}
                  onClick={() => carregarItens(indiceItem + 1)}
                  style={{ padding: '3px 10px' }}
                >
                  Próximo →
                </button>
              </div>
            </div>

            {/* Enunciado */}
            <div style={{ marginBottom: 10 }}>
              <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--tinta-suave)', marginBottom: 2 }}>
                Enunciado do Problema:
              </strong>
              <div
                style={{
                  background: 'var(--superficie-2, #0f172a)',
                  padding: '8px 12px',
                  borderRadius: 4,
                  fontSize: '0.95rem',
                  lineHeight: 1.4,
                }}
              >
                {itemAtual.enunciado}
              </div>
            </div>

            {/* Raciocínio CoT passo a passo */}
            {itemAtual.cot ? (
              <div style={{ marginBottom: 10 }}>
                <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--tinta-suave)', marginBottom: 2 }}>
                  Cadeia de Raciocínio Esperada (CoT):
                </strong>
                <pre
                  className="mono"
                  style={{
                    background: 'var(--superficie-2, #0f172a)',
                    padding: '8px 12px',
                    borderRadius: 4,
                    fontSize: '0.85rem',
                    lineHeight: 1.45,
                    whiteSpace: 'pre-wrap',
                    margin: 0,
                  }}
                >
                  {itemAtual.cot}
                </pre>
              </div>
            ) : null}

            {/* Resposta esperada verificada */}
            <div className="linha" style={{ gap: 8, alignItems: 'center' }}>
              <strong style={{ fontSize: '0.85rem', color: 'var(--tinta-suave)' }}>
                Resposta gabaritada (verificada por tabela-verdade):
              </strong>
              <span
                className="mono"
                style={{
                  padding: '2px 8px',
                  background: 'var(--primaria)',
                  color: '#fff',
                  borderRadius: 4,
                  fontWeight: 'bold',
                }}
              >
                {String(itemAtual.resposta)}
              </span>
              <span className="suave" style={{ fontSize: '0.75rem' }}>
                ({itemAtual.resposta === 1 || itemAtual.resposta === '1' ? '1 = verdadeiro' : '0 = falso'})
              </span>
            </div>
          </div>
        ) : (
          <p className="suave" style={{ padding: 12 }}>
            Nenhum item encontrado com os filtros atuais.
          </p>
        )}
      </PassoCaderno>
    </div>
  )
}
