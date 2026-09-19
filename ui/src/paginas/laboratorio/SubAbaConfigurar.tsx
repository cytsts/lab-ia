import { useEffect, useState } from 'react'
import {
  api,
  type CatalogoModelos,
  type Dataset,
  type Preset,
  type ResultadoNovo,
} from '../../api'
import { Distintivo } from '../../ds/componentes'
import { PassoCaderno } from './PassoCaderno'

interface SubAbaConfigurarProps {
  aoIrParaExecutar?: (nomeConfig: string) => void
}

export function SubAbaConfigurar({ aoIrParaExecutar }: SubAbaConfigurarProps) {
  // Estado geral
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetAtivo, setDatasetAtivo] = useState('logica-pq')
  const [modo, setModo] = useState<'zero' | 'refinar'>('refinar')
  const [origemBase, setOrigemBase] = useState<'runs' | 'gguf'>('runs')
  const [catalogo, setCatalogo] = useState<CatalogoModelos | null>(null)
  const [baseSelecionada, setBaseSelecionada] = useState<string>('logica-1')
  const [presets, setPresets] = useState<Preset[]>([])

  // Parâmetros do modelo
  const [nomeRun, setNomeRun] = useState('meu-experimento')
  const [preset, setPreset] = useState('equilibrado')
  const [arquitetura, setArquitetura] = useState<'gpt2' | 'moderna'>('moderna')
  const [dim, setDim] = useState(256)
  const [camadas, setCamadas] = useState(6)
  const [cabecas, setCabecas] = useState(8)
  const [janela, setJanela] = useState(256)
  const [lote, setLote] = useState(32)
  const [passos, setPassos] = useState(2500)
  const [lr, setLr] = useState(0.0003)
  const [abandono, setAbandono] = useState(0.0)

  // Campos LoRA / QLoRA
  const [loraR, setLoraR] = useState(8)
  const [loraAlpha, setLoraAlpha] = useState(16)
  const [tipoAjuste, setTipoAjuste] = useState<'lora' | 'qlora'>('lora')
  const [bitsQuant, setBitsQuant] = useState<'8' | '4'>('8')

  // Estimativas e YAML gerado
  const [resultadoNovo, setResultadoNovo] = useState<ResultadoNovo | null>(null)
  const [yamlEditavel, setYamlEditavel] = useState<string>('')
  const [erroValidacao, setErroValidacao] = useState<string | null>(null)
  const [gerando, setGerando] = useState<boolean>(false)

  // Carregar dados iniciais
  useEffect(() => {
    api.datasets().then((ds) => {
      setDatasets(ds)
      if (ds.length > 0 && !ds.some((d) => d.id === datasetAtivo)) {
        setDatasetAtivo(ds[0].id)
      }
    })
    api.presets().then(setPresets)
    api.modelos().then((cat) => {
      setCatalogo(cat)
      if (cat.runs.length > 0) {
        setBaseSelecionada(cat.runs[0].id)
      }
    })
  }, [])

  // Validação imediata (RF2.6 & CA4)
  useEffect(() => {
    if (dim % cabecas !== 0) {
      setErroValidacao(`dim (${dim}) precisa ser divisível por cabecas (${cabecas})`)
      return
    }
    const dimCabeca = dim / cabecas
    if (arquitetura === 'moderna' && dimCabeca % 2 !== 0) {
      setErroValidacao(`RoPE precisa de dim_cabeca par (${dimCabeca} é ímpar)`)
      return
    }
    setErroValidacao(null)
  }, [dim, cabecas, arquitetura])

  // Gerar configuração assistida com estimativas (RF2.5, RF2.7)
  const gerarConfig = async () => {
    if (erroValidacao) return
    setGerando(true)
    try {
      const sobrescritas: Record<string, number | string> = {
        dim,
        camadas,
        cabecas,
        janela_ctx: janela,
        lote,
        passos,
        lr,
        abandono,
        arquitetura,
      }
      if (modo === 'refinar') {
        sobrescritas['lora_r'] = loraR
        sobrescritas['lora_alpha'] = loraAlpha
        sobrescritas['lora_tipo'] = tipoAjuste
        if (tipoAjuste === 'qlora') {
          sobrescritas['quant_bits'] = Number(bitsQuant)
        }
      }

      const res = await api.novo({
        nome: nomeRun.trim() || 'meu-experimento',
        dados: datasetAtivo,
        preset,
        sobrescritas: sobrescritas as Record<string, number>,
        forcar: true,
      })
      setResultadoNovo(res)
      setYamlEditavel(res.yaml)
    } catch (e) {
      setErroValidacao((e as Error).message)
    } finally {
      setGerando(false)
    }
  }

  const comandoPasso2 =
    modo === 'zero'
      ? `lab-ia novo --nome ${nomeRun} --dados ${datasetAtivo} --preset ${preset}`
      : `lab-ia ajustar --config configs/${nomeRun}.yaml --base ${baseSelecionada}`

  const comandoPasso3 = `lab-ia novo --nome ${nomeRun} --dados ${datasetAtivo} --dim ${dim} --camadas ${camadas} --cabecas ${cabecas}`

  return (
    <div className="sub-aba-configurar">
      {/* Passo 1: Dados selecionados */}
      <PassoCaderno
        numero={1}
        titulo="Escolher os dados do experimento"
        comandoCli={`lab-ia dados --listar`}
        estado="concluido"
        mensagemEstado={`dataset '${datasetAtivo}' selecionado`}
      >
        <div className="linha" style={{ gap: 10, alignItems: 'center' }}>
          <label htmlFor="cfg-dataset" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
            Dataset:
          </label>
          <select
            id="cfg-dataset"
            value={datasetAtivo}
            onChange={(e) => setDatasetAtivo(e.target.value)}
            style={{ padding: '6px 10px', minWidth: 220 }}
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}
              </option>
            ))}
          </select>
        </div>
      </PassoCaderno>

      {/* Passo 2: Escolher o modo e a base */}
      <PassoCaderno
        numero={2}
        titulo="Escolher o modo e a base de partida"
        comandoCli={comandoPasso2}
        estado={baseSelecionada || modo === 'zero' ? 'concluido' : 'parado'}
      >
        {/* Escolha do modo */}
        <div className="linha" style={{ gap: 20, marginBottom: 14 }}>
          <label className="linha" style={{ gap: 6, cursor: 'pointer' }}>
            <input
              type="radio"
              name="modo_treino"
              value="zero"
              checked={modo === 'zero'}
              onChange={() => setModo('zero')}
            />
            <span>( ) Treinar do zero (pesos novos)</span>
          </label>
          <label className="linha" style={{ gap: 6, cursor: 'pointer' }}>
            <input
              type="radio"
              name="modo_treino"
              value="refinar"
              checked={modo === 'refinar'}
              onChange={() => setModo('refinar')}
            />
            <span>(●) Refinar (LoRA / QLoRA sobre base congelada)</span>
          </label>
        </div>

        {/* Seleção de base para refino */}
        {modo === 'refinar' ? (
          <div style={{ background: 'var(--superficie-2, #181c24)', padding: 12, borderRadius: 6 }}>
            {/* Abas de Origem da Base */}
            <div className="linha" style={{ gap: 8, marginBottom: 12, borderBottom: '1px solid var(--borda)', paddingBottom: 8 }}>
              <button
                type="button"
                className={`botao${origemBase === 'runs' ? ' botao--primario' : ''}`}
                onClick={() => setOrigemBase('runs')}
                style={{ fontSize: '0.85rem', padding: '4px 12px' }}
              >
                Runs do Laboratório ({catalogo?.runs.length ?? 0})
              </button>
              <button
                type="button"
                className={`botao${origemBase === 'gguf' ? ' botao--primario' : ''}`}
                onClick={() => setOrigemBase('gguf')}
                style={{ fontSize: '0.85rem', padding: '4px 12px' }}
              >
                Modelos GGUF ({catalogo?.gguf.length ?? 0})
              </button>
            </div>

            {/* Lista de Runs Locais */}
            {origemBase === 'runs' ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {catalogo?.runs.map((r) => {
                  const bm = r.benchmarks?.[0]
                  return (
                    <label
                      key={r.id}
                      className="linha"
                      style={{
                        padding: '6px 10px',
                        borderRadius: 4,
                        background: baseSelecionada === r.id ? 'var(--superficie, #273549)' : 'transparent',
                        cursor: 'pointer',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div className="linha" style={{ gap: 8 }}>
                        <input
                          type="radio"
                          name="base_run"
                          value={r.id}
                          checked={baseSelecionada === r.id}
                          onChange={() => setBaseSelecionada(r.id)}
                        />
                        <strong className="mono">{r.id}</strong>
                        <span className="suave" style={{ fontSize: '0.75rem' }}>({r.dispositivo})</span>
                      </div>
                      <div className="linha" style={{ gap: 6 }}>
                        {r.medido && bm ? (
                          <Distintivo status="ok">
                            medido: {(bm.acuracia_global * 100).toFixed(1)}% em lógica ({bm.itens} itens)
                          </Distintivo>
                        ) : (
                          <Distintivo status="info">não medido</Distintivo>
                        )}
                      </div>
                    </label>
                  )
                })}
              </div>
            ) : (
              /* Lista de GGUFs */
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {catalogo?.gguf.length === 0 ? (
                  <p className="suave" style={{ padding: 8 }}>
                    Nenhum arquivo .gguf encontrado em <code className="mono">modelos/</code>.
                  </p>
                ) : (
                  catalogo?.gguf.map((g) => (
                    <div
                      key={g.arquivo}
                      style={{
                        padding: '8px 10px',
                        border: '1px solid var(--borda)',
                        borderRadius: 4,
                        background: 'var(--superficie, #1e293b)',
                      }}
                    >
                      <div className="linha" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                        <strong className="mono">{g.arquivo}</strong>
                        <span className="mono" style={{ fontSize: '0.8rem' }}>
                          {g.tamanho_mb} MB · {g.quantizacao}
                        </span>
                      </div>
                      <div className="linha" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
                        <Distintivo status="aviso">{g.rotulo_honestidade}</Distintivo>
                        <button
                          type="button"
                          className="botao"
                          disabled={!catalogo?.llama_cpp_instalado}
                          style={{ fontSize: '0.75rem', padding: '2px 8px' }}
                        >
                          Medir agora
                        </button>
                      </div>
                    </div>
                  ))
                )}

                {!catalogo?.llama_cpp_instalado ? (
                  <div className="aviso" style={{ marginTop: 6, fontSize: '0.8rem' }}>
                    llama.cpp não detectado — execute: <code className="mono">{catalogo?.comando_instalacao}</code>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        ) : null}
      </PassoCaderno>

      {/* Passo 3: Ajustar e revisar parâmetros com validação */}
      <PassoCaderno
        numero={3}
        titulo="Ajustar hiperparâmetros e estimar recursos"
        comandoCli={comandoPasso3}
        estado={erroValidacao ? 'falhou' : resultadoNovo ? 'concluido' : 'parado'}
        mensagemEstado={erroValidacao ? erroValidacao : undefined}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 14 }}>
          <div>
            <label htmlFor="cfg-nome" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Nome do Experimento:
            </label>
            <input
              id="cfg-nome"
              type="text"
              value={nomeRun}
              onChange={(e) => setNomeRun(e.target.value)}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-preset" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Preset:
            </label>
            <select
              id="cfg-preset"
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
              style={{ width: '100%', padding: '4px 8px' }}
            >
              {presets.map((p) => (
                <option key={p.nome} value={p.nome}>
                  {p.nome} — {p.descricao}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="cfg-arquitetura" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Arquitetura:
            </label>
            <select
              id="cfg-arquitetura"
              value={arquitetura}
              onChange={(e) => setArquitetura(e.target.value as 'gpt2' | 'moderna')}
              style={{ width: '100%', padding: '4px 8px' }}
            >
              <option value="moderna">Moderna (RMSNorm, RoPE, SwiGLU, GQA)</option>
              <option value="gpt2">GPT-2 (Clássica)</option>
            </select>
          </div>

          <div>
            <label htmlFor="cfg-dim" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Dimensão (largura):
            </label>
            <input
              id="cfg-dim"
              type="number"
              value={dim}
              onChange={(e) => setDim(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-cabecas" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Cabeças de Atenção:
            </label>
            <input
              id="cfg-cabecas"
              type="number"
              value={cabecas}
              onChange={(e) => setCabecas(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-camadas" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Camadas:
            </label>
            <input
              id="cfg-camadas"
              type="number"
              value={camadas}
              onChange={(e) => setCamadas(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-passos" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Passos de Treino:
            </label>
            <input
              id="cfg-passos"
              type="number"
              value={passos}
              onChange={(e) => setPassos(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-lr" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Taxa de Aprendizado (lr):
            </label>
            <input
              id="cfg-lr"
              type="number"
              step="0.0001"
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-janela" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Janela de Contexto:
            </label>
            <input
              id="cfg-janela"
              type="number"
              value={janela}
              onChange={(e) => setJanela(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-lote" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Tamanho do Lote:
            </label>
            <input
              id="cfg-lote"
              type="number"
              value={lote}
              onChange={(e) => setLote(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>

          <div>
            <label htmlFor="cfg-abandono" style={{ display: 'block', fontSize: '0.8rem', marginBottom: 3 }}>
              Abandono (Dropout):
            </label>
            <input
              id="cfg-abandono"
              type="number"
              step="0.05"
              min="0"
              max="0.5"
              value={abandono}
              onChange={(e) => setAbandono(Number(e.target.value))}
              style={{ width: '100%', padding: '4px 8px' }}
            />
          </div>
        </div>

        {/* Campos LoRA se for refinamento */}
        {modo === 'refinar' ? (
          <div
            className="linha"
            style={{
              gap: 12,
              padding: 10,
              background: 'var(--superficie-2, #0f172a)',
              borderRadius: 6,
              marginBottom: 12,
            }}
          >
            <div className="linha" style={{ gap: 6 }}>
              <label htmlFor="lora-r" style={{ fontSize: '0.8rem' }}>Posto r:</label>
              <input
                id="lora-r"
                type="number"
                value={loraR}
                onChange={(e) => setLoraR(Number(e.target.value))}
                style={{ width: 60, padding: '2px 6px' }}
              />
            </div>
            <div className="linha" style={{ gap: 6 }}>
              <label htmlFor="lora-alpha" style={{ fontSize: '0.8rem' }}>Alpha:</label>
              <input
                id="lora-alpha"
                type="number"
                value={loraAlpha}
                onChange={(e) => setLoraAlpha(Number(e.target.value))}
                style={{ width: 60, padding: '2px 6px' }}
              />
            </div>
            <div className="linha" style={{ gap: 6 }}>
              <label htmlFor="lora-tipo" style={{ fontSize: '0.8rem' }}>Tipo:</label>
              <select
                id="lora-tipo"
                value={tipoAjuste}
                onChange={(e) => setTipoAjuste(e.target.value as 'lora' | 'qlora')}
                style={{ padding: '2px 6px' }}
              >
                <option value="lora">LoRA (16-bit)</option>
                <option value="qlora">QLoRA (quantizado)</option>
              </select>
            </div>
            {tipoAjuste === 'qlora' ? (
              <div className="linha" style={{ gap: 6 }}>
                <label htmlFor="lora-bits" style={{ fontSize: '0.8rem' }}>Bits:</label>
                <select
                  id="lora-bits"
                  value={bitsQuant}
                  onChange={(e) => setBitsQuant(e.target.value as '8' | '4')}
                  style={{ padding: '2px 6px' }}
                >
                  <option value="8">8-bit</option>
                  <option value="4">4-bit (NF4)</option>
                </select>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Erro de validação imediata (RF2.6 / CA4) */}
        {erroValidacao ? (
          <div className="aviso aviso--erro" style={{ marginBottom: 12 }}>
            ✕ <strong>Configuração Inválida:</strong> {erroValidacao}
          </div>
        ) : null}

        {/* Botão Gerar e Estimativas */}
        <div className="linha" style={{ gap: 12, alignItems: 'center' }}>
          <button
            type="button"
            className="botao botao--primario"
            onClick={gerarConfig}
            disabled={Boolean(erroValidacao) || gerando}
          >
            {gerando ? '… Calculando' : 'Calcular Estimativas e Gerar YAML'}
          </button>

          {resultadoNovo ? (
            <div className="linha" style={{ gap: 12, fontSize: 'var(--miudo)' }}>
              <span>
                <strong>Parâmetros:</strong> {resultadoNovo.parametros.total.toLocaleString('pt-BR')}
              </span>
              <span>
                <strong>VRAM prevista:</strong> {resultadoNovo.vram.mb_total} MB
              </span>
              <span>
                <strong>Tempo previsto:</strong> {resultadoNovo.desempenho.segundos_previstos}s
              </span>
              <span>
                <strong>Épocas:</strong> {resultadoNovo.epocas?.toFixed(1) ?? '—'}
              </span>
            </div>
          ) : null}
        </div>
      </PassoCaderno>

      {/* Passo 4: Revisão e Edição do YAML */}
      {yamlEditavel ? (
        <PassoCaderno
          numero={4}
          titulo="Revisão final do YAML e disparo"
          comandoCli={`lab-ia train --config configs/${nomeRun}.yaml`}
          estado="concluido"
        >
          <p className="suave" style={{ fontSize: 'var(--miudo)', marginBottom: 6 }}>
            O YAML abaixo pode ser editado antes do treinamento:
          </p>
          <textarea
            className="mono"
            rows={10}
            value={yamlEditavel}
            onChange={(e) => setYamlEditavel(e.target.value)}
            style={{
              width: '100%',
              background: 'var(--superficie-2, #0f172a)',
              padding: 10,
              fontSize: '0.85rem',
              lineHeight: 1.4,
              borderRadius: 6,
              marginBottom: 12,
            }}
          />
          {aoIrParaExecutar ? (
            <button
              type="button"
              className="botao botao--primario"
              onClick={() => aoIrParaExecutar(`${nomeRun}.yaml`)}
            >
              Ir para Sub-aba Executar com esta Configuração →
            </button>
          ) : null}
        </PassoCaderno>
      ) : null}
    </div>
  )
}
