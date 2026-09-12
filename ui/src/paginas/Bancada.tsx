import { useEffect, useState } from 'react'
import {
  api,
  type Dataset,
  type Preset,
  type RelatorioVarredura,
  type ResultadoNovo,
  type ResumoVarredura,
  type Saude,
} from '../api'
import type { EstadoNucleo } from '../tipos'
import { Campo, Cartao, Distintivo } from '../ds/componentes'
import { useAvisos } from '../ds/avisos'

/** Página da bancada (B1/B3): trazer dados, gerar config explicada e disparar o treino. */
export function Bancada() {
  const { avisar } = useAvisos()
  const [saude, setSaude] = useState<Saude | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [presets, setPresets] = useState<Preset[]>([])
  const [resultado, setResultado] = useState<ResultadoNovo | null>(null)
  const [ocupado, setOcupado] = useState(false)

  const [fontes, setFontes] = useState('')
  const [idDataset, setIdDataset] = useState('')
  const [fracVal, setFracVal] = useState('0.05')
  const [minChars, setMinChars] = useState('20')

  const [nome, setNome] = useState('')
  const [dados, setDados] = useState('')
  const [preset, setPreset] = useState('rapido')

  const [configs, setConfigs] = useState<string[]>([])
  const [base, setBase] = useState('')
  const [grades, setGrades] = useState('')
  const [passos, setPassos] = useState('')
  const [varreduraSeca, setVarreduraSeca] = useState<RelatorioVarredura | null>(null)
  const [varreduras, setVarreduras] = useState<ResumoVarredura[]>([])

  const [tentativas, setTentativas] = useState(0)
  const [nucleo, setNucleo] = useState<EstadoNucleo | null>(null)

  const recarregar = () => {
    api.saude().then(setSaude).catch(() => setSaude(null))
    api.datasets().then(setDatasets).catch(() => setDatasets([]))
    api.presets().then(setPresets).catch(() => setPresets([]))
    api.configs().then(setConfigs).catch(() => setConfigs([]))
    api.varreduras().then(setVarreduras).catch(() => setVarreduras([]))
    window.labia?.nucleo?.().then(setNucleo).catch(() => setNucleo(null))
  }
  useEffect(recarregar, [])

  // No app portátil o núcleo sobe junto com a janela: enquanto ele não responde,
  // tenta de novo em vez de mandar o usuário abrir um terminal.
  useEffect(() => {
    if (saude || tentativas >= 60) return
    const relogio = setTimeout(() => {
      recarregar()
      setTentativas((n) => n + 1)
    }, 2000)
    return () => clearTimeout(relogio)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saude, tentativas])

  const listaDeFontes = () =>
    fontes
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)

  const preparar = async () => {
    const entradas = listaDeFontes()
    if (!idDataset.trim() || entradas.length === 0) {
      avisar('informe o id do dataset e pelo menos um arquivo ou pasta', true)
      return
    }
    setOcupado(true)
    try {
      const manifesto = (await api.prepararDados({
        fontes: entradas,
        id: idDataset.trim(),
        frac_val: Number(fracVal),
        min_chars: Number(minChars),
      })) as { limpeza?: { paragrafos_finais?: number }; saidas?: { trem?: { chars?: number } } }
      avisar(
        `dataset '${idDataset.trim()}' pronto: ${manifesto.limpeza?.paragrafos_finais ?? '?'} parágrafos, ${manifesto.saidas?.trem?.chars ?? '?'} caracteres de treino`,
      )
      setDados(idDataset.trim())
      recarregar()
    } catch (e) {
      avisar(String((e as Error).message), true)
    } finally {
      setOcupado(false)
    }
  }

  const gerar = async () => {
    if (!nome.trim() || !dados) {
      avisar('informe o nome do experimento e escolha um dataset', true)
      return
    }
    setOcupado(true)
    try {
      const r = await api.novo({ nome: nome.trim(), dados, preset })
      setResultado(r)
      avisar(`config ${r.arquivo_config} gerada`)
    } catch (e) {
      avisar(String((e as Error).message), true)
    } finally {
      setOcupado(false)
    }
  }

const listaDeGrades = () =>
    grades
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)

  const conferirGrade = async () => {
    if (!base || listaDeGrades().length === 0) {
      avisar('escolha a config base e escreva pelo menos uma grade (chave=v1,v2)', true)
      return
    }
    setOcupado(true)
    try {
      const relatorio = await api.varrerSeco({
        base,
        grades: listaDeGrades(),
        passos: passos ? Number(passos) : undefined,
      })
      setVarreduraSeca(relatorio)
      avisar(
        relatorio.falhas && relatorio.falhas.length
          ? `${relatorio.falhas.length} combinação(ões) impossível(is) — veja o detalhe`
          : `${relatorio.variantes.length} combinação(ões) válidas, prontas para rodar`,
        Boolean(relatorio.falhas && relatorio.falhas.length),
      )
    } catch (e) {
      avisar(String((e as Error).message), true)
      setVarreduraSeca(null)
    } finally {
      setOcupado(false)
    }
  }

  const rodarVarredura = async () => {
    if (!base || listaDeGrades().length === 0) {
      avisar('escolha a config base e escreva pelo menos uma grade', true)
      return
    }
    setOcupado(true)
    try {
      const r = await api.varrer({
        base,
        grades: listaDeGrades(),
        passos: passos ? Number(passos) : undefined,
      })
      avisar(`varredura ${r.prefixo} iniciada (pid ${r.pid}) — acompanhe em Executar`)
      recarregar()
    } catch (e) {
      avisar(String((e as Error).message), true)
    } finally {
      setOcupado(false)
    }
  }

  const treinarAgora = async () => {
    if (!resultado) return
    setOcupado(true)
    try {
      const r = await api.executar('train', resultado.arquivo_config)
      avisar(`treino iniciado (pid ${r.pid}) — acompanhe em Executar`)
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
            Núcleo
            {saude ? (
              <Distintivo status={saude.cuda ? 'ok' : 'aviso'}>
                {saude.cuda ? (saude.gpu ?? 'CUDA') : 'CPU'}
              </Distintivo>
            ) : (
              <Distintivo status="perigo">núcleo indisponível</Distintivo>
            )}
          </span>
        }
      >
        {saude ? (
          <p className="mono">
            {saude.runs} runs · {saude.datasets} datasets · {saude.configs} configs · API {saude.versao_api}
          </p>
        ) : (
          <p className="suave">
            {nucleo && nucleo.estado !== 'erro' ? (
              <>núcleo subindo… (porta {nucleo.porta})</>
            ) : (
              <>
                sem resposta em 127.0.0.1:8765 — suba o núcleo com <span className="mono">lab-ia servir</span>
                {nucleo?.erro ? <> · {nucleo.erro}</> : null}
              </>
            )}
          </p>
        )}
      </Cartao>

      <Cartao titulo="1 · Trazer seus dados">
        <Campo rotulo="arquivos, pastas ou curingas (um por linha)">
          <textarea
            className="campo"
            rows={4}
            style={{ width: '100%', font: 'inherit' }}
            aria-label="fontes de dados"
            placeholder={'C:\\meus\\textos\nC:\\outros\\*.md'}
            value={fontes}
            onChange={(e) => setFontes(e.target.value)}
          />
        </Campo>
        <Campo rotulo="id do dataset">
          <input
            className="campo"
            aria-label="id do dataset"
            placeholder="meus-textos"
            value={idDataset}
            onChange={(e) => setIdDataset(e.target.value)}
          />
        </Campo>
        <div className="linha" style={{ gap: 12 }}>
          <Campo rotulo="fração de validação">
            <input className="campo" aria-label="fração de validação" value={fracVal} onChange={(e) => setFracVal(e.target.value)} />
          </Campo>
          <Campo rotulo="mínimo de caracteres por parágrafo">
            <input className="campo" aria-label="mínimo de caracteres" value={minChars} onChange={(e) => setMinChars(e.target.value)} />
          </Campo>
          <button className="botao botao--primario" disabled={ocupado} onClick={preparar}>
            Preparar dataset
          </button>
        </div>
        {datasets.length ? (
          <table className="tabela">
            <thead>
              <tr>
                <th>dataset</th>
                <th>chars (treino)</th>
                <th>chars (validação)</th>
                <th>idioma</th>
                <th>tokens aprox.</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td className="mono">{d.id}</td>
                  <td className="mono">{d.chars_trem}</td>
                  <td className="mono">{d.chars_val}</td>
                  <td className="mono">{d.idioma}</td>
                  <td className="mono">{d.tokens_aprox_trem}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="suave">nenhum dataset preparado ainda.</p>
        )}
      </Cartao>

      <Cartao titulo="2 · Gerar a configuração">
        <Campo rotulo="dataset">
          <select className="campo" aria-label="dataset" value={dados} onChange={(e) => setDados(e.target.value)}>
            <option value="">— escolha —</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}
              </option>
            ))}
          </select>
        </Campo>
        <Campo rotulo="nome do experimento">
          <input className="campo" aria-label="nome do experimento" placeholder="meu-run" value={nome} onChange={(e) => setNome(e.target.value)} />
        </Campo>
        <Campo rotulo="preset">
          <select className="campo" aria-label="preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
            {presets.length ? (
              presets.map((p) => (
                <option key={p.nome} value={p.nome}>
                  {p.nome} — {p.descricao}
                </option>
              ))
            ) : (
              <option value={preset}>{preset}</option>
            )}
          </select>
        </Campo>
        <button className="botao botao--primario" disabled={ocupado} onClick={gerar}>
          Gerar config
        </button>

        {resultado ? (
          <>
            <pre className="mono" style={{ whiteSpace: 'pre-wrap' }}>{resultado.resumo}</pre>
            <button className="botao botao--primario" disabled={ocupado} onClick={treinarAgora}>
              ▶ Treinar agora ({resultado.arquivo_config})
            </button>
          </>
        ) : null}
      </Cartao>
<Cartao titulo="3 · Varrer hiperparâmetros">
        <p className="suave">
          Todas as combinações treinam com o mesmo orçamento de passos — é o que torna a comparação
          honesta. Cada variante vira um run normal, reproduzível por <span className="mono">lab-ia train</span>.
        </p>
        <Campo rotulo="config base">
          <select className="campo" aria-label="config base" value={base} onChange={(e) => setBase(e.target.value)}>
            <option value="">— escolha —</option>
            {configs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Campo>
        <Campo rotulo="grades (uma por linha, no formato chave=v1,v2)">
          <textarea
            className="campo"
            rows={3}
            style={{ width: '100%', font: 'inherit' }}
            aria-label="grades da varredura"
            placeholder={'lr=0.00015,0.0003,0.0006\nlote=16,32'}
            value={grades}
            onChange={(e) => setGrades(e.target.value)}
          />
        </Campo>
        <Campo rotulo="passos por variante (vazio = o da config base)">
          <input className="campo" aria-label="passos por variante" value={passos} onChange={(e) => setPassos(e.target.value)} />
        </Campo>
        <div className="linha" style={{ gap: 12 }}>
          <button className="botao" disabled={ocupado} onClick={conferirGrade}>
            Conferir grade
          </button>
          <button className="botao botao--primario" disabled={ocupado} onClick={rodarVarredura}>
            Rodar varredura
          </button>
        </div>

        {varreduraSeca ? (
          <pre className="mono" style={{ whiteSpace: 'pre-wrap' }}>{varreduraSeca.tabela}</pre>
        ) : null}

        {varreduras.length ? (
          <table className="tabela">
            <thead>
              <tr>
                <th>varredura</th>
                <th>variantes</th>
                <th>melhor</th>
                <th>val</th>
              </tr>
            </thead>
            <tbody>
              {varreduras.map((v) => (
                <tr key={v.prefixo}>
                  <td className="mono">{v.prefixo}</td>
                  <td className="mono">{v.variantes}</td>
                  <td className="mono">{v.melhor?.run_id ?? '—'}</td>
                  <td className="mono">{v.melhor ? v.melhor.melhor_val.toFixed(4) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Cartao>
    </>
  )
}

