import { useEffect, useState } from 'react'
import { api } from '../api'
import { Campo, Cartao, Distintivo } from '../ds/componentes'
import { useAvisos } from '../ds/avisos'

export function Executar() {
  const [configs, setConfigs] = useState<string[]>([])
  const [acao, setAcao] = useState('train')
  const [config, setConfig] = useState('')
  const [execucoes, setExecucoes] = useState<{ chave: string; pid: number; vivo: boolean; codigo_saida: number | null }[]>([])
  const [ocupado, setOcupado] = useState(false)
  const { avisar } = useAvisos()

  const recarregar = () => {
    api.configs().then(setConfigs).catch((e: Error) => avisar(e.message, true))
    api.execucoes().then(setExecucoes).catch(() => setExecucoes([]))
  }
  useEffect(() => {
    recarregar()
    const t = setInterval(recarregar, 5000)
    return () => clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const disparar = async () => {
    setOcupado(true)
    try {
      const r = await api.executar(acao, config)
      avisar(`iniciado ${acao} (${config}) — pid ${r.pid}`)
      recarregar()
    } catch (e) {
      avisar(String((e as Error).message), true)
    } finally {
      setOcupado(false)
    }
  }

  return (
    <>
      <Cartao titulo="Lançar experimento">
        <Campo rotulo="ação">
          <select value={acao} onChange={(e) => setAcao(e.target.value)}>
            <option value="train">treinar do zero (G1/G4)</option>
            <option value="ajustar">fine-tuning LoRA/QLoRA (G2/G5)</option>
          </select>
        </Campo>
        <Campo rotulo="configuração (configs/)">
          <select value={config} onChange={(e) => setConfig(e.target.value)}>
            <option value="">— escolha —</option>
            {configs.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </Campo>
        <button className="botao botao--primario" disabled={!config || ocupado} onClick={disparar}>
          {ocupado ? '…' : '▶ Executar'}
        </button>
      </Cartao>
      <Cartao titulo="Execuções">
        {execucoes.length === 0 ? <p className="suave">nada em execução.</p> : null}
        {execucoes.map((x) => (
          <div key={x.chave} className="linha" style={{ justifyContent: 'space-between' }}>
            <span className="mono">{x.chave}</span>
            <Distintivo status={x.vivo ? 'aviso' : x.codigo_saida === 0 ? 'ok' : 'perigo'}>
              {x.vivo ? `pid ${x.pid}` : x.codigo_saida === 0 ? 'finalizado' : `falhou (${x.codigo_saida})`}
            </Distintivo>
          </div>
        ))}
      </Cartao>
    </>
  )
}
