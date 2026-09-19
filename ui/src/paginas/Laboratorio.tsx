import { useEffect, useState } from 'react'
import { api, type Saude } from '../api'
import { SubAbaConfigurar } from './laboratorio/SubAbaConfigurar'
import { SubAbaCurvas } from './laboratorio/SubAbaCurvas'
import { SubAbaDados } from './laboratorio/SubAbaDados'
import { SubAbaExecutar } from './laboratorio/SubAbaExecutar'
import { SubAbaMedir } from './laboratorio/SubAbaMedir'
import { SubAbaTestar } from './laboratorio/SubAbaTestar'

export type SubAbaLaboratorio = 'dados' | 'configurar' | 'executar' | 'curvas' | 'testar' | 'medir'

const SUB_ABAS: { id: SubAbaLaboratorio; rotulo: string; descricao: string }[] = [
  { id: 'dados', rotulo: '1. Dados', descricao: 'corpus, procedência e navegador CoT' },
  { id: 'configurar', rotulo: '2. Configurar', descricao: 'do zero ou refino (GGUF/runs)' },
  { id: 'executar', rotulo: '3. Executar', descricao: 'log ao vivo, progresso e parada' },
  { id: 'curvas', rotulo: '4. Curvas', descricao: 'perda, diagnóstico e deriva' },
  { id: 'testar', rotulo: '5. Testar', descricao: 'perguntas e exploração de CoT' },
  { id: 'medir', rotulo: '6. Medir', descricao: 'benchmark padronizado e comparativo' },
]

export function Laboratorio() {
  const [subAba, setSubAba] = useState<SubAbaLaboratorio>('dados')
  const [saude, setSaude] = useState<Saude | null>(null)
  const [nucleoFora, setNucleoFora] = useState<boolean>(false)
  const [configParaExecutar, setConfigParaExecutar] = useState<string>('')

  // Conferir saúde do núcleo (RNF7, CA9)
  const conferirSaude = () => {
    api.saude()
      .then((s) => {
        setSaude(s)
        setNucleoFora(false)
      })
      .catch(() => {
        setSaude(null)
        setNucleoFora(true)
      })
  }

  useEffect(() => {
    conferirSaude()
    const t = setInterval(conferirSaude, 5000)
    return () => clearInterval(t)
  }, [])

  const transicaoParaExecutar = (nomeConfig: string) => {
    setConfigParaExecutar(nomeConfig)
    setSubAba('executar')
  }

  // Se o núcleo estiver fora do ar (CA9)
  if (nucleoFora) {
    return (
      <div className="nucleo-indisponivel" style={{ padding: 24, maxWidth: 640, margin: '0 auto' }}>
        <div
          className="aviso aviso--erro"
          role="alert"
          style={{ padding: 20, borderRadius: 8, borderLeft: '6px solid var(--perigo, #ef4444)' }}
        >
          <h2 style={{ margin: '0 0 10px 0', fontSize: '1.2rem' }}>
            ⚠ Núcleo do Lab-IA Indisponível
          </h2>
          <p style={{ margin: '0 0 12px 0', lineHeight: 1.5 }}>
            A interface visual não conseguiu se comunicar com a API do núcleo local em{' '}
            <code className="mono">127.0.0.1:8765</code>.
          </p>
          <div style={{ marginBottom: 12 }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>
              Para iniciar o serviço manualmente no terminal:
            </span>
            <pre
              className="mono"
              style={{
                background: '#090d13',
                color: '#38bdf8',
                padding: '8px 12px',
                borderRadius: 6,
                marginTop: 6,
              }}
            >
              .venv\Scripts\python -m labia.cli servir
            </pre>
          </div>
          <button type="button" className="botao botao--primario" onClick={conferirSaude}>
            ↻ Tentar reconectar agora
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="pagina-laboratorio">
      <header style={{ marginBottom: 18 }}>
        <div className="linha" style={{ justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '1.4rem' }}>
              🔬 Laboratório de Modelos
            </h1>
            <span className="suave" style={{ fontSize: 'var(--miudo)' }}>
              Ciclo interativo encadeado estilo caderno · do dado ao raciocínio medido
            </span>
          </div>

          {saude ? (
            <div className="linha" style={{ gap: 8, fontSize: '0.8rem' }}>
              <span className="suave">GPU: {saude.gpu || 'CPU'}</span>
              <span className="suave">· {saude.runs} runs</span>
              <span className="suave">· {saude.datasets} datasets</span>
            </div>
          ) : null}
        </div>

        {/* Barra de Sub-abas Encadeadas (RF1.1) */}
        <nav
          className="linha sub-abas"
          style={{
            gap: 6,
            background: 'var(--superficie-2, #181c24)',
            padding: 6,
            borderRadius: 8,
            overflowX: 'auto',
          }}
          aria-label="sub-abas do laboratório"
        >
          {SUB_ABAS.map((aba) => {
            const ativa = subAba === aba.id
            return (
              <button
                key={aba.id}
                type="button"
                className={`botao${ativa ? ' botao--primario' : ''}`}
                onClick={() => setSubAba(aba.id)}
                style={{
                  padding: '6px 14px',
                  fontSize: '0.85rem',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                }}
              >
                <strong>{aba.rotulo}</strong>
                <span
                  style={{
                    fontSize: '0.7rem',
                    opacity: ativa ? 0.9 : 0.6,
                    fontWeight: 'normal',
                  }}
                >
                  {aba.descricao}
                </span>
              </button>
            )
          })}
        </nav>
      </header>

      {/* Conteúdo da Sub-aba Ativa */}
      <main className="sub-aba-conteudo">
        {subAba === 'dados' && <SubAbaDados />}
        {subAba === 'configurar' && <SubAbaConfigurar aoIrParaExecutar={transicaoParaExecutar} />}
        {subAba === 'executar' && <SubAbaExecutar configPreSelecionada={configParaExecutar} />}
        {subAba === 'curvas' && <SubAbaCurvas />}
        {subAba === 'testar' && <SubAbaTestar />}
        {subAba === 'medir' && <SubAbaMedir />}
      </main>
    </div>
  )
}
