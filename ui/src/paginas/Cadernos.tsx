import { useState } from 'react'
import { Cartao, Distintivo } from '../ds/componentes'

/** Entrada da experiência central: os experimentos vivem no JupyterLab local. */
export function Cadernos() {
  const [abrindo, setAbrindo] = useState(false)
  const [erro, setErro] = useState<string | null>(null)

  const abrir = async () => {
    if (!window.labia?.abrirCadernos) {
      setErro('Abra pelo Electron ou use o comando abaixo no terminal do projeto.')
      return
    }
    setAbrindo(true)
    setErro(null)
    try {
      const resultado = await window.labia.abrirCadernos()
      if (!resultado.ok) setErro(resultado.erro ?? 'não foi possível iniciar os cadernos')
    } catch (e) {
      setErro(e instanceof Error ? e.message : String(e))
    } finally {
      setAbrindo(false)
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <header style={{ marginBottom: 18 }}>
        <h1 style={{ margin: 0 }}>Cadernos</h1>
        <p className="suave">
          Seu espaço para investigar ideias: escreva Python e Markdown, execute células na ordem que decidir e mantenha
          variáveis, modelos e gráficos na mesma sessão.
        </p>
      </header>

      <Cartao titulo="Abrir laboratório interativo">
        <p>
          O Lab-IA inicia um JupyterLab local em uma janela separada. Ele já usa o Python do projeto, então você pode
          importar <code className="mono">labia</code>, PyTorch e os seus dados sem criar uma ponte nova para cada experimento.
        </p>
        <button type="button" className="botao botao--primario" onClick={abrir} disabled={abrindo}>
          {abrindo ? 'Iniciando JupyterLab…' : 'Abrir cadernos'}
        </button>
        {erro ? <p className="aviso aviso--erro" role="alert">{erro}</p> : null}
      </Cartao>

      <Cartao titulo="Como o espaço é organizado">
        <ul>
          <li><code className="mono">cadernos/</code>: seus notebooks autorais.</li>
          <li><code className="mono">cadernos/exemplos/00-primeiro-experimento.ipynb</code>: ponto de partida copiável.</li>
          <li><code className="mono">cadernos/resultados/</code>: checkpoints, tabelas e artefatos que você decidir salvar.</li>
        </ul>
        <p>
          <Distintivo status="info">Sessão</Distintivo>{' '}
          salvar um notebook preserva código e saídas. Reiniciar o kernel limpa memória e GPU; para continuar um modelo,
          salve e carregue um checkpoint explicitamente.
        </p>
      </Cartao>

      <Cartao titulo="Também pelo terminal">
        <pre className="mono">.venv\Scripts\python -m labia.cli laboratorio</pre>
        <p className="suave">
          Para só criar as pastas e o exemplo: <code className="mono">lab-ia laboratorio --sem-abrir</code>.
        </p>
      </Cartao>
    </div>
  )
}
