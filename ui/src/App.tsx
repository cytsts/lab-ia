import { useState } from 'react'
import { Painel } from './paginas/Painel'
import { DetalheDoRun } from './paginas/DetalheDoRun'
import { Executar } from './paginas/Executar'
import { Eventos } from './paginas/Eventos'
import { CatalogoDS } from './paginas/CatalogoDS'
import { Bancada } from './paginas/Bancada'
import { Comparar } from './paginas/Comparar'
import { Trilha } from './paginas/Trilha'
import { Cadernos } from './paginas/Cadernos'
import { AlternadorDeTema } from './ds/tema'

type Pagina = 'painel' | 'cadernos' | 'run' | 'bancada' | 'executar' | 'comparar' | 'trilha' | 'eventos' | 'ds'

const ABAS: { id: Pagina; rotulo: string }[] = [
  { id: 'painel', rotulo: 'Painel' },
  { id: 'cadernos', rotulo: 'Cadernos' },
  { id: 'bancada', rotulo: 'Bancada' },
  { id: 'executar', rotulo: 'Executar' },
  { id: 'comparar', rotulo: 'Comparar' },
  { id: 'trilha', rotulo: 'Trilha' },
  { id: 'eventos', rotulo: 'Eventos' },
  { id: 'ds', rotulo: 'Design System' },
]

export function App() {
  const [pagina, setPagina] = useState<Pagina>('painel')
  const [runAberto, setRunAberto] = useState<string | null>(null)

  const abrir = (id: string) => {
    setRunAberto(id)
    setPagina('run')
  }

  return (
    <div className="app">
      <header className="topo">
        <span className="mascote" aria-hidden>🦉</span>
        <span className="marca">Lab-IA</span>
        <nav className="nav" aria-label="navegação principal">
          {ABAS.map((aba) => (
            <button
              key={aba.id}
              className={`botao${pagina === aba.id ? ' botao--primario' : ''}`}
              onClick={() => setPagina(aba.id)}
            >
              {aba.rotulo}
            </button>
          ))}
          <AlternadorDeTema />
        </nav>
      </header>
      <main>
        {pagina === 'painel' && <Painel abrirRun={abrir} />}
        {pagina === 'cadernos' && <Cadernos />}
        {pagina === 'run' && runAberto && <DetalheDoRun id={runAberto} />}
        {pagina === 'bancada' && <Bancada />}
        {pagina === 'executar' && <Executar />}
        {pagina === 'comparar' && <Comparar />}
        {pagina === 'trilha' && <Trilha />}
        {pagina === 'eventos' && <Eventos />}
        {pagina === 'ds' && <CatalogoDS />}
      </main>
      <footer className="rodape">
        <span>laboratório de IA em pt-BR · núcleo local em 127.0.0.1:8765</span>
        <button className="botao" onClick={() => setPagina('painel')}>← voltar</button>
      </footer>
    </div>
  )
}
