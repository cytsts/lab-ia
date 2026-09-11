import { useState } from 'react'
import { Painel } from './paginas/Painel'
import { DetalheDoRun } from './paginas/DetalheDoRun'
import { Executar } from './paginas/Executar'
import { Eventos } from './paginas/Eventos'
import { CatalogoDS } from './paginas/CatalogoDS'
import { AlternadorDeTema } from './ds/tema'

type Pagina = 'painel' | 'run' | 'executar' | 'eventos' | 'ds'

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
          <button className={`botao${pagina === 'painel' ? ' botao--primario' : ''}`} onClick={() => setPagina('painel')}>Painel</button>
          <button className={`botao${pagina === 'executar' ? ' botao--primario' : ''}`} onClick={() => setPagina('executar')}>Executar</button>
          <button className={`botao${pagina === 'eventos' ? ' botao--primario' : ''}`} onClick={() => setPagina('eventos')}>Eventos</button>
          <button className={`botao${pagina === 'ds' ? ' botao--primario' : ''}`} onClick={() => setPagina('ds')}>Design System</button>
          <AlternadorDeTema />
        </nav>
      </header>
      <main>
        {pagina === 'painel' && <Painel abrirRun={abrir} />}
        {pagina === 'run' && runAberto && <DetalheDoRun id={runAberto} />}
        {pagina === 'executar' && <Executar />}
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
