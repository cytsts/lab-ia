import { useState } from 'react'
import { BarraProgresso, Campo, Cartao, Distintivo } from '../ds/componentes'
import { BarrasDeUso, GraficoDeMetricas } from '../ds/graficos'
import { useAvisos, ProvedorDeAvisos } from '../ds/avisos'
import { AlternadorDeTema } from '../ds/tema'

function exemplar() {
  return {
    series: [
      { nome: 'treino', cor: 'var(--serie-a)', pontos: [0, 1, 2, 3, 4, 5].map((x) => ({ x, y: 6 - x * 0.9 })) },
      { nome: 'validação', cor: 'var(--serie-b)', pontos: [0, 1, 2, 3, 4, 5].map((x) => ({ x, y: 6.2 - x * 0.8 })) },
    ],
  }
}

/** Catálogo vivo do Design System (RF3). */
export function CatalogoDS() {
  const [vazio, setVazio] = useState(false)
  const { avisar } = useAvisos()
  const { series } = exemplar()
  return (
    <>
      <Cartao titulo="Fundos">
        <p className="suave">
          Tokens em <code>src/ds/tokens.css</code>: paleta clara/escura AA, tipografia, espaçamento, raio e sombra.
        </p>
        <div className="linha">
          {['--primaria', '--secundaria', '--ok', '--aviso', '--perigo', '--info'].map((t) => (
            <span key={t} style={{ width: 44, height: 44, borderRadius: 10, background: `var(${t})`, display: 'inline-block' }} title={t} />
          ))}
        </div>
      </Cartao>
      <Cartao titulo="Botões">
        <div className="linha">
          <button className="botao">Neutro</button>
          <button className="botao botao--primario">Primário</button>
          <button className="botao" disabled>Desativado</button>
          <AlternadorDeTema />
        </div>
      </Cartao>
      <Cartao titulo="Distintivos de estado">
        <div className="linha">
          <Distintivo status="ok">concluído</Distintivo>
          <Distintivo status="aviso">em andamento</Distintivo>
          <Distintivo status="perigo">falhou</Distintivo>
          <Distintivo status="info">quantizado</Distintivo>
        </div>
      </Cartao>
      <Cartao titulo="Progresso e campos">
        <BarraProgresso atual={62} total={100} />
        <br />
        <Campo rotulo=" slider de exemplo">
          <input type="range" min={1} max={100} defaultValue={42} aria-label="deslizador" />
        </Campo>
        <button className="botao" onClick={() => setVazio(!vazio)}>
          alternar gráfico vazio (estado limite)
        </button>
      </Cartao>
      <Cartao titulo="Gráficos">
        <GraficoDeMetricas titulo="Curva exemplar" series={vazio ? [] : series} />
        <BarrasDeUso uso={[0.34, 0.26, 0.22, 0.18]} />
      </Cartao>
      <Cartao titulo="Microinterações — avisos">
        <div className="linha">
          <button className="botao" onClick={() => avisar('tudo certo por aqui ✨')}>aviso normal</button>
          <button className="botao" onClick={() => avisar('ops, algo falhou', true)}>aviso de erro</button>
        </div>
      </Cartao>
    </>
  )
}

/* reexporta para permitir uso isolado em testes sem o Provedor */
export { ProvedorDeAvisos }
