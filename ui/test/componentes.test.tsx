import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { BarrasDeUso, GraficoDeMetricas } from '../src/ds/graficos'
import { BarraProgresso, Distintivo } from '../src/ds/componentes'
import { AlternadorDeTema } from '../src/ds/tema'
import { Painel } from '../src/paginas/Painel'
import { api } from '../src/api'

vi.mock('../src/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/api')>()
  return { ...real, api: { corre: vi.fn(), metricas: vi.fn(), estado: vi.fn(), tamanhos: vi.fn(), comparativo: vi.fn(), configs: vi.fn(), eventos: vi.fn(), execucoes: vi.fn(), executar: vi.fn() }, baseDaApi: () => 'http://api.falsa' }
})

beforeEach(() => vi.clearAllMocks())

describe('GraficoDeMetricas', () => {
  it('renderiza uma polilinha por série com pontos (CA3)', () => {
    render(
      <GraficoDeMetricas
        titulo="Curva de perda"
        series={[
          { nome: 'treino', cor: 'red', pontos: [{ x: 0, y: 3 }, { x: 1, y: 2 }] },
          { nome: 'validação', cor: 'blue', pontos: [{ x: 0, y: 3.2 }, { x: 1, y: 2.4 }] },
        ]}
      />,
    )
    const linhas = document.querySelectorAll('polyline')
    expect(linhas).toHaveLength(2)
    expect(linhas[0].getAttribute('points')).toContain(',')
    expect(screen.getByText('Curva de perda')).toBeInTheDocument()
    expect(screen.getByText('validação')).toBeInTheDocument()
  })

  it('estado vazio amigável', () => {
    render(<GraficoDeMetricas titulo="nada" series={[]} />)
    expect(screen.getByText(/sem dados/)).toBeInTheDocument()
  })

  it('barras de uso por especialista', () => {
    render(<BarrasDeUso uso={[0.5, 0.25, 0.125, 0.125]} />)
    expect(screen.getAllByText(/%/).length).toBe(4)
  })
})

describe('componentes básicos', () => {
  it('distintivo e progresso expõem acessibilidade', () => {
    render(
      <div>
        <Distintivo status="ok">concluído</Distintivo>
        <BarraProgresso atual={50} total={200} />
      </div>,
    )
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25')
    expect(screen.getByText('concluído')).toHaveClass('distintivo--ok')
  })

  it('alternador de tema grava e aplica data-tema (CA4)', async () => {
    const usuario = userEvent.setup()
    document.documentElement.dataset.tema = 'claro'
    render(<AlternadorDeTema />)
    await usuario.click(screen.getByRole('button', { name: /alternar tema/i }))
    expect(document.documentElement.dataset.tema).toBe('escuro')
    expect(localStorage.getItem('labia-tema')).toBe('escuro')
  })
})

describe('Painel', () => {
  it('lista runs com badges vindos da API (CA2)', async () => {
    vi.mocked(api.corre).mockResolvedValue([
      { run_id: 'g1-treino-zero', concluido: true, passo: 2500, passos_totais: 2500 },
      { run_id: 'g4-moe-zero', passo: 1000, passos_totais: 2500, concluido: false },
    ] as never)
    render(<Painel abrirRun={vi.fn()} />)
    expect(await screen.findByText('g1-treino-zero')).toBeInTheDocument()
    expect(screen.getByText('g4-moe-zero')).toBeInTheDocument()
    expect(screen.getByText('concluído')).toBeInTheDocument()
    expect(screen.getByText('em andamento')).toBeInTheDocument()
  })

  it('mostra erro quando o núcleo está offline', async () => {
    vi.mocked(api.corre).mockRejectedValue(new Error('connection refused'))
    render(<Painel abrirRun={vi.fn()} />)
    expect(await screen.findByText(/núcleo indisponível/)).toBeInTheDocument()
  })
})
