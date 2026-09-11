import { describe, expect, it, vi, beforeEach } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ProvedorDeAvisos, useAvisos } from '../src/ds/avisos'
import { App } from '../src/App'
import { CatalogoDS } from '../src/paginas/CatalogoDS'
import { DetalheDoRun } from '../src/paginas/DetalheDoRun'
import { Eventos } from '../src/paginas/Eventos'
import { Executar } from '../src/paginas/Executar'
import { api, baseDaApi, API_BASE_PADRAO } from '../src/api'

vi.mock('../src/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/api')>()
  return {
    ...real,
    baseDaApi: () => localStorage.getItem('labia-api') ?? real.API_BASE_PADRAO,
    api: {
      corre: vi.fn(), metricas: vi.fn(), estado: vi.fn(), tamanhos: vi.fn(),
      comparativo: vi.fn(), configs: vi.fn(), eventos: vi.fn(), execucoes: vi.fn(), executar: vi.fn(),
    },
  }
})

beforeEach(() => vi.clearAllMocks())

describe('api', () => {
  it('usa override do localStorage e padrão local', () => {
    expect(baseDaApi()).toBe(API_BASE_PADRAO)
    localStorage.setItem('labia-api', 'http://exemplo:1')
    expect(baseDaApi()).toBe('http://exemplo:1')
  })
})

describe('Avisos', () => {
  function Alarme() {
    const { avisar } = useAvisos()
    return <button onClick={() => avisar('oi do teste')}>dispara</button>
  }
  it('mostra e some sozinho', () => {
    vi.useFakeTimers()
    render(
      <ProvedorDeAvisos>
        <Alarme />
      </ProvedorDeAvisos>,
    )
    fireEvent.click(screen.getByText('dispara'))
    expect(screen.getByRole('status')).toHaveTextContent('oi do teste')
    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})

describe('DetalheDoRun', () => {
  it('renderiza curva, tabela de quantização e comparativo', async () => {
    vi.mocked(api.estado).mockResolvedValue({ run_id: 'x', concluido: true } as never)
    vi.mocked(api.metricas).mockResolvedValue([
      { passo: 0, loss_val: 8.3 },
      { passo: 10, loss_val: 5.1, loss_trem: 5.2 },
      { passo: 20, loss_val: 4.6, loss_trem: 4.7 },
    ] as never)
    vi.mocked(api.tamanhos).mockResolvedValue({
      modo: 'int8', fator_alvos: 3.95, bits_efetivos_por_parametro: 8.09,
      bytes_fp32_alvos: 4_700_000, bytes_quantizados_alvos: 1_190_000,
      perda_val_antes: 8.729, perda_val_depois: 8.732,
    } as never)
    vi.mocked(api.comparativo).mockResolvedValue({ cot_menos_direta_pp: 5.3 } as never)
    render(<DetalheDoRun id="x" />)
    expect(await screen.findByText('Curva de perda')).toBeInTheDocument()
    expect(document.querySelectorAll('polyline')).toHaveLength(2)
    expect(screen.getByText('Quantização')).toBeInTheDocument()
    expect(screen.getByText(/3\.95×/)).toBeInTheDocument()
    expect(screen.getByText('Benchmark de raciocínio')).toBeInTheDocument()
  })

  it('run sem métricas mostra estado vazio', async () => {
    vi.mocked(api.estado).mockResolvedValue({ run_id: 'y', concluido: false } as never)
    vi.mocked(api.metricas).mockResolvedValue([] as never)
    render(<DetalheDoRun id="y" />)
    expect(await screen.findByText(/sem métricas/)).toBeInTheDocument()
  })
})

describe('Eventos', () => {
  it('mostra linhas do feed', async () => {
    vi.mocked(api.eventos).mockResolvedValue([
      { tempo: '2026-09-11T12:00:00+00:00', tipo: 'treino_iniciado', run: 'g1' },
    ] as never)
    render(<Eventos />)
    expect(await screen.findByText('treino_iniciado')).toBeInTheDocument()
    expect(screen.getByText(/g1/)).toBeInTheDocument()
  })
})

describe('Executar', () => {
  it('lança execução válida via API', async () => {
    vi.mocked(api.configs).mockResolvedValue(['g1_treino_zero.yaml'] as never)
    vi.mocked(api.execucoes).mockResolvedValue([{ chave: 'train-a', pid: 1, vivo: true, codigo_saida: null }] as never)
    vi.mocked(api.executar).mockResolvedValue({ chave: 'train-a', pid: 1 } as never)
    render(
      <ProvedorDeAvisos>
        <Executar />
      </ProvedorDeAvisos>,
    )
    await waitFor(() => expect(screen.getByText('train-a')).toBeInTheDocument())
    const usuario = userEvent.setup()
    await usuario.selectOptions(screen.getByRole('combobox', { name: /configuração/i }), 'g1_treino_zero.yaml')
    await usuario.click(screen.getByRole('button', { name: /executar/i }))
    await waitFor(() => expect(api.executar).toHaveBeenCalledWith('train', 'g1_treino_zero.yaml'))
  })

  it('mostra erro do servidor como aviso', async () => {
    vi.mocked(api.configs).mockResolvedValue([] as never)
    vi.mocked(api.execucoes).mockResolvedValue([] as never)
    vi.mocked(api.executar).mockRejectedValue(new Error('ação inválida'))
    render(
      <ProvedorDeAvisos>
        <Executar />
      </ProvedorDeAvisos>,
    )
    expect(await screen.findByText(/nada em execução/)).toBeInTheDocument()
  })
})

describe('App (integração de navegação)', () => {
  it('navega entre páginas pelo topo', async () => {
    vi.mocked(api.eventos).mockResolvedValue([] as never)
    vi.mocked(api.configs).mockResolvedValue([] as never)
    vi.mocked(api.execucoes).mockResolvedValue([] as never)
    vi.mocked(api.corre).mockResolvedValue([{ run_id: 'g1', concluido: true, passo: 1, passos_totais: 1 }] as never)
    render(
      <ProvedorDeAvisos>
        <App />
      </ProvedorDeAvisos>,
    )
    const usuario = userEvent.setup()
    expect(await screen.findByText('Lab-IA')).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: 'Eventos' }))
    expect(screen.getByText(/Log de eventos/)).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: 'Design System' }))
    expect(screen.getByText('Fundos')).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: 'Painel' }))
    expect(await screen.findByText('g1')).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: /abrir run g1/ }))
    expect(await screen.findByText(/sem métricas/)).toBeInTheDocument()
  })
})

describe('CatalogoDS', () => {
  it('interage com estado limite do gráfico', async () => {
    render(
      <ProvedorDeAvisos>
        <CatalogoDS />
      </ProvedorDeAvisos>,
    )
    expect(screen.getByText('Curva exemplar')).toBeInTheDocument()
    const usuario = userEvent.setup()
    await usuario.click(screen.getByRole('button', { name: /alternar gráfico vazio/ }))
    expect(screen.getByText(/sem dados/)).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: /aviso normal/ }))
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})
