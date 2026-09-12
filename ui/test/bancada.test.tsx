import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ProvedorDeAvisos } from '../src/ds/avisos'
import { Bancada } from '../src/paginas/Bancada'
import { Comparar } from '../src/paginas/Comparar'
import { api } from '../src/api'

vi.mock('../src/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/api')>()
  return {
    ...real,
    api: {
      corre: vi.fn(),
      executar: vi.fn(),
      saude: vi.fn(),
      datasets: vi.fn(),
      presets: vi.fn(),
      prepararDados: vi.fn(),
      novo: vi.fn(),
      comparar: vi.fn(),
      urlCurvas: vi.fn(() => 'http://127.0.0.1:8765/comparar/curvas.svg?runs=a'),
      configs: vi.fn(),
      varreduras: vi.fn(),
      varrerSeco: vi.fn(),
      varrer: vi.fn(),
    },
  }
})

const SAUDE = {
  ok: true, versao_api: '0.2.0', raiz: 'K:\\lab', cuda: true, gpu: 'RTX 3070',
  runs: 3, datasets: 1, configs: 2, guia: ['bancada.md'],
}

const DATASETS = [
  { id: 'livros-ptbr', chars_trem: 800978, chars_val: 41138, idioma: 'pt', tokens_aprox_trem: 222493, fontes: ['x.txt'] },
]

const PRESETS = [
  { nome: 'rapido', descricao: 'primeiro treino de verdade', modelo: { dim: 192 }, passos: 1000, lote: 24 },
  { nome: 'equilibrado', descricao: 'linha de base', modelo: { dim: 256 }, passos: 2500, lote: 32 },
]

const NOVO = {
  caminho: 'K:\\lab\\configs\\meu-run.yaml',
  caminho_relativo: 'configs\\meu-run.yaml',
  arquivo_config: 'meu-run.yaml',
  resumo: 'run     : meu-run\ntempo   : ~14,1 s',
  config: {}, parametros: { total: 2600064, ativos: 2600064 },
  desempenho: { segundos_previstos: 14.1, tokens_por_s_previsto: 326898, segundos_por_passo: 0.0141 },
  vram: { mb_total: 193.8 }, epocas: 10.42, yaml: 'nome: meu-run',
}

const COMPARACAO = {
  runs: [
    {
      run_id: 'bom', familia: 'treino', passos: 1000, parametros: 5847040, melhor_val: 4.6259,
      passo_melhor_val: 1500, val_final: 4.6736, drift: 0.0477, melhor_val_orcamento_comum: 5.0256,
      tokens_por_s: 294180, diagnostico: 'deteriorando', comentario: 'a validação subiu nas últimas avaliações',
    },
    {
      run_id: 'ruim', familia: 'treino', passos: 2500, parametros: null, melhor_val: 4.6996,
      passo_melhor_val: 1250, val_final: 4.9687, drift: 0.2691, melhor_val_orcamento_comum: null,
      tokens_por_s: null, diagnostico: 'overfit', comentario: 'passou a decorar o corpus',
    },
  ],
  orcamento_comum: 1000,
  familias: ['treino'],
  ranking: ['bom', 'ruim'],
  melhor: 'bom',
  veredito: 'bom generaliza melhor: val 4.6259 contra 4.6996 de ruim.',
  limiares: { drift_overfit: 0.2, estagnacao: 0.01, salto_instavel: 1 },
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.saude).mockResolvedValue(SAUDE as never)
  vi.mocked(api.datasets).mockResolvedValue(DATASETS as never)
  vi.mocked(api.presets).mockResolvedValue(PRESETS as never)
  vi.mocked(api.corre).mockResolvedValue([{ run_id: 'bom' }, { run_id: 'ruim' }] as never)
  vi.mocked(api.configs).mockResolvedValue(['livros-rapido.yaml'] as never)
  vi.mocked(api.varreduras).mockResolvedValue([] as never)
})

const montar = (no: React.ReactElement) => render(<ProvedorDeAvisos>{no}</ProvedorDeAvisos>)

describe('Bancada', () => {
  it('mostra o estado do núcleo e os datasets prontos', async () => {
    montar(<Bancada />)
    expect(await screen.findByText('RTX 3070')).toBeInTheDocument()
    // o id do dataset aparece na tabela E no seletor do formulário seguinte
    expect(await screen.findAllByText('livros-ptbr')).toHaveLength(2)
    expect(screen.getByText(/222493/)).toBeInTheDocument()
  })

  it('avisa quando o núcleo está fora do ar', async () => {
    vi.mocked(api.saude).mockRejectedValue(new Error('fetch failed'))
    montar(<Bancada />)
    expect(await screen.findByText('núcleo indisponível')).toBeInTheDocument()
    expect(screen.getByText(/lab-ia servir/)).toBeInTheDocument()
  })

  it('prepara um dataset a partir das fontes digitadas', async () => {
    vi.mocked(api.prepararDados).mockResolvedValue({ limpeza: { paragrafos_finais: 12 }, saidas: { trem: { chars: 900 } } } as never)
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.type(screen.getByLabelText('fontes de dados'), 'C:\\a.txt\nC:\\b.md')
    await usuario.type(screen.getByLabelText('id do dataset'), 'meu')
    await usuario.click(screen.getByRole('button', { name: /preparar dataset/i }))
    await waitFor(() =>
      expect(api.prepararDados).toHaveBeenCalledWith({
        fontes: ['C:\\a.txt', 'C:\\b.md'], id: 'meu', frac_val: 0.05, min_chars: 20,
      }),
    )
    expect(await screen.findByRole('status')).toHaveTextContent(/12 parágrafos/)
  })

  it('exige id e fontes antes de preparar', async () => {
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.click(screen.getByRole('button', { name: /preparar dataset/i }))
    expect(api.prepararDados).not.toHaveBeenCalled()
    expect(await screen.findByRole('status')).toHaveTextContent(/informe o id/)
  })

  it('gera a config e oferece treinar na sequência', async () => {
    vi.mocked(api.novo).mockResolvedValue(NOVO as never)
    vi.mocked(api.executar).mockResolvedValue({ chave: 'train-meu-run.yaml', pid: 42 } as never)
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.selectOptions(await screen.findByLabelText('dataset'), 'livros-ptbr')
    await usuario.type(screen.getByLabelText('nome do experimento'), 'meu-run')
    await usuario.selectOptions(screen.getByLabelText('preset'), 'rapido')
    await usuario.click(screen.getByRole('button', { name: /gerar config/i }))
    await waitFor(() =>
      expect(api.novo).toHaveBeenCalledWith({ nome: 'meu-run', dados: 'livros-ptbr', preset: 'rapido' }),
    )
    expect(await screen.findByText(/~14,1 s/)).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: /treinar agora/i }))
    await waitFor(() => expect(api.executar).toHaveBeenCalledWith('train', 'meu-run.yaml'))
  })

  it('mostra o erro do núcleo como aviso ao gerar config', async () => {
    vi.mocked(api.novo).mockRejectedValue(new Error('dim (100) precisa ser divisível por cabecas (8)'))
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.selectOptions(await screen.findByLabelText('dataset'), 'livros-ptbr')
    await usuario.type(screen.getByLabelText('nome do experimento'), 'x')
    await usuario.click(screen.getByRole('button', { name: /gerar config/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/divisível/)
  })
})

describe('Comparar', () => {
  it('compara os runs escolhidos e mostra tabela, veredito e curvas', async () => {
    vi.mocked(api.comparar).mockResolvedValue(COMPARACAO as never)
    const usuario = userEvent.setup()
    montar(<Comparar />)
    expect(await screen.findByLabelText('incluir bom')).toBeInTheDocument()
    await usuario.click(screen.getByRole('button', { name: /^comparar$/i }))
    await waitFor(() => expect(api.comparar).toHaveBeenCalledWith(['bom', 'ruim']))
    expect(await screen.findByText('overfit')).toBeInTheDocument()
    expect(screen.getByText(/bom generaliza melhor/)).toBeInTheDocument()
    expect(screen.getByText(/passou a decorar o corpus/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /curvas de perda/i })).toHaveAttribute(
      'src', expect.stringContaining('curvas.svg'),
    )
    expect(screen.getByText(/orçamento de 1000 passos/)).toBeInTheDocument()
  })

  it('compara apenas os runs marcados', async () => {
    vi.mocked(api.comparar).mockResolvedValue({ ...COMPARACAO, runs: [COMPARACAO.runs[1]] } as never)
    const usuario = userEvent.setup()
    montar(<Comparar />)
    await usuario.click(await screen.findByLabelText('incluir bom'))
    await usuario.click(screen.getByRole('button', { name: /^comparar$/i }))
    await waitFor(() => expect(api.comparar).toHaveBeenCalledWith(['ruim']))
  })

  it('mostra o erro quando não há run comparável', async () => {
    vi.mocked(api.comparar).mockRejectedValue(new Error('nenhum run com métricas em runs/'))
    const usuario = userEvent.setup()
    montar(<Comparar />)
    await usuario.click(await screen.findByRole('button', { name: /^comparar$/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/nenhum run com métricas/)
  })
})
describe('Bancada — varredura', () => {
  const SECO = {
    prefixo: 'varrer-livros-rapido',
    base: 'livros-rapido.yaml',
    modo: 'grade',
    passos_por_variante: 500,
    semente: 42,
    grades: { lr: ['0.00015', '0.0003'] },
    variantes: [
      { indice: 1, run_id: 'varrer-livros-rapido-01', sobrecritas: { lr: 0.00015 } },
      { indice: 2, run_id: 'varrer-livros-rapido-02', sobrecritas: { lr: 0.0003 } },
    ],
    falhas: [],
    tabela: 'varredura : varrer-livros-rapido\nrun  lr  melhor val',
  }

  it('confere a grade antes de gastar GPU', async () => {
    vi.mocked(api.varrerSeco).mockResolvedValue(SECO as never)
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.selectOptions(await screen.findByLabelText('config base'), 'livros-rapido.yaml')
    await usuario.type(screen.getByLabelText('grades da varredura'), 'lr=0.00015,0.0003')
    await usuario.click(screen.getByRole('button', { name: /conferir grade/i }))
    await waitFor(() =>
      expect(api.varrerSeco).toHaveBeenCalledWith({
        base: 'livros-rapido.yaml',
        grades: ['lr=0.00015,0.0003'],
        passos: undefined,
      }),
    )
    expect(await screen.findByText(/varredura : varrer-livros-rapido/)).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/2 combinação/)
  })

  it('avisa quando alguma combinação é impossível', async () => {
    vi.mocked(api.varrerSeco).mockResolvedValue({ ...SECO, falhas: ['varrer-livros-rapido-02'] } as never)
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.selectOptions(await screen.findByLabelText('config base'), 'livros-rapido.yaml')
    await usuario.type(screen.getByLabelText('grades da varredura'), 'dim=64,65')
    await usuario.click(screen.getByRole('button', { name: /conferir grade/i }))
    expect(await screen.findByRole('status')).toHaveTextContent(/impossível/)
  })

  it('exige base e grade', async () => {
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.click(screen.getByRole('button', { name: /conferir grade/i }))
    expect(api.varrerSeco).not.toHaveBeenCalled()
    expect(await screen.findByRole('status')).toHaveTextContent(/config base/)
  })

  it('dispara a varredura com as grades digitadas', async () => {
    vi.mocked(api.varrer).mockResolvedValue({ chave: 'varrer-sw', pid: 77, prefixo: 'sw' } as never)
    const usuario = userEvent.setup()
    montar(<Bancada />)
    await usuario.selectOptions(await screen.findByLabelText('config base'), 'livros-rapido.yaml')
    await usuario.type(screen.getByLabelText('grades da varredura'), 'lr=0.00015,0.0003\nlote=16,32')
    await usuario.type(screen.getByLabelText('passos por variante'), '500')
    await usuario.click(screen.getByRole('button', { name: /rodar varredura/i }))
    await waitFor(() =>
      expect(api.varrer).toHaveBeenCalledWith({
        base: 'livros-rapido.yaml',
        grades: ['lr=0.00015,0.0003', 'lote=16,32'],
        passos: 500,
      }),
    )
    expect(await screen.findByRole('status')).toHaveTextContent(/varredura sw iniciada/)
  })

  it('lista as varreduras já feitas', async () => {
    vi.mocked(api.varreduras).mockResolvedValue([
      { prefixo: 'sw-lr', base: 'livros-rapido.yaml', melhor: { run_id: 'sw-lr-06', melhor_val: 5.131 }, variantes: 6, falhas: [] },
    ] as never)
    montar(<Bancada />)
    expect(await screen.findByText('sw-lr')).toBeInTheDocument()
    expect(screen.getByText('sw-lr-06')).toBeInTheDocument()
    expect(screen.getByText('5.1310')).toBeInTheDocument()
  })
})
