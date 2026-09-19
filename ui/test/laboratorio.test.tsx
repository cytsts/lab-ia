import { describe, expect, it, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { ProvedorDeAvisos } from '../src/ds/avisos'
import { Laboratorio } from '../src/paginas/Laboratorio'
import { api } from '../src/api'

vi.mock('../src/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/api')>()
  return {
    ...real,
    api: {
      saude: vi.fn(),
      corre: vi.fn(),
      metricas: vi.fn(),
      estado: vi.fn(),
      datasets: vi.fn(),
      dataset: vi.fn(),
      itensDataset: vi.fn(),
      presets: vi.fn(),
      novo: vi.fn(),
      modelos: vi.fn(),
      configs: vi.fn(),
      execucoes: vi.fn(),
      executar: vi.fn(),
      pararExecucao: vi.fn(),
      logExecucao: vi.fn(),
      progresso: vi.fn(),
      testar: vi.fn(),
      benchmarksDoRun: vi.fn(),
      dispararBenchmark: vi.fn(),
      comparar: vi.fn(),
    },
  }
})

beforeEach(() => {
  vi.clearAllMocks()

  vi.mocked(api.saude).mockResolvedValue({
    ok: true,
    versao_api: '0.2.0',
    raiz: 'K:/Dev/lab-ia',
    cuda: true,
    gpu: 'NVIDIA GeForce RTX 3070',
    runs: 16,
    datasets: 2,
    configs: 5,
    guia: [],
  })

  vi.mocked(api.datasets).mockResolvedValue([
    {
      id: 'logica-pq',
      chars_trem: 989705,
      chars_val: 49253,
      idioma: 'pt',
      tokens_aprox_trem: 274918,
      fontes: [],
    },
    {
      id: 'livros-ptbr',
      chars_trem: 832561,
      chars_val: 42754,
      idioma: 'pt',
      tokens_aprox_trem: 231266,
      fontes: [],
    },
  ])

  vi.mocked(api.dataset).mockResolvedValue({
    id: 'logica-pq',
    manifesto: {
      versao: 1,
      parametros: { familias: ['avaliacao', 'tautologia', 'satisfativel'] },
      distribuicao: { por_split: { treino: 3000, teste: 300, dificil: 200 } },
    },
    familias: ['avaliacao', 'tautologia', 'satisfativel'],
    splits: { treino: 3000, teste: 300, dificil: 200 },
    tokens_aprox_trem: 274918,
    sha256: 'ebe6b2b2a66558f7',
    idioma: 'pt',
    tem_benchmark: true,
  })

  vi.mocked(api.itensDataset).mockResolvedValue({
    id: 'logica-pq',
    total: 3500,
    filtrados: 300,
    desde: 0,
    limite: 1,
    tem_benchmark: true,
    itens: [
      {
        id: 'avaliacao-teste-00001',
        enunciado: 'p = 1, q = 0. Calcule: (p E q).',
        cot: 'Passo 1: (p E q) = 0\nResposta: 0',
        resposta: 0,
        familia: 'avaliacao',
        split: 'teste',
      },
    ],
  })

  vi.mocked(api.presets).mockResolvedValue([
    {
      nome: 'equilibrado',
      descricao: '6 camadas, dim 256',
      modelo: { dim: 256, camadas: 6 },
      passos: 2500,
      lote: 32,
    },
  ])

  vi.mocked(api.modelos).mockResolvedValue({
    runs: [
      {
        id: 'logica-1',
        origem: 'run',
        parametros: 5800000,
        dispositivo: 'cuda',
        medido: true,
        treinado_em_raciocinio: true,
        benchmarks: [{ estrategia: 'cot', split: 'teste', acuracia_global: 0.8, itens: 60 }],
      },
    ],
    gguf: [
      {
        arquivo: 'qwen2.5-1.5b-q4_k_m.gguf',
        origem: 'gguf',
        tamanho_mb: 986,
        quantizacao: 'Q4_K_M',
        reasoning_declarado: true,
        rotulo_honestidade: 'reasoning: declarado pelo publicador — não medido aqui',
        medido: false,
      },
    ],
    llama_cpp_instalado: true,
    comando_instalacao: 'pip install llama-cpp-python',
  })

  vi.mocked(api.corre).mockResolvedValue([
    { run_id: 'logica-1', tipo: 'treino', concluido: true, dispositivo: 'cuda' },
  ])

  vi.mocked(api.configs).mockResolvedValue(['logica-refino.yaml', 'livros-rapido.yaml'])
  vi.mocked(api.execucoes).mockResolvedValue([])
  vi.mocked(api.logExecucao).mockResolvedValue({ chave: 'teste', linhas: ['iniciando...'], existe: true, total: 1 })
  vi.mocked(api.progresso).mockResolvedValue({
    run_id: 'logica-1',
    passo: 2500,
    passos_totais: 2500,
    concluido: true,
    loss_trem: 0.18,
    loss_val: 0.18,
    lr: 0.0001,
    tokens_por_s: 280000,
    tempo_s: 150,
    tempo_restante_s: 0,
    sparkline: [{ passo: 100, loss_trem: 6.0, loss_val: 6.1 }, { passo: 2500, loss_trem: 0.18, loss_val: 0.18 }],
  })
  vi.mocked(api.metricas).mockResolvedValue([
    { passo: 0, loss_trem: 6.5, loss_val: 6.4 },
    { passo: 1500, loss_trem: 0.19, loss_val: 0.18 },
    { passo: 2500, loss_trem: 0.18, loss_val: 0.19 },
  ])
  vi.mocked(api.comparar).mockResolvedValue({
    runs: [
      {
        run_id: 'logica-1',
        familia: 'moderna',
        passos: 2500,
        parametros: 5800000,
        melhor_val: 0.18,
        passo_melhor_val: 1500,
        val_final: 0.19,
        drift: 0.01,
        melhor_val_orcamento_comum: 0.18,
        tokens_por_s: 280000,
        diagnostico: 'estavel',
        comentario: 'treino converge bem',
      },
    ],
    orcamento_comum: 2500,
    familias: ['moderna'],
    ranking: ['logica-1'],
    melhor: 'logica-1',
    veredito: 'estável',
    limiares: {},
  })
  vi.mocked(api.benchmarksDoRun).mockResolvedValue([
    {
      run: 'logica-1',
      estrategia: 'cot',
      split: 'teste',
      semente: 1234,
      itens: 60,
      acuracia_global: 0.8,
      acuracia_por_familia: { avaliacao: 0.9, tautologia: 0.7 },
      taxa_resposta_valida: 1.0,
      amostra_erros: [],
    },
  ])
})

describe('Página Laboratório (B13)', () => {
  it('renderiza o título e as 6 sub-abas encadeadas (RF1.1)', async () => {
    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    expect(screen.getByText(/Laboratório de Modelos/i)).toBeInTheDocument()
    expect(screen.getByText('1. Dados')).toBeInTheDocument()
    expect(screen.getByText('2. Configurar')).toBeInTheDocument()
    expect(screen.getByText('3. Executar')).toBeInTheDocument()
    expect(screen.getByText('4. Curvas')).toBeInTheDocument()
    expect(screen.getByText('5. Testar')).toBeInTheDocument()
    expect(screen.getByText('6. Medir')).toBeInTheDocument()
  })

  it('navega para SubAbaDados e mostra itens e manifesto (CA1)', async () => {
    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    await waitFor(() => {
      expect(screen.getByText(/Escolher o dataset/i)).toBeInTheDocument()
      expect(screen.getByText(/Navegador de itens/i)).toBeInTheDocument()
    })

    // Confere se o item 1 do dataset aparece
    await waitFor(() => {
      expect(screen.getByText(/p = 1, q = 0. Calcule: \(p E q\)./i)).toBeInTheDocument()
    })
  })

  it('valida configuração imediatamente com erro claro (RF2.6, CA4)', async () => {
    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    // Clica na sub-aba Configurar
    fireEvent.click(screen.getByText('2. Configurar'))

    await waitFor(() => {
      expect(screen.getByLabelText(/Dimensão \(largura\):/i)).toBeInTheDocument()
    })

    // Altera dim para 100 com 8 cabeças (100 não é divisível por 8)
    const inputDim = screen.getByLabelText(/Dimensão \(largura\):/i)
    fireEvent.change(inputDim, { target: { value: '100' } })

    // Verifica que a mensagem de erro da CA4 aparece imediatamente.
    // O texto é renderizado dentro de um div com fragmentos: "✕ ", "<strong>Configuração Inválida:</strong>", " dim (100)..."
    // Usamos um matcher de função para busca parcial no conteúdo textual total do elemento.
    await waitFor(() => {
      const erroEl = screen.getByText(
        (content, element) =>
          !!element &&
          element.classList.contains('aviso--erro') &&
          content.includes('dim') &&
          element.textContent?.includes('dim (100) precisa ser divisível por cabecas (8)') === true,
      )
      expect(erroEl).toBeInTheDocument()
    })
  })

  it('mostra modelo GGUF com rótulo declarado de honestidade (RF2.2, CA13)', async () => {
    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    fireEvent.click(screen.getByText('2. Configurar'))

    await waitFor(() => {
      expect(screen.getByText(/Modelos GGUF/i)).toBeInTheDocument()
    })

    // Clica na aba de Modelos GGUF
    fireEvent.click(screen.getByText(/Modelos GGUF/i))

    await waitFor(() => {
      expect(screen.getByText('qwen2.5-1.5b-q4_k_m.gguf')).toBeInTheDocument()
      expect(screen.getByText(/reasoning: declarado pelo publicador — não medido aqui/i)).toBeInTheDocument()
    })
  })

  it('exibe o rótulo de exploração obrigatório na sub-aba Testar (RF7.3, CA15)', async () => {
    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    fireEvent.click(screen.getByText('5. Testar'))

    await waitFor(() => {
      expect(screen.getByLabelText(/Enunciado da pergunta:/i)).toBeInTheDocument()
    })

    // Simula resposta gerada
    vi.mocked(api.testar).mockResolvedValueOnce({
      run: 'logica-1',
      enunciado: 'p = 1, q = 0.',
      estrategia: 'cot',
      texto_gerado: 'Passo 1: p = 1\nResposta: 1',
      resposta_extraida: 1,
      exploracao: false,
      tempo_s: 0.5,
    })

    fireEvent.click(screen.getByText('▶ Gerar Resposta'))

    await waitFor(() => {
      expect(screen.getByText(/Painel de Exploração \(Teacher Forcing\)/i)).toBeInTheDocument()
      expect(
        screen.getByText(/aqui você está forçando o caminho \(teacher forcing\): o resultado não é a acurácia do modelo/i),
      ).toBeInTheDocument()
    })
  })

  it('mostra tela de núcleo indisponível quando a API falha (RNF7, CA9)', async () => {
    vi.mocked(api.saude).mockRejectedValueOnce(new Error('Connection refused'))

    render(
      <ProvedorDeAvisos>
        <Laboratorio />
      </ProvedorDeAvisos>,
    )

    await waitFor(() => {
      expect(screen.getByText(/Núcleo do Lab-IA Indisponível/i)).toBeInTheDocument()
      expect(screen.getByText(/\.venv\\Scripts\\python -m labia\.cli servir/i)).toBeInTheDocument()
    })
  })
})
