import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ProvedorDeAvisos } from '../src/ds/avisos'
import { Trilha } from '../src/paginas/Trilha'
import { api } from '../src/api'

vi.mock('../src/api', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/api')>()
  return {
    ...real,
    api: { trilha: vi.fn(), licao: vi.fn(), rodarLicao: vi.fn(), logExecucao: vi.fn() },
  }
})

const INDICE = {
  total: 2,
  licoes: [
    {
      numero: 1,
      titulo: 'Lição 1 — o tensor, o grafo e o gradiente',
      arquivo: 'trilha/licoes/01-tensor-e-gradiente.md',
      experimento: 'trilha/experimentos/e01_autograd.py',
      resumo: 'Por que existe .backward() e por que esquecer zero_grad',
      tem_experimento: true,
    },
    {
      numero: 8,
      titulo: 'Lição 8 — MoE: mais parâmetros sem mais conta',
      arquivo: 'trilha/licoes/08-moe.md',
      experimento: 'trilha/experimentos/e08_moe.py',
      resumo: 'Parâmetros ativos, colapso do roteador e a perda auxiliar',
      tem_experimento: true,
    },
  ],
  citacoes_quebradas: [] as { licao: number; arquivo: string; motivo: string }[],
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.trilha).mockResolvedValue(INDICE as never)
  vi.mocked(api.licao).mockResolvedValue('# Lição 1\n\n**No núcleo:** core/labia/models/gpt.py' as never)
  vi.mocked(api.logExecucao).mockResolvedValue({ chave: 'trilha-01', linhas: [], existe: true } as never)
})

const montar = () => render(<ProvedorDeAvisos><Trilha /></ProvedorDeAvisos>)

describe('Trilha', () => {
  it('lista as lições e mostra o material em dia quando não há citação quebrada', async () => {
    montar()
    expect(await screen.findByText(/1\. o tensor, o grafo e o gradiente/)).toBeInTheDocument()
    expect(screen.getByText(/8\. MoE/)).toBeInTheDocument()
    expect(screen.getByText('material em dia com o código')).toBeInTheDocument()
    expect(screen.getByText('e01_autograd.py')).toBeInTheDocument()
  })

  it('avisa quando alguma citação ao código morreu', async () => {
    vi.mocked(api.trilha).mockResolvedValue({
      ...INDICE,
      citacoes_quebradas: [{ licao: 3, arquivo: 'core/labia/models/gpt.py', motivo: 'símbolo não existe mais' }],
    } as never)
    montar()
    expect(await screen.findByText(/1 citação\(ões\) quebrada/)).toBeInTheDocument()
  })

  it('abre a lição e mostra o texto', async () => {
    const usuario = userEvent.setup()
    montar()
    await usuario.click(await screen.findByLabelText('abrir lição 1'))
    await waitFor(() => expect(api.licao).toHaveBeenCalledWith(1))
    expect(await screen.findByLabelText('texto da lição')).toHaveTextContent(/No núcleo/)
  })

  it('roda o experimento da lição e acompanha o log', async () => {
    vi.mocked(api.rodarLicao).mockResolvedValue({ chave: 'trilha-01', pid: 55, licao: 1, experimento: 'e01_autograd.py' } as never)
    vi.mocked(api.logExecucao).mockResolvedValue({
      chave: 'trilha-01',
      linhas: ['1) autograd x derivada analítica', '   erro máximo = 0.00e+00'],
      existe: true,
    } as never)
    const usuario = userEvent.setup()
    montar()
    await usuario.click(await screen.findByLabelText('abrir lição 1'))
    await usuario.click(await screen.findByRole('button', { name: /rodar experimento/i }))
    await waitFor(() => expect(api.rodarLicao).toHaveBeenCalledWith(1))
    expect(await screen.findByText(/erro máximo = 0.00e\+00/)).toBeInTheDocument()
    expect(await screen.findByRole('status')).toHaveTextContent(/rodando e01_autograd\.py/)
  })

  it('mostra o erro do núcleo como aviso', async () => {
    vi.mocked(api.trilha).mockRejectedValue(new Error('núcleo indisponível'))
    montar()
    expect(await screen.findByRole('status')).toHaveTextContent(/núcleo indisponível/)
  })
})
