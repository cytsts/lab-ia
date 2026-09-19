import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { Cadernos } from '../src/paginas/Cadernos'

describe('Cadernos', () => {
  beforeEach(() => {
    window.labia = undefined
  })

  it('explica o espaço autoral e oferece o comando de abertura', () => {
    render(<Cadernos />)
    expect(screen.getByRole('heading', { name: 'Cadernos' })).toBeInTheDocument()
    expect(screen.getByText(/00-primeiro-experimento\.ipynb/)).toBeInTheDocument()
    expect(screen.getByText(/labia\.cli laboratorio/)).toBeInTheDocument()
  })

  it('abre o JupyterLab pela ponte do Electron', async () => {
    const abrirCadernos = vi.fn().mockResolvedValue({ ok: true, url: 'http://127.0.0.1:8889' })
    window.labia = { abrirCadernos }
    render(<Cadernos />)
    fireEvent.click(screen.getByRole('button', { name: 'Abrir cadernos' }))
    await waitFor(() => expect(abrirCadernos).toHaveBeenCalledOnce())
  })

  it('mostra um erro claro fora do Electron', () => {
    render(<Cadernos />)
    fireEvent.click(screen.getByRole('button', { name: 'Abrir cadernos' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/Electron ou use o comando/)
  })
})
