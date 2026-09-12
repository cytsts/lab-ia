// @vitest-environment node
import { describe, expect, it, vi } from 'vitest'
import { join } from 'node:path'
import {
  escolherRaiz,
  escolherComandoNucleo,
  semearWorkspace,
  esperarSaude,
  garantirNucleo,
  consultarSaude,
} from '../electron/nucleo.mjs'

describe('escolherRaiz', () => {
  it('respeita LABIA_RAIZ acima de tudo', () => {
    expect(escolherRaiz({ ambiente: { LABIA_RAIZ: 'D:\\lab' }, empacotado: true, pastaUsuario: 'C:\\u' })).toBe('D:\\lab')
  })

  it('empacotado usa a pasta do usuário (gravável), não a do app', () => {
    expect(escolherRaiz({ empacotado: true, pastaUsuario: join('C:', 'Users', 'x', 'Lab-IA'), pastaApp: 'C:\\tmp\\app' })).toBe(
      join('C:', 'Users', 'x', 'Lab-IA', 'lab'),
    )
  })

  it('em desenvolvimento usa a raiz do repositório', () => {
    const raiz = escolherRaiz({ empacotado: false, pastaApp: join('K:', 'Dev', 'lab-ia', 'ui', 'electron') })
    expect(raiz).toBe(join('K:', 'Dev', 'lab-ia', 'ui', 'electron', '..', '..'))
  })
})

function comandoOuFalha(opcoes: Parameters<typeof escolherComandoNucleo>[0]) {
  const comando = escolherComandoNucleo(opcoes)
  if (!comando) throw new Error('esperava um comando de núcleo, veio null')
  return comando
}

describe('escolherComandoNucleo', () => {
  it('prefere o exe congelado do pacote quando empacotado', () => {
    const comando = comandoOuFalha({
      empacotado: true, recursos: join('C:', 'app', 'resources'), existe: () => true,
    })
    expect(comando.exe).toContain('nucleo')
    expect(comando.args).toEqual([])
  })

  it('devolve null quando o pacote não traz o núcleo', () => {
    const comando = escolherComandoNucleo({ empacotado: true, recursos: 'C:\\app', existe: () => false })
    expect(comando).toBeNull()
  })

  it('usa o python do venv em desenvolvimento', () => {
    const raiz = join('K:', 'Dev', 'lab-ia')
    const venv = process.platform === 'win32' ? join(raiz, '.venv', 'Scripts', 'python.exe') : join(raiz, '.venv', 'bin', 'python')
    const comando = comandoOuFalha({ empacotado: false, raiz, existe: (p: string) => p === venv })
    expect(comando.exe).toBe(venv)
    expect(comando.args).toEqual(['-m', 'labia.cli'])
  })

  it('cai no python do PATH quando não há venv', () => {
    const comando = comandoOuFalha({ empacotado: false, raiz: 'K:\\lab-ia', existe: () => false })
    expect(comando.origem).toBe('python do PATH')
    expect(comando.args).toEqual(['-m', 'labia.cli'])
  })

  it('LABIA_NUCLEO ganha de tudo', () => {
    const comando = comandoOuFalha({ ambiente: { LABIA_NUCLEO: 'D:\\x\\lab-ia.exe' }, empacotado: true, existe: () => true })
    expect(comando.exe).toBe('D:\\x\\lab-ia.exe')
  })
})

describe('semearWorkspace', () => {
  const criarFake = () => {
    const copiados: [string, string][] = []
    const criados: string[] = []
    const semeadas = semearWorkspace({
      raiz: 'C:\\lab',
      semente: 'C:\\app\\semente',
      existe: (p: string) => p === 'C:\\app\\semente' || p.startsWith('C:\\app\\semente\\'),
      copiar: (o: string, d: string) => {
        copiados.push([o, d])
      },
      criar: (p: string) => {
        criados.push(p)
      },
    })
    return { semeadas, copiados, criados }
  }

  it('copia configs, docs e data quando o workspace está vazio', () => {
    const { semeadas, copiados, criados } = criarFake()
    expect(semeadas).toEqual(['configs', 'docs', 'data'])
    expect(copiados.map(([o]) => o)).toEqual([
      join('C:\\app\\semente', 'configs'),
      join('C:\\app\\semente', 'docs'),
      join('C:\\app\\semente', 'data'),
    ])
    expect(criados).toEqual(['C:\\lab'])
  })

  it('não sobrescreve o que o usuário já tem', () => {
    const copiados: [string, string][] = []
    const semeadas = semearWorkspace({
      raiz: 'C:\\lab',
      semente: 'C:\\app\\semente',
      existe: (p: string) => p === 'C:\\lab\\configs' || p === 'C:\\app\\semente' || p.startsWith('C:\\app\\semente\\'),
      copiar: (o: string, d: string) => {
        copiados.push([o, d])
      },
      criar: () => {},
    })
    expect(semeadas).toEqual(['docs', 'data'])
    expect(copiados.map(([, d]) => d)).not.toContain('C:\\lab\\configs')
  })

  it('sem semente (dev) não faz nada', () => {
    expect(semearWorkspace({ raiz: 'C:\\lab', semente: '' })).toEqual([])
  })
})

describe('esperarSaude', () => {
  it('volta assim que o núcleo responde', async () => {
    let tentativas = 0
    const resultado = await esperarSaude({
      buscar: async () => {
        tentativas += 1
        return { ok: true, json: async () => ({ ok: true }) }
      },
      dormir: async () => {},
      agora: () => 0,
      limiteMs: 1000,
    })
    expect(resultado.ok).toBe(true)
    expect(tentativas).toBe(1)
  })

  it('desiste quando o processo morre', async () => {
    const resultado = await esperarSaude({
      buscar: async () => {
        throw new Error('recusou conexão')
      },
      vivo: () => false,
      dormir: async () => {},
      agora: () => 0,
      limiteMs: 1000,
    })
    expect(resultado).toEqual({ ok: false, erro: 'o processo do núcleo morreu antes de responder' })
  })

  it('desiste quando estoura o tempo', async () => {
    let relogio = 0
    const resultado = await esperarSaude({
      buscar: async () => {
        throw new Error('ECONNREFUSED')
      },
      dormir: async () => {
        relogio += 500
      },
      agora: () => relogio,
      limiteMs: 1000,
    })
    expect(resultado.ok).toBe(false)
    expect(resultado.erro).toMatch(/não respondeu em 1 s/)
  })
})

describe('garantirNucleo', () => {
  const processoFalso = { pid: 4242, exitCode: null, killed: false, stdout: null, stderr: null }

  it('reaproveita núcleo já saudável em vez de subir outro', async () => {
    const lancar = vi.fn()
    const resultado = await garantirNucleo({
      raiz: 'C:\\lab',
      comando: { exe: 'x', args: [], origem: 'teste' },
      buscar: async () => ({ ok: true, json: async () => ({ ok: true }) }),
      lancar,
    })
    expect(resultado.estado).toBe('reutilizado')
    expect(lancar).not.toHaveBeenCalled()
  })

  it('sobe o núcleo com a porta e a raiz certas e espera o /saude', async () => {
    const lancar = vi.fn(() => processoFalso)
    const criar = vi.fn()
    const resultado = await garantirNucleo({
      raiz: 'C:\\lab',
      porta: 9999,
      comando: { exe: 'lab-ia.exe', args: [], origem: 'nucleo empacotado' },
      buscar: async () => {
        throw new Error('ECONNREFUSED')
      },
      lancar,
      criar,
      abrirLog: () => ({ write: () => {} }),
      esperar: async () => ({ ok: true, saude: { ok: true } }),
    })
    expect(resultado.estado).toBe('iniciado')
    expect(lancar).toHaveBeenCalledWith('lab-ia.exe', ['servir', '--porta', '9999', '--raiz', 'C:\\lab'], {
      cwd: 'C:\\lab',
      windowsHide: true,
    })
    expect(criar).toHaveBeenCalledWith(join('C:\\lab', '.lab-ia', 'logs'), { recursive: true })
  })

  it('relata erro quando o núcleo não responde', async () => {
    const resultado = await garantirNucleo({
      raiz: 'C:\\lab',
      comando: { exe: 'lab-ia.exe', args: [], origem: 'x' },
      buscar: async () => {
        throw new Error('ECONNREFUSED')
      },
      lancar: () => processoFalso,
      criar: () => {},
      abrirLog: () => ({ write: () => {} }),
      esperar: async () => ({ ok: false, erro: 'o núcleo não respondeu em 180 s' }),
    })
    expect(resultado.estado).toBe('erro')
    expect(resultado.erro).toMatch(/180 s/)
  })

  it('sem comando, explica em vez de estourar', async () => {
    const resultado = await garantirNucleo({
      raiz: 'C:\\lab',
      comando: null,
      buscar: async () => {
        throw new Error('ECONNREFUSED')
      },
    })
    expect(resultado).toEqual({ estado: 'erro', erro: 'núcleo não encontrado no pacote nem no venv' })
  })
})

describe('consultarSaude', () => {
  it('devolve null quando o núcleo está fora do ar', async () => {
    expect(await consultarSaude(8765, async () => {
      throw new Error('ECONNREFUSED')
    })).toBeNull()
  })
})
