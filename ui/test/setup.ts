import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// O setup roda também nos testes de ambiente node (ex.: test/nucleo.test.ts), onde
// não existe window nem localStorage — por isso as guardas de ambiente.
const temDom = typeof window !== 'undefined'

if (temDom && !window.matchMedia) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })) as unknown as typeof window.matchMedia
}

if (temDom) {
  beforeEach(() => {
    localStorage.clear()
  })
}

afterEach(() => {
  vi.useRealTimers()
})
