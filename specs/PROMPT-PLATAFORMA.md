# Prompt — Plataforma Lab-IA (loop Gauntlet completo)

> Prompt de trabalho da fase seguinte. Substitui o original (perdido) e foi escrito a
> partir do **estado real do repositório**: todo número aqui foi medido nesta máquina.

> **Atualização de direção — 15/09/2026:** o usuário confirmou que quer fazer
> seus próprios experimentos em um ambiente semelhante a Jupyter, com abertura a
> soluções open source. [B14](B14.md) e a direção vigente em [PLANO](PLANO.md)
> prevalecem sobre prioridades, critérios de encerramento e pendências históricas
> deste documento. Os épicos A–E são backlog; sua conclusão não substitui B14.
> O retrato do ambiente abaixo é histórico e precisa ser revalidado em L0.

## 0. Visão de produto vigente

Uma plataforma local para o usuário criar seus próprios experimentos de IA em
cadernos editáveis, com células Python/Markdown, sessão compartilhada entre células,
resultados, gráficos, interrupção, reinício e persistência de artefatos.

O núcleo próprio é uma biblioteca à disposição do usuário. Treino do zero,
LoRA/QLoRA, quantização e avaliação são capacidades que ele pode combinar e alterar.
As lições e benchmarks existentes são exemplos editáveis. Testes automatizados
verificam o software; não constituem o laboratório nem definem seus experimentos.

Avaliar uma base open source, começando por Jupyter, antes de criar infraestrutura
própria de notebook. A escolha técnica será registrada em L0. A interface Electron
existente pode apoiar essa experiência, sem ser uma restrição obrigatória.

## 1. Identidade do projeto (contexto do Hub)

| campo | valor |
|---|---|
| Projeto | lab-ia |
| languageOfRecord | **pt-BR** |
| Workspace | K:\Dev\lab-ia |
| Harness | dsh (DeepSeek Harness) · agente deepseek-v4-flash |
| Estado do Gauntlet | **CLI e MCP ausentes nesta máquina** → execução em **modo direto**, lacuna já registrada em specs/PLANO.md |

**Sobre o Gauntlet:** a gramática abaixo (ciclos, gates, reviews A/B, evidência) é a do
protocolo v0.7.0. Sem o Hub instalado, ela é seguida como disciplina e registrada em
`specs/PLANO.md`; quando o Hub existir, os mesmos itens são submetidos sem reescrita.

## 2. Regras invioláveis do loop

1. **Ferramenta prova, LLM interpreta.** Item só avança de ciclo com evidência
   re-executável (comando + saída) registrada.
2. **Nenhuma mudança sem teste que a cubra.** Se não dá para testar, o Dev devolve à
   ideação em vez de implementar e torcer.
3. **Reviewer A + Reviewer B independentes**, sobre a mesma revisão material: A com lente
   de correção/engenharia, B com lente adversarial/produto. Um não vê a conclusão do outro.
4. **Alteração material após revisão torna a revisão STALE.**
5. **Complete só com os dois reviews frescos, PASS e sem blockers.**
6. **Idioma:** pt-BR em tudo que o projeto controla; nome de API/biblioteca fica como é.
7. **Números, não adjetivos:** todo critério de aceite diz o valor esperado.

## 3. Baseline verificado (ponto de partida)

### 3.1 O que já existe e está verde

| área | estado | evidência |
|---|---|---|
| Núcleo do zero (G1–G10) | FEITO | 71 testes originais; runs reais na GPU |
| Bancada B1–B5 | FEITO | dados próprios, config assistida, comparar, varredura, curvas |
| Trilha B6 | FEITO | 10 lições, 10 experimentos, 10 notebooks |
| API + janela | FEITO | abas Painel, Bancada, Executar, Comparar, Trilha, Eventos, DS |
| Portátil | parcial | o app sobe o núcleo sozinho; pacote montado; **build não executado** |
| Suíte | verde | **236 testes pytest** (1 pulado: matplotlib) e **54 testes de UI** |
| GPU (RTX 3070) | FEITO | treino `livros-rapido` (14,48 s de laço, val 8,33 → 5,07); varredura de 6 variantes x 500 passos |

### 3.2 Correções já registradas na trilha de auditoria

- **MoE:** a perda auxiliar não tinha gradiente (estava dentro de `torch.no_grad`) — o
  `coef_auxiliar` não fazia nada. Corrigido; evidência: colapso de 94,5% vai a 26,6% com
  a auxiliar ativa, contra 81,6% sem ela.
- **Retomada:** morrer entre gravar o último checkpoint e fechar o estado deixava o run
  "não concluído" para sempre. Corrigido, com teste determinístico.
- **Estimador de tempo:** modelo só de FLOPs errava 2x; virou modelo de dois termos
  (erro médio 4,8% em validação cruzada).

### 3.3 Trabalho em andamento (não commitado, já verificado)

- `core/labia/models/componentes.py` **novo**: RMSNorm, RoPE, SwiGLU e GQA escritos.
- `core/labia/models/gpt.py` **parcialmente alterado**: ConfigGPT ganhou `norm`, `pos`,
  `mlp`, `n_cabecas_kv`, `rope_base`; atenção e bloco já escolhem por config.
- **Retrocompatibilidade confirmada:** 57 testes (g1 + g4 + presets) verdes com os padrões
  antigos (layernorm / posição aprendida / gelu / MHA).
- **Falta para F1 fechar:** RoPE ligado no `GPT.emb` e no `forward`, buffers de frequência,
  `init_pesos` para RMSNorm/SwiGLU, contagem de parâmetros, validação de config, testes,
  benchmark na GPU, lições e seletor na UI.

## 4. Pré-condições, limites e direito de recusa

O Dev/Executor **deve devolver à ideação** ao encontrar: inviabilidade, dependência
ausente, contradição, escopo excessivo ou falta de testabilidade.

Limites duros deste ambiente (medidos, não supostos):

| limite | consequência para o plano |
|---|---|
| **Sem rede** (PyPI falha por SSL) | nada que dependa de instalar pacote pode ser critério de aceite |
| Sem jupyter | notebooks verificados por estrutura; a execução é evidência humana |
| Sem llama.cpp | o Épico B começa por um WI de instalação |
| VRAM 8 GB | nenhum WI pode exigir pico acima de ~6 GB |
| Disco ~89 GB livres | pacote CUDA ocupa 4,84 GB; sabor CPU ~500 MB |
| Gauntlet Hub ausente | modo direto, lacuna registrada |

## 5. Épicos e Work Items

### Épico A — B7 · Biblioteca de arquiteturas (GPT-1 ao estado da arte)
*Escolhido por você como primeiro.*

| WI | objetivo | critérios de aceite (com número) |
|---|---|---|
| **A.1** Componentes modernos | RMSNorm, RoPE, SwiGLU, GQA em componentes.py | CA1 RoPE preserva a norma (erro < 1e-6) e q·k depende só da distância entre posições; CA2 SwiGLU com 8/3·dim fica dentro de ±5% dos parâmetros da FFN densa; CA3 GQA com 2 cabeças de K/V corta os parâmetros de K/V em 50%; CA4 suíte de componentes verde |
| **A.2** Integração e retrocompatibilidade | ligar as opções ponta a ponta no GPT | CA1 checkpoint antigo carrega e a saída é idêntica à de antes (< 1e-6); CA2 as 4 combinações norm x mlp treinam 20 passos sem erro; CA3 RoPE sem embedding de posição reduz parâmetros em janela x dim |
| **A.3** Contagem e validação | contar_parametros e validação cientes da arquitetura | CA1 a conta bate com GPT.contar_parametros() em 8 combinações parametrizadas; CA2 config inválida (dim ímpar para RoPE, cabeças não múltiplas de kv) falha com mensagem que diz o conserto |
| **A.4** Benchmark real na 3070 | medir o que a modernização custa e rende | CA1 tabela com parâmetros, pico de VRAM e tokens/s por variante; CA2 veredito escrito sobre o que muda de GPT-2 para moderno na mesma VRAM; CA3 perda em orçamento igual, com 3 sementes |
| **A.5** Lições 11 a 13 | RMSNorm, RoPE, SwiGLU+GQA como aula | CA1 cada lição com experimento, teste e notebook; CA2 trilha --conferir com 0 citações quebradas; CA3 gera_notebooks --checar em dia |
| **A.6** UI | escolher arquitetura pela janela | CA1 lab-ia novo --arquitetura moderna gera config válida; CA2 a página Bancada expõe o seletor; CA3 a API valida as opções |

### Épico B — B8 · Modelos pré-treinados + llama.cpp/GGUF

| WI | objetivo | critérios de aceite |
|---|---|---|
| **B.1** Registro e download | catálogo de 0,5 a 1,5 B com hash e cache local | CA1 download confere sha256; CA2 modelo ausente falha dizendo o comando; CA3 nada baixa sozinho sem confirmação |
| **B.2** LoRA/QLoRA reais | ajustar modelo pré-treinado no corpus do usuário | CA1 perda na tarefa cai 80% ou mais em até 300 passos na 3070; CA2 adaptador ocupa menos de 5% dos parâmetros; CA3 comparação com o nano do zero na mesma tarefa |
| **B.3** GGUF via llama.cpp | converter, quantizar e comparar | CA1 Q4_K_M, Q5_K_M e Q8_0 gerados; CA2 tabela tamanho x perda x tokens/s contra int8/NF4 nativos; CA3 veredito sobre quando cada formato compensa |
| **B.4** Serving local | servidor com medição e integração | CA1 latência e throughput medidos na 3070; CA2 chat local na janela; CA3 erro claro quando o modelo não cabe na VRAM |
| **B.5** Lições 14 a 16 | do pré-treinado ao GGUF | mesmo padrão de A.5 |

### Épico C — B9 · Currículo completo (graduação em IA)

| WI | objetivo |
|---|---|
| **C.1** Mapa curricular | grafo de pré-requisitos, ordem, carga estimada e o que cada módulo habilita |
| **C.2** Matemática | álgebra linear, cálculo, probabilidade e otimização **medidos no código** |
| **C.3** ML clássico | regressão, árvores, SVM, clustering, validação cruzada, viés e variância |
| **C.4** Deep learning e LLM | consolidar as 10 lições existentes dentro do mapa |
| **C.5** Método científico | desenho de experimento, erro, revisão e escrita de resultado |
| **C.6** UI | trilha com progresso, pré-requisitos e marcação de concluído |

Critério transversal: **todo módulo tem exercício com resultado esperado medido por teste.**

### Épico D — B10 · Rigor de pesquisa

| WI | objetivo | critério de aceite |
|---|---|---|
| **D.1** Múltiplas sementes | varredura com média e intervalo | CA1 saída traz média ± intervalo; CA2 a bancada marca "dentro do ruído" quando for o caso |
| **D.2** Caderno de hipóteses | hipótese, experimento, resultado, conclusão | CA1 arquivo versionado; CA2 vinculado ao run que testou |
| **D.3** Relatório reproduzível | um comando reproduz o achado | CA1 relatório com comando, semente, hash do dado e resultado |
| **D.4** Reauditoria | revisar achados antigos com incerteza | CA1 os achados do PLANO (+5,3 p.p. do CoT; MoE igual ao denso) ganham barra de erro ou marcação de "1 semente" |

### Épico E — B11 · Design System completo + pacote portátil

| WI | objetivo | critério de aceite |
|---|---|---|
| **E.1** Tokens em camadas | primitivos, semânticos, componentes | CA1 nenhum componente usa cor ou medida crua; CA2 catálogo vivo atualizado |
| **E.2** Componentes faltantes | tabela, abas, modal, formulário, estados | CA1 cada um com teste de render e de estado |
| **E.3** Acessibilidade | contraste, foco, teclado, leitor de tela | CA1 contraste AA verificado; CA2 navegação completa por teclado; CA3 teste automatizado de foco |
| **E.4** Sabor do pacote | **decisão humana bloqueante** | CA1 sabor escolhido; CA2 build gerado; CA3 verificação em pasta limpa |
| **E.5** Build e verificação | instalador em máquina sem Python | CA1 verifica_pacote verde no bundle; CA2 o app sobe o núcleo e treina um micro modelo **sem rede e sem Python instalado** |

## 6. O ciclo (igual para todo Work Item)

    IDEAÇÃO --> INTAKE --> SPEC --> BUILD --> CHECKPOINT --> REVIEW A + B --> GATE --> COMPLETE
       ^          |                    |           |              |
       +-- recusa-+                    |           |              +-- RETURN (máx. 2)
                                       +- evidência+

1. **Ideação** — problema, hipótese e alternativas descartadas com o motivo.
2. **Intake** — o Dev aceita ou **devolve** (inviabilidade, dependência ausente,
   contradição, escopo excessivo, falta de testabilidade).
3. **Spec** — `specs/<ID>.md` com requisitos e critérios de aceite em Given-When-Then,
   cada um com o número esperado.
4. **Build** — implementação mais testes. Sem teste, não entra.
5. **Checkpoint** — evidência (comando e saída) registrada em `specs/PLANO.md`.
6. **Review A** (correção/engenharia) e **Review B** (adversarial/produto), independentes,
   sobre a mesma revisão material. Alteração depois disso torna a revisão STALE.
7. **Gate** — PASS exige CA verdes, evidências registradas, zero blockers e reviews frescos.
8. **Advance** ou **Return** (volta a ciclo anterior, respeitando o orçamento de 2 retornos).
9. **Complete** — só com os dois reviews frescos e PASS.

## 7. Fóruns previstos (deliberação estruturada)

| ID | pauta | quando | por quê |
|---|---|---|---|
| **F-1** | Do zero ou pré-treinado como eixo do currículo? | antes de abrir o Épico B | risco de virar dois caminhos paralelos e nenhum profundo |
| **F-2** | Qual sabor do pacote portátil? | antes de E.4 | 4,84 GB CUDA x ~500 MB CPU x arquivo único de auto-extração |
| **F-3** | Manter o DS próprio ou adotar base pronta? | antes do Épico E | decide o custo de E.1 a E.3 |

Fórum é deliberação com pauta, rodadas e fechamento — não chat livre. O Mestre de
Cerimônias organiza; não substitui especialista nem ignora blocker de domínio.

## 8. Human Inbox (perguntas bloqueantes)

| ID | pergunta | bloqueia |
|---|---|---|
| **H-1** | Qual sabor do pacote portátil (CUDA, CPU ou arquivo único)? | E.4 e E.5 |
| **H-2** | Quais pesos pré-treinados você aceita baixar (licença e origem)? | B.1 |
| **H-3** | Quanto tempo por semana você dedica ao currículo? | dimensionamento do Épico C |

## 9. Capability gaps (registro operacional, não pergunta humana)

| ID | gap | mitigação enquanto durar |
|---|---|---|
| GAP-1 | **Sem rede** (PyPI falha por SSL) | o que exige instalação vira WI de instalação com evidência do humano; nada em CPU fica bloqueado |
| GAP-2 | **Gauntlet Hub ausente** | modo direto com a mesma disciplina; lacuna já registrada |
| GAP-3 | **Sem LLM local (llama.cpp)** | resolvido pelo Épico B; até lá o trabalho cognitivo fica no agente principal |
| GAP-4 | **Sem jupyter** | notebooks verificados por estrutura e correspondência com o experimento |

## 10. Router e workers

- Trabalho cognitivo delegável (esboço de lição, classificação, revisão, documentação)
  deveria passar pelo **Router** e consumir o worker local antes de escalonar. Sem
  llama.cpp isso é GAP-3 — registrado, não contornado por fora.
- O worker local é **cognitivo e read-only**: não edita arquivo, não roda shell, não
  altera o Hub e não conclui item.

## 11. Artifact graph (por épico)

| épico | artefatos |
|---|---|
| A | core/labia/models/componentes.py · gpt.py · specs/B7.md · tests/b7/ · trilha/licoes/1[123]-*.md · trilha/experimentos/e1[123]_*.py · runs/_arquiteturas/ |
| B | core/labia/pretreinado/ · core/labia/gguf/ · cache local de modelos · specs/B8.md · lições 14 a 16 |
| C | trilha/CURRICULO.md · módulos em trilha/licoes/ · specs/B9.md |
| D | core/labia/bancada/estatistica.py · runs/_caderno/ · specs/B10.md |
| E | ui/src/ds/tokens/ · componentes novos · specs/B11.md · release/ |

## 12. Orçamento, escalonamento e parada

- **Orçamento:** 2 retornos por Work Item antes de escalar; 60 rodadas de execução.
- **Escalonar quando:** o mesmo FAIL se repetir em dois ciclos com material distinto;
  dependência externa bloquear (rede, Hub, decisão humana); custo estimado passar do
  orçamento do épico.
- **Parar quando:** os cinco épicos fecharem com reviews PASS, ou quando um blocker
  humano (H-1 a H-3) impedir o próximo WI.

## 13. Definition of Done vigente

O aceite mínimo é o da [B14](B14.md): criar um caderno vazio, escrever código livre,
carregar dados próprios, usar e modificar modelos, executar treino e métricas de
sua escolha, inspecionar resultados, interromper/reiniciar, salvar e reabrir o
trabalho. Restaurar modelo por checkpoint é explícito; salvar notebook não preserva
memória do kernel. Deve haver uma sessão de aceite com experimento escolhido pelo
usuário, além da demonstração técnica e das verificações de engenharia.

Currículo completo, modelos externos, benchmarks extensos e o acabamento do Design
System continuam desejados, mas não bloqueiam o primeiro laboratório utilizável.

## 14. Ordem de execução vigente

L0 → L1 → L2 → L3 → L4 → L5, conforme B14.

**Primeiro passo concreto:** prova de conceito local da base open source: caderno
com duas células compartilhando estado, importação de `labia`, saída visível e
salvamento/reabertura. Registrar a decisão técnica e as limitações verificadas.
Depois entregar a integração do núcleo e o ciclo autoral completo antes de retomar
a expansão dos épicos A–E. As decisões de pesos e pacote final não bloqueiam L0.
