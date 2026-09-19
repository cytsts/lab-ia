"""Normaliza e atualiza os cadernos didáticos de Qwen/GGUF.

O arquivo existe para que alterações estruturais dos notebooks sejam revisáveis e
reproduzíveis, em vez de dependerem de edições manuais no JSON do Jupyter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "cadernos" / "lab_jupyter_llm_qwen_gguf"


def _id(texto: str) -> str:
    return hashlib.sha1(texto.encode()).hexdigest()[:8]


def codigo(texto: str) -> dict:
    return {
        "id": _id(texto),
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True),
    }


def markdown(texto: str) -> dict:
    return {
        "id": _id(texto),
        "cell_type": "markdown",
        "metadata": {},
        "source": texto.splitlines(keepends=True),
    }


def carregar(nome: str) -> dict:
    return json.loads((PASTA / nome).read_text(encoding="utf-8"))


def salvar(nome: str, notebook: dict) -> None:
    for celula in notebook["cells"]:
        if celula["cell_type"] == "code":
            celula["execution_count"] = None
            celula["outputs"] = []
    (PASTA / nome).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def trocar_codigo(notebook: dict, indice: int, texto: str) -> None:
    assert notebook["cells"][indice]["cell_type"] == "code"
    notebook["cells"][indice] = codigo(texto)


def atualizar_modelo_puro() -> None:
    nb = carregar("01_modelo_puro_do_zero.ipynb")
    trocar_codigo(nb, 4, """from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from pathlib import Path

TOKENIZER_DIR = Path('artifacts/tokenizer_puro')
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

tok = Tokenizer(models.BPE(unk_token='<unk>'))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer_bpe = trainers.BpeTrainer(
    vocab_size=2000,
    min_frequency=2,
    # Sem o alfabeto inicial, um corpus curto pode acabar com apenas os tokens especiais.
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=['<pad>', '<s>', '</s>', '<unk>'],
)
tok.train([str(DATA)], trainer=trainer_bpe)
# `tok.model.save` grava apenas vocab.json+merges.txt (o MODELO BPE). Para recarregar o
# tokenizer completo depois (`Tokenizer.from_file`), é preciso o `tok.save` com caminho
# de arquivo — em diretório ele falha no Windows.
tok.save(str(TOKENIZER_DIR / 'tokenizer.json'))
print('Tokenizer salvo em', TOKENIZER_DIR)
""")
    trocar_codigo(nb, 5, """from transformers import PreTrainedTokenizerFast

# No Transformers 5, GPT2TokenizerFast(vocab_file=..., merges_file=...) não
# reconstrói mais um tokenizer BPE local. Preservamos o objeto treinado em memória.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',
)
print('Vocab:', len(tokenizer))
print(tokenizer.tokenize('Treinar um Transformer do zero é instrutivo.'))
""")
    trocar_codigo(nb, 12, """from inspect import signature
from transformers import TrainingArguments, Trainer

OUT = 'artifacts/modelo_puro'
argumentos = dict(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=8 if torch.cuda.is_available() else 2,
    per_device_eval_batch_size=8 if torch.cuda.is_available() else 2,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    report_to='none',
    seed=SEED,
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
)
# `overwrite_output_dir` existia no Transformers 4, mas foi removido no 5.
if 'overwrite_output_dir' in signature(TrainingArguments).parameters:
    argumentos['overwrite_output_dir'] = True
args = TrainingArguments(**argumentos)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    data_collator=collator,
)
trainer.train()
""")
    trocar_codigo(nb, 15, """model.eval()
device = model.device
prompt = 'Um Transformer causal aprende'
inputs = tokenizer(prompt, return_tensors='pt').to(device)
comprimento = inputs['input_ids'].shape[-1]
if comprimento == 0:
    raise RuntimeError('O tokenizer produziu uma entrada vazia; execute novamente as células 3 e 4.')
if comprimento >= BLOCK_SIZE:
    raise RuntimeError(f'O prompt tem {comprimento} tokens, mas o contexto é {BLOCK_SIZE}.')

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=min(80, BLOCK_SIZE - comprimento),
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
""")
    salvar("01_modelo_puro_do_zero.ipynb", nb)


def atualizar_qlora() -> None:
    nb = carregar("02_qwen3_0_6b_raciocinio_qlora.ipynb")
    trocar_codigo(nb, 20, """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_config = ADAPTER_DIR / 'adapter_config.json'
if not adapter_config.is_file():
    raise FileNotFoundError(
        f'Adaptador ausente em {ADAPTER_DIR}. Execute as células 5 (SFT) e 6 (salvar) antes do merge.'
    )

# CPU é suficiente para 0.6B e evita pressão na VRAM.
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map='cpu')
merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload()

MERGED_DIR.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size='2GB')
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(MERGED_DIR)
print('Modelo merged salvo em:', MERGED_DIR)
""")
    # Evidência (execução 2026-09-17, transformers 5.17): `warmup_ratio` foi removido no
    # Transformers 5 e `paged_adamw_8bit` exige UVM, indisponível no Windows.
    trocar_codigo(nb, 13, """from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    # `warmup_ratio` foi removido no Transformers 5; restou apenas `warmup_steps`.
    # No smoke test (~24 passos de otimizador), 5 passos de aquecimento ~ 5% do treino.
    warmup_steps=5,
    lr_scheduler_type='cosine',
    logging_steps=2,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    report_to='none',
    seed=SEED,
    gradient_checkpointing=True,
    bf16=cuda and torch.cuda.is_bf16_supported(),
    fp16=cuda and not torch.cuda.is_bf16_supported(),
    # `paged_adamw_8bit` exige memória unificada (UVM), indisponível no Windows.
    # `adamw_bnb_8bit` economiza estado do otimizador sem paginação.
    optim='adamw_bnb_8bit' if cuda else 'adamw_torch',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds['train'],
    eval_dataset=tok_ds['test'],
    data_collator=collator,
)
trainer.train()
""")
    # Depois do treino, o gradient checkpointing continua ativo e desliga o cache da
    # geração sem avisar; o teste do adapter precisa desativá-lo antes de gerar.
    trocar_codigo(nb, 17, """model.eval()
# O treino deixou o gradient checkpointing ativo, o que força use_cache=False e
# deixa a geração lenta. Desligue antes de gerar.
if getattr(model, "gradient_checkpointing_enable", None):
    model.gradient_checkpointing_disable()
model.config.use_cache = True

messages = [{"role":"user", "content":"/think Uma máquina processa 45 itens por minuto. Quantos itens processa em 8 minutos?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
new_tokens = out[0, inputs['input_ids'].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
""")
    salvar("02_qwen3_0_6b_raciocinio_qlora.ipynb", nb)


def novos_cadernos() -> None:
    geracao = {
        "cells": [
            markdown("""# 05 — Gerar com o modelo treinado do zero\n\nUse este caderno depois do `01_modelo_puro_do_zero.ipynb` para testar prompts sem repetir o treinamento. O modelo didático é pequeno: avalie se ele reproduz padrões do corpus, não conhecimento geral.\n"""),
            codigo("""from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path('artifacts/modelo_puro')
if not (MODEL_DIR / 'config.json').is_file():
    raise FileNotFoundError('Modelo ausente. Execute primeiro o caderno 01 até a célula de salvamento.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
print(f'Modelo carregado em {device}; contexto: {model.config.n_positions} tokens')
"""),
            markdown("## Testar prompts\n\nMude `prompt` e compare amostragem com geração determinística (`do_sample=False`)."),
            codigo("""prompt = 'Modelos de linguagem'
inputs = tokenizer(prompt, return_tensors='pt').to(device)
limite = model.config.n_positions - inputs['input_ids'].shape[-1]
if limite <= 0:
    raise ValueError('O prompt ocupa todo o contexto; use um prompt menor.')

with torch.no_grad():
    saida = model.generate(
        **inputs,
        max_new_tokens=min(80, limite),
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(saida[0], skip_special_tokens=True))
"""),
            markdown("## Exercício\n\nCrie cinco prompts de continuação retirados do corpus, anote quais respostas são coerentes e altere apenas uma variável de geração por vez."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    avaliacao = {
        "cells": [
            markdown("""# 06 — Avaliação reproduzível de respostas\n\nEste caderno cria uma pequena bateria de casos e mede uma regra objetiva: a resposta contém o resultado esperado. Ele serve para comparar versões do mesmo modelo; para um projeto real, substitua os casos por um conjunto de teste que não participou do treino.\n"""),
            codigo("""from pathlib import Path
import json

ARQUIVO = Path('data/avaliacao_basica.jsonl')
casos = [
    {'prompt': 'Quanto é 6 vezes 8?', 'esperado': '48'},
    {'prompt': 'Quanto é 25% de 120?', 'esperado': '30'},
    {'prompt': 'Qual é o próximo número: 2, 4, 8, 16?', 'esperado': '32'},
]
ARQUIVO.parent.mkdir(exist_ok=True)
with ARQUIVO.open('w', encoding='utf-8') as arquivo:
    for caso in casos:
        arquivo.write(json.dumps(caso, ensure_ascii=False) + '\\n')
print(f'{len(casos)} casos salvos em {ARQUIVO}')
"""),
            markdown("## Carregar o modelo\n\nEste caderno é independente: ele carrega o artefato gerado no caderno 01. Para avaliar GGUF ou uma API, substitua apenas a função `gerar`."),
            codigo("""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path('artifacts/modelo_puro')
if not (MODEL_DIR / 'config.json').is_file():
    raise FileNotFoundError('Modelo ausente. Execute primeiro o caderno 01 até a célula de salvamento.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()

def gerar(prompt: str) -> str:
    entrada = tokenizer(prompt, return_tensors='pt').to(device)
    restante = model.config.n_positions - entrada['input_ids'].shape[-1]
    if restante <= 0:
        return ''
    with torch.no_grad():
        saida = model.generate(**entrada, max_new_tokens=min(48, restante), do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(saida[0, entrada['input_ids'].shape[-1]:], skip_special_tokens=True)
"""),
            codigo("""resultados = []
for caso in casos:
    resposta = gerar(caso['prompt'])
    acertou = caso['esperado'] in resposta
    resultados.append({**caso, 'resposta': resposta, 'acertou': acertou})

acertos = sum(item['acertou'] for item in resultados)
print(f'Acurácia: {acertos}/{len(resultados)} = {acertos / len(resultados):.1%}')
for item in resultados:
    print(('✓' if item['acertou'] else '✗'), item['prompt'], '=>', repr(item['resposta']))
"""),
            markdown("## Exercício\n\nAmplie a bateria, mantenha-a versionada e compare a acurácia antes e depois de uma alteração. Leia também as respostas: uma métrica simples não substitui inspeção qualitativa."),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    salvar("05_gerar_modelo_puro.ipynb", geracao)
    salvar("06_avaliar_respostas.ipynb", avaliacao)


def cadernos_de_estudo() -> None:
    """07–09: cadernos que ligam os experimentos Jupyter ao núcleo nativo do Lab-IA."""
    metadados = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }

    nativo = {"cells": [
        markdown("""# 07 — O GPT nativo do Lab-IA

Os cadernos 01–06 usam HuggingFace Transformers + PEFT. O núcleo do Lab-IA tem uma
implementação **própria** de GPT (`labia.models.gpt`) em que a arquitetura é objeto de
estudo: dá para trocar normalização (LayerNorm ↔ RMSNorm), posição (aprendido ↔ RoPE),
MLP (GELU ↔ SwiGLU), atenção (MHA ↔ GQA) e FFN (densa ↔ MoE) por **uma variável da config**.

Pré-requisito: caderno 01 executado (usa `data/corpus.txt` e `artifacts/tokenizer_puro`).
"""),
        codigo("""from pathlib import Path
import torch
from tokenizers import Tokenizer

from labia.models.gpt import ConfigGPT, GPT, gerar as gerar_com_prompt
from labia.trainer.dados import dividir_corpus, montar_dataset

ARQ_TOKENIZER = Path('artifacts/tokenizer_puro/tokenizer.json')
if not ARQ_TOKENIZER.is_file():
    raise FileNotFoundError('Tokenizer ausente. Execute primeiro o caderno 01 (ele salva artifacts/tokenizer_puro).')
tok = Tokenizer.from_file(str(ARQ_TOKENIZER))
corpus = Path('data/corpus.txt').read_text(encoding='utf-8')
print('vocab:', tok.get_vocab_size(), '| corpus com', len(corpus), 'caracteres')
"""),
        markdown("""## 1. Arquitetura como configuração

`ConfigGPT` expõe exatamente as escolhas que separaram GPT-2 (2019) dos modelos atuais.
Rode primeiro com RMSNorm + RoPE + SwiGLU; nos exercícios você volta para as opções do
GPT-2 original e compara a mesma métrica.""",),
        codigo("""JANELA = 128
torch.manual_seed(42)
config = ConfigGPT(
    vocab=tok.get_vocab_size(), dim=128, camadas=4, cabecas=4,
    janela_ctx=JANELA, abandono=0.1,
    norm='rmsnorm',   # 'layernorm' = GPT-2 original
    pos='rope',       # 'aprendido'  = GPT-2 original
    mlp='swiglu',     # 'gelu'         = FFN original
)
modelo = GPT(config)
modelo.init_pesos(semente=42)
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
modelo = modelo.to(disp)
print(f'{modelo.contar_parametros()/1e6:.2f} M parâmetros | dispositivo: {disp}')
"""),
        markdown("""## 2. Dataset pelo próprio núcleo

`dividir_corpus` separa trem/validação por parágrafos com semente fixa e
`montar_dataset` produz os pares *(entrada → próximo token)* — o deslocamento
que o `Trainer` do caderno 01 escondia acontece aqui explicitamente."""),
        codigo("""trem_texto, val_texto = dividir_corpus(corpus, semente=42)
x_trem, y_trem = montar_dataset(tok, trem_texto, JANELA, stride=JANELA // 2)
x_val, y_val = montar_dataset(tok, val_texto, JANELA)
print(f'trem: {tuple(x_trem.shape)} | validação: {tuple(x_val.shape)}')

@torch.no_grad()
def perda_media(x, y, lote=32, max_lotes=16):
    modelo.eval()
    perdas = []
    for i in range(min(max_lotes, -(-len(x) // lote))):
        _, p = modelo(x[i * lote:(i + 1) * lote].to(disp), y[i * lote:(i + 1) * lote].to(disp))
        perdas.append(float(p))
    return sum(perdas) / len(perdas)

print(f'perda inicial (modelo aleatório): {perda_media(x_val, y_val):.3f}')
"""),
        markdown("""## 3. Treino na mão

240 passos de AdamW com clip de gradiente — o mínimo que o `Trainer` faz por trás.
Repare como a perda de validação despenca: o corpus é repetitivo, e isso é proposital
para o smoke test (não confunda decorar variações de 6 frases com linguagem real)."""),
        codigo("""otim = torch.optim.AdamW(modelo.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.1)
L, PASSOS = 32, 240
modelo.train()
for passo in range(PASSOS):
    ini = (passo * L) % max(1, len(x_trem) - L)
    _, perda = modelo(x_trem[ini:ini + L].to(disp), y_trem[ini:ini + L].to(disp))
    otim.zero_grad(set_to_none=True)
    perda.backward()
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
    otim.step()
    if (passo + 1) % 60 == 0:
        val = perda_media(x_val, y_val)
        print(f'passo {passo + 1:3d} | perda trem {float(perda):.3f} | val {val:.3f}')
        modelo.train()
"""),
        markdown("""## 4. Geração com o `gerar()` do núcleo

`labia.models.gpt.gerar` aplica temperatura + top-k e devolve **só a continuação**."""),
        codigo("""modelo.eval()
saida = gerar_com_prompt(modelo, 'Modelos de linguagem', tok, passos_max=60, temperatura=0.8, topo_k=40, semente=7)
print(saida)
"""),
        markdown("""## Exercícios

1. Troque `norm`, `pos` e `mlp` para os valores do GPT-2 original e registre as duas
   perdas de validação no mesmo caderno — o que mudou?
2. Reduza `cabecas` para 2 e aumente `n_cabecas_kv` para 1 (GQA): mesmo custo, menos cache.
3. Gere com `temperatura=0` (determinístico) e compare com a amostragem.
4. Continue para o caderno 09, onde a FFN vira MoE.""",),
    ], "metadata": metadados, "nbformat": 4, "nbformat_minor": 5}

    lora_nb = {"cells": [
        markdown("""# 08 — LoRA e QLoRA no núcleo do Lab-IA

O caderno 02 usou PEFT/bitsandbytes para LoRA/QLoRA. O Lab-IA tem implementações
próprias (`labia.models.lora`, `labia.models.quant`) para você ver o mecanismo sem
camada de biblioteca: base congelada + ramo de baixo posto *B·A*, quantização int8/NF4
e fusão dos pesos no fim.

Pré-requisito: caderno 01 executado (mesmo tokenizer e corpus do caderno 07).
"""),
        codigo("""import copy
from pathlib import Path
import torch
from tokenizers import Tokenizer

from labia.models.gpt import ConfigGPT, GPT
from labia.models.lora import (
    aplicar_lora, carregar_adaptador, mesclar_lora, salvar_adaptador,
)
from labia.trainer.dados import montar_dataset

ARQ_TOKENIZER = Path('artifacts/tokenizer_puro/tokenizer.json')
if not ARQ_TOKENIZER.is_file():
    raise FileNotFoundError('Tokenizer ausente. Execute primeiro o caderno 01.')
tok = Tokenizer.from_file(str(ARQ_TOKENIZER))
corpus = Path('data/corpus.txt').read_text(encoding='utf-8')

JANELA = 128
torch.manual_seed(42)
config = ConfigGPT(vocab=tok.get_vocab_size(), dim=128, camadas=4, cabecas=4,
                   janela_ctx=JANELA, abandono=0.0, norm='rmsnorm', pos='rope', mlp='swiglu')
modelo = GPT(config)
modelo.init_pesos(semente=42)
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
modelo = modelo.to(disp)

def treinar(modelo_alvo, x, y, passos=60, lote=16, lr=5e-4):
    treinaveis = [p for p in modelo_alvo.parameters() if p.requires_grad]
    otim = torch.optim.AdamW(treinaveis, lr=lr)
    modelo_alvo.train()
    for passo in range(passos):
        ini = (passo * lote) % max(1, len(x) - lote)
        _, perda = modelo_alvo(x[ini:ini + lote].to(disp), y[ini:ini + lote].to(disp))
        otim.zero_grad(set_to_none=True)
        perda.backward()
        otim.step()
    modelo_alvo.eval()
    return float(perda)

@torch.no_grad()
def ce_do(modelo_alvo, x, y, lote=32):
    total, n = 0.0, 0
    for i in range(0, len(x), lote):
        _, p = modelo_alvo(x[i:i + lote].to(disp), y[i:i + lote].to(disp))
        total += float(p); n += 1
    return total / max(1, n)

x_trem, y_trem = montar_dataset(tok, corpus, JANELA, stride=JANELA // 2)
treinar(modelo, x_trem, y_trem, passos=120)
estado_base = copy.deepcopy(modelo.state_dict())
print('base densa treinada; valência do próximo experimento:', round(ce_do(modelo, x_trem, y_trem), 3))
"""),
        markdown("""## 1. Domínio novo, três caminhos de ajuste

O "domínio novo" abaixo são regras de lógica proposicional fora do corpus original.
Três estratégias com o mesmo orçamento de passos:

| Estratégia | O que treina |
|---|---|
| **A. Full fine-tuning** | todos os pesos |
| **B. LoRA** | só os ramos A/B injetados (base congelada) |
| **C. QLoRA** | idem, mas a base vive quantizada em **NF4** |

Medimos as duas coisas que interessam: aprender o domínio novo **e** não esquecer o corpus."""),
        codigo("""regras = '\\n'.join([
    'Premissa 1: se chover entao a rua molha.',
    'Premissa 2: choveu.',
    'Conclusao: a rua esta molha. (modus ponens)',
    'Premissa 1: se estudar entao aprender.',
    'Premissa 2: nao aprendeu.',
    'Conclusao: nao estudou. (modus tollens)',
    'Se A implica B e B implica C, entao A implica C.',
    'Nao e verdade que A e nao A.',
]) * 6
x_novo, y_novo = montar_dataset(tok, regras, JANELA, stride=JANELA // 2)

def fabricar():
    m = GPT(config).to(disp)
    m.load_state_dict(estado_base)
    return m

resultados = {}

# A — full fine-tuning
m = fabricar()
treinar(m, x_novo, y_novo, passos=60)
resultados['A_full'] = (ce_do(m, x_novo, y_novo), ce_do(m, x_trem, y_trem),
                        sum(p.numel() for p in m.parameters() if p.requires_grad))

# B — LoRA
m = fabricar()
stats = aplicar_lora(m, r=8, alpha=16)
treinar(m, x_novo, y_novo, passos=60, lr=1e-3)
resultados['B_lora'] = (ce_do(m, x_novo, y_novo), ce_do(m, x_trem, y_trem), stats['treinaveis'])

# C — QLoRA (base NF4 congelada)
m = fabricar()
stats = aplicar_lora(m, r=8, alpha=16, quant='nf4')
treinar(m, x_novo, y_novo, passos=60, lr=1e-3)
resultados['C_qlora'] = (ce_do(m, x_novo, y_novo), ce_do(m, x_trem, y_trem), stats['treinaveis'])

total_param = sum(t.numel() for t in estado_base.values())
print(f"{'estratégia':<8} {'CE domínio novo':>16} {'CE corpus (esquecimento)':>26} {'parâmetros treináveis':>24}")
for nome, (novo, antigo, trein) in resultados.items():
    print(f'{nome:<8} {novo:>16.3f} {antigo:>26.3f} {trein:>18,} ({trein / total_param:.1%})')
"""),
        markdown("""## 2. Salvar e recarregar o adaptador

O checkpoint guarda só os ramos A/B treináveis (e os buffers quantizados, se houver).
Isto vem **antes** da fusão: depois de fundir, não existe mais adaptador para salvar."""),
        codigo("""pasta_adaptador = Path('artifacts/lora_labia')
m = fabricar()
aplicar_lora(m, r=8, alpha=16)
treinar(m, x_novo, y_novo, passos=60, lr=1e-3)
salvar_adaptador(m, pasta_adaptador, meta_extra={'dominio': 'regras-logicas'})

m_recuperado = fabricar()
aplicar_lora(m_recuperado, r=8, alpha=16)
meta = carregar_adaptador(m_recuperado, pasta_adaptador)
print('meta do adaptador:', {k: meta[k] for k in ('r', 'alpha', 'quant', 'treinaveis')})
print('CE identico após recarregar:', abs(ce_do(m_recuperado, x_novo, y_novo) - ce_do(m, x_novo, y_novo)) < 1e-6)
"""),
        markdown("""## 3. Fundir o adaptador (o `merge_and_unload` do caderno 02)

A fusão aplica `W ← W + escala·(B·A)` e devolve lineares puros: treino termina,
inferência sem custo de ramo extra."""),
        codigo("""amostra = x_trem[:4].to(disp)
antes = m(amostra)[0].clone()
trocados = mesclar_lora(m)
depois = m(amostra)[0]
print(f'{trocados} lineares fundidos | maior diferença antes/depois: '
      f'{(antes - depois).abs().max().item():.2e} (esperado: ~0, só arredondamento)')
"""),
        markdown("""## Correspondência com o caderno 02

| Caderno 02 (bibliotecas) | Aqui (núcleo Lab-IA) |
|---|---|
| `LoraConfig` + `get_peft_model` | `aplicar_lora(modelo, r, alpha)` |
| `BitsAndBytesConfig(load_in_4bit=True)` | `aplicar_lora(..., quant='nf4')` |
| `merge_and_unload()` | `mesclar_lora(modelo)` |
| `save_pretrained` / `PeftModel.from_pretrained` | `salvar_adaptador` / `carregar_adaptador` |

A quantização nativa é mais simples que a do bitsandbytes (sem double-quant nem
paginação) — suficiente para estudo; o caderno 02 continua sendo o caminho de produção.

## Exercícios

1. Rode B e C com `r=1` e `r=32`: quanto de posto basta para este domínio?
2. Troque `quant='nf4'` por `'int8'` e compare tamanho real (`labia.models.quant.tamanho_state_dict`).
3. Amplie `alvos` além de `ALVOS_PADRAO` e observe o efeito nos parâmetros treináveis.""",),
    ], "metadata": metadados, "nbformat": 4, "nbformat_minor": 5}

    avaliacao = {"cells": [
        markdown("""# 09 — Perplexidade contra uma régua medida + MoE

Duas ferramentas de medição do laboratório em um caderno:

1. **Perplexidade com baseline**: números absolutos de perda não significam nada sem uma
   régua. A régua mínima honesta é o classificador **unigrama** (frequência de tokens no
   trem) — um modelo que não o supera não aprendeu estrutura, decorou frequências.
2. **MoE e custo real por token**: o núcleo suporta FFN esparsa; medimos parâmetros
   *totais* versus *ativos por token* e o mapa de uso dos especialistas.

Pré-requisito: caderno 01 executado (carrega `artifacts/modelo_puro`).
"""),
        codigo("""from pathlib import Path
import math
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from labia.trainer.dados import dividir_corpus

MODEL_DIR = Path('artifacts/modelo_puro')
if not (MODEL_DIR / 'config.json').is_file():
    raise FileNotFoundError('Modelo ausente. Execute primeiro o caderno 01 até a célula de salvamento.')

disp = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
modelo = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(disp).eval()

corpus = Path('data/corpus.txt').read_text(encoding='utf-8')
trem_texto, val_texto = dividir_corpus(corpus, semente=42)
BLOCK = modelo.config.n_positions
print(f'contexto do modelo: {BLOCK} | dispositivo: {disp}')
"""),
        markdown("""## 1. Baseline unigrama (a régua)

`CE_unigrama = média(-log p(token))` no conjunto de validação, com suavização.
Perplexidade é `exp(CE)` — leia como "quantos candidatos, em média, o modelo considera
por token".""",),
        codigo("""trem_ids = tokenizer(trem_texto, add_special_tokens=False)['input_ids']
val_ids = tokenizer(val_texto, add_special_tokens=False)['input_ids']

contagem = Counter(trem_ids)
V = len(tokenizer)
alfa = 1.0
def log_p(tok_id: int) -> float:
    return math.log((contagem.get(tok_id, 0) + alfa) / (sum(contagem.values()) + alfa * V))

ce_uni = -sum(log_p(t) for t in val_ids) / len(val_ids)
print(f'unigrama: CE {ce_uni:.3f} | perplexidade {math.exp(ce_uni):.1f} | vocab {V} | tokens no val {len(val_ids)}')
"""),
        markdown("""## 2. Perplexidade do modelo no mesmo conjunto de validação"""),
        codigo("""# GPT2LMHeadModel já desloca os alvos internamente quando recebe `labels`.
@torch.no_grad()
def ce_modelo(ids):
    blocos = [ids[i:i + BLOCK] for i in range(0, len(ids) - 1, BLOCK)]
    total, n_tok = 0.0, 0
    for b in blocos:
        if len(b) < 2:
            continue
        entrada = torch.tensor([b], device=disp)
        saida = modelo(entrada, labels=entrada)
        total += float(saida.loss) * len(b)
        n_tok += len(b)
    return total / max(1, n_tok)

ce_mod = ce_modelo(val_ids)
ppl_mod = math.exp(ce_mod)
ppl_uni = math.exp(ce_uni)
print(f'modelo:    CE {ce_mod:.3f} | perplexidade {ppl_mod:.1f}')
print(f'unigrama:  CE {ce_uni:.3f} | perplexidade {ppl_uni:.1f}')
veredito = 'superou a régua' if ppl_mod < ppl_uni else 'NÃO superou a régua — mais treino ou mais dados'
print(f'relação: {ppl_uni / ppl_mod:.2f}× menos incerteza que o unigrama → {veredito}')
"""),
        markdown("""Repare que a comparação com uma métrica **absoluta** seria sem sentido em um corpus de
centenas de linhas repetidas; o que vale é a posição relativa à régua medida no mesmo
dados. Isso é o que o projeto usa como barra nas metas de treino.""",),
        markdown("""## 3. MoE: tamanho ≠ custo por token

Um FFN MoE com *n* especialistas guardando *n* pesos carrega só *k* por token.""",),
        codigo("""from labia.models.gpt import ConfigGPT, GPT
from labia.trainer.dados import montar_dataset
from tokenizers import Tokenizer

tok_puro = Tokenizer.from_file('artifacts/tokenizer_puro/tokenizer.json')
torch.manual_seed(42)
cfg_moe = ConfigGPT(vocab=tok_puro.get_vocab_size(), dim=128, camadas=2, cabecas=4,
                    janela_ctx=128, abandono=0.0, n_especialistas=4, top_k=1,
                    coef_auxiliar=0.01, norm='rmsnorm', pos='rope')
moe = GPT(cfg_moe).to(disp)
moe.init_pesos(semente=42)

x, y = montar_dataset(tok_puro, trem_texto, 128, stride=64)
lote = x[:16].to(disp)
moe.eval()
with torch.no_grad():
    moe(lote)
print('especialistas: 4 | top-k: 1')
print(f'parâmetros totais: {moe.contar_parametros():,} | ativos por token: {moe.contar_parametros_ativos():,}'
      f' ({moe.contar_parametros_ativos() / moe.contar_parametros():.0%})')
print('mapa de uso (recém-inicializado):', moe.mapa_uso_especialistas())
"""),
        codigo("""# Um pouco de treino: o roteador se equilibra com a perda auxiliar em ação.
otim = torch.optim.AdamW(moe.parameters(), lr=5e-4)
moe.train()
for passo in range(120):
    ini = (passo * 16) % max(1, len(x) - 16)
    _, perda = moe(x[ini:ini + 16].to(disp), y[ini:ini + 16].to(disp))
    otim.zero_grad(set_to_none=True)
    perda.backward()
    otim.step()
moe.eval()
with torch.no_grad():
    moe(lote)
print('aux do roteador no último passo:', moe.ultimo_aux)
print('mapa de uso após treino:', moe.mapa_uso_especialistas())
"""),
        markdown("""## Exercícios

1. Zere `coef_auxiliar` e treine de novo: o mapa degrada para "um especialista come tudo"?
2. Troque `top_k` para 2 e refaça a conta de parâmetros ativos.
3. Aplique este caderno ao modelo do caderno 02 (Qwen refinado) trocando o `MODEL_DIR`:
   a régua unigrama muda com o tokenizer — o que acontece com a perplexidade relativa?""",),
    ], "metadata": metadados, "nbformat": 4, "nbformat_minor": 5}

    salvar("07_o_gpt_nativo_do_labia.ipynb", nativo)
    salvar("08_lora_e_quantizacao_no_labia.ipynb", lora_nb)
    salvar("09_perplexidade_regua_e_moe.ipynb", avaliacao)


def main() -> None:
    atualizar_modelo_puro()
    atualizar_qlora()
    novos_cadernos()
    cadernos_de_estudo()
    for caminho in PASTA.glob("[0-9]*.ipynb"):
        salvar(caminho.name, carregar(caminho.name))
    print("Cadernos atualizados.")


if __name__ == "__main__":
    main()
