# Laboratório Jupyter — treinamento de LLMs, Qwen3 e GGUF

Arquivos:

- `00_diagnostico_ambiente.ipynb` — hardware, versões e diretórios.
- `01_modelo_puro_do_zero.ipynb` — tokenizer e Transformer causal treinados do zero.
- `02_qwen3_0_6b_raciocinio_qlora.ipynb` — SFT com LoRA/QLoRA sobre Qwen3-0.6B.
- `03_exportar_e_testar_gguf.ipynb` — merge, conversão GGUF, Q4_K_M e llama.cpp.
- `04_finetune_direto_gguf_experimental.ipynb` — llama-finetune em F32, trilha WIP/avançada.
- `05_gerar_modelo_puro.ipynb` — carregar e testar o modelo didático treinado no caderno 01.
- `06_avaliar_respostas.ipynb` — bateria pequena e reproduzível para comparar respostas.
- `07_o_gpt_nativo_do_labia.ipynb` — o GPT do núcleo (`labia.models.gpt`) treinado na mão, com RMSNorm/RoPE/SwiGLU ligáveis.
- `08_lora_e_quantizacao_no_labia.ipynb` — LoRA/QLoRA nativos (`labia.models.lora`/`quant`): comparar ajuste total, LoRA e NF4, fundir e salvar o adaptador.
- `09_perplexidade_regua_e_moe.ipynb` — perplexidade medida contra uma régua unigrama + custo real de um FFN MoE.

Ordem recomendada: **00 → 01** (fundamentos) e depois **02 → 03** (refinamento/deploy). Os cadernos **05 → 06** dão sequência ao modelo do 01; **07 → 08 → 09** estudam o núcleo nativo do Lab-IA (dependem do 01). O 04 é opcional.

Os notebooks 01, 02, 05, 06, 07, 08 e 09 foram executados de ponta a ponta neste ambiente (transformers 5.17, CUDA). Os 03 e 04 exigem clonar e compilar o `llama.cpp` e não foram rodados aqui.

O dataset embutido nos notebooks é apenas um smoke test. Troque por um dataset de treinamento e um conjunto de avaliação independentes antes de comparar modelos. Os resultados de execução não são versionados: execute as células na ordem indicada para produzir seus próprios artefatos.
