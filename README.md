# PyTorch-implementation-of-LLM-TKESS
PyTorch implementation of LLM-TKESS: a text-based knowledge-embedded soft sensing framework based on large language models for industrial process tasks.
# LLM-TKESS
PyTorch implementation of the paper:

"A Text-Based Knowledge-Embedded Soft Sensing Modeling Approach for Industrial Process Tasks Based on Large Language Model"
## Framework Overview

LLM-TKESS consists of two training stages:

### Stage 1: LLM-SS Alignment Phase

Autoregressive parameter-efficient fine-tuning (PEFT) is employed to align industrial process variables with the semantic space of large language models, producing a soft sensing foundation model named **LLM-SS**.
```bash
--model_id GPT4Indpensim \
--model LLM_TKESS \
--is_gpt 1 \
--init_checkpoint checkpoints/gpt2-pytorch_model.bin \
--seq_len 96 \
--enc_in 7 \
--c_out 7 \
--batch_size 64 \
--learning_rate 0.0001 \
--train_epochs 30 \
--decay_fac 0.5 \
--d_model 768 \
--d_ff 768 \
--n_heads 4 \
--dropout 0.3 \
--gpt_layers 6 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--lora_layer 2 \
--freq 0 \
--percent 100 \
--itr 1 \
--tmax 20 \
--cos 1
### Stage 2: Downstream Soft Sensing Adaptation

Two downstream soft sensing paradigms are developed based on lightweight adapter tuning:

- **LLM-DSS**
  - LLM-based Data-driven Soft Sensor

- **LLM-PDSS**
  - Prompt and Data Mixed Embedding-driven Soft Sensor
