# PyTorch-implementation-of-LLM-TKESS
PyTorch implementation of LLM-TKESS: a text-based knowledge-embedded soft sensing framework based on large language models for industrial process tasks.
# LLM-TKESS
PyTorch implementation of the paper:

"A Text-Based Knowledge-Embedded Soft Sensing Modeling Approach for Industrial Process Tasks Based on Large Language Model"
## Framework Overview

LLM-TKESS consists of two training stages:
https://github.com/842029379/PyTorch-implementation-of-LLM-TKESS/blob/main/README.md
### Stage 1: LLM-SS Alignment Phase

Autoregressive parameter-efficient fine-tuning (PEFT) is employed to align industrial process variables with the semantic space of large language models, producing a soft sensing foundation model named **LLM-SS**.

### Stage 2: Downstream Soft Sensing Adaptation

Two downstream soft sensing paradigms are developed based on lightweight adapter tuning:

- **LLM-DSS**
  - LLM-based Data-driven Soft Sensor

- **LLM-PDSS**
  - Prompt and Data Mixed Embedding-driven Soft Sensor
