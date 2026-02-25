## Vak - Fine Tuned CodeLLM (SmolLM-135M + QLoRA)
Vak (VishwaKarma): A metaphor for creation, architecture, and building something new.
Just as VishwaKarma is the divine architect, Vak represetns reshaping a based model into a specialized code assistant through fine-tuning.

## Overview
Vak is a lightweight fine tuned CodeLLM built using:
- SmolLM-135M(from Hugging Face)
- PEFT (Parameter Efficient Fine Tuning)
- QLoRA (4bit quanitezed LoRA fine tuning)
- Managed with uv for dependency and environment management

This project focuses on:
1. Efficient fine tuning on limited hardware
2. Clean ML Engineering practices
3. Modular Architecture

### Base Model

Fine tuned:
1. Model: ```HuggingFaceTB/SmolLM-135M```
2. Parameters: ~135M
3. Type: Causal Language Model

SmolLM-135M is lightweight and suitable for:
- local experimentation
- low VRAM GPUs
- PEFT based research workflows

### Fine Tuning strategy
Vak uses:
1. QLoRA (4 bit quantization)
    - Load model in 4 bit
    - NF4 quantization
    - Memory Efficient

2. PEFT (LoRA adapters)
    - inject low rank adapters into attention layers
    - train only a small % of total parameters
    - Faster convergence
    - Reduced memory footprint

### Installation using ```uv```
This project uses uv for fast dependency management.
1. Install uv
```bash
pip install uv
```
2. Create environment & install dependencies
```bash
uv venv
```
and with pyproject.toml:
```bash
uv sync
```

Vak is built with:
- Config driven training rather than hardcoded hyperparameters
- Reproducible experiments
- Minimal yet scalable architecture
