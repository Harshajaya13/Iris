# Iris Architecture & Technical Implementation

This document provides a technical breakdown of **Iris**—a Causal GPT Language Model built from scratch in PyTorch. 

The codebase is organized into **3 distinct phases** depending on whether you want the main PyTorch production pipeline, educational notebooks with step-by-step comments, or raw comment-less Python implementations built from basic tensor math.

<br/>

---

<br/>

## The 3 Codebase Phases

| Phase | Path | Purpose | Content Type |
|---|---|---|---|
| **Phase 1: Production Pipeline** | `./` (Root) | Main execution pipeline ready to prepare data, train, and generate text. | Standard PyTorch (`torch.nn` modules) |
| **Phase 2: Educational Deep-Dives** | `Optional/` | Interactive notebooks breaking down Transformer building blocks step-by-step. | Notebooks with detailed comments & explanations |
| **Phase 3: Raw Clean Code** | `Optional/CleanCode/` | Low-level Python implementations built from scratch without high-level PyTorch abstractions. | Comment-less raw Python (minimal PyTorch abstractions) |

<br/>

---

<br/>

## Main Architecture (`model.py`)

Iris is implemented as an autoregressive decoder-only Transformer built from scratch using PyTorch (`torch.nn`).

<br/>

### Model Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| **Total Parameters** | `~20 Million` | Custom proof-of-concept decoder scale |
| **Vocabulary Size** (`vocab_size`) | `50,257` | GPT-2 BPE Tokenizer Vocabulary |
| **Sequence Length** (`seq_len`) | `64` | Maximum context window length |
| **Embedding Dimension** (`num_dims`) | `128` | Model hidden dimension ($d_{model}$) |
| **Attention Heads** (`num_heads`) | `4` | Multi-head attention count ($d_{head} = 32$) |
| **Transformer Layers** (`num_layers`) | `4` | Number of stacked Transformer blocks |
| **Dropout** (`p`) | `0.1` | Regularization dropout rate |

<br/>

---

<br/>

### Inference Strategy: Internalized Wisdom (No RAG)

Unlike conventional search-based AI setups that use Retrieval-Augmented Generation (RAG) to query vector databases at runtime, Iris is designed to **internalize philosophical principles directly within its weights**. 

- **No runtime document search**: Eliminates vector database lookups and raw scripture snippet injection.

- **Philosophical Reconnection**: The model is trained to process user crisis inputs through internal attention layers and reconnect situations directly to timeless philosophical patterns.

<br/>

---

<br/>

### Component Breakdown

#### 1. Causal Self-Attention (`CausalSelfAttention`)
- **Projections**: Linear projections for Query ($Q$), Key ($K$), and Value ($V$) without bias.
- **Scaled Dot-Product Attention**:
  ```
  Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + Mask ) @ V
  ```
- **Causal Masking**: Upper triangular mask (`torch.triu`) setting future token attention scores to $-\infty$.
- **Output Projection & Dropout**: Linear output projection followed by residual dropout.

<br/>

#### 2. Feed-Forward Network (`MyMLP`)
- **Expansion**: $4 \times d_{model}$ ($128 \rightarrow 512 \rightarrow 128$).
- **Activation**: Gaussian Error Linear Unit (`GELU`).
- **Dropout**: Regularization dropout after final projection.

<br/>

#### 3. Transformer Block (`Transformer`)
- Pre-LayerNorm block design for enhanced training stability:
  $$\text{Input} \rightarrow \mathbf{x} + \text{Attn}(\text{LN}_1(\mathbf{x})) \rightarrow \mathbf{x}' + \text{MLP}(\text{LN}_2(\mathbf{x}'))$$

<br/>

#### 4. Top-Level GPT Model (`GPT`)
- **Embeddings**: Token embeddings + learned 1D positional embeddings.
- **Blocks**: Stack of `num_layers` Transformer blocks.
- **Head**: Final LayerNorm (`ln_f`) followed by linear LM head mapping to `vocab_size`.

<br/>

---

<br/>

## Pipeline & Data Workflow

### Data Preparation (`prepare.py`)
- Reads raw training corpus (`canary_dataset.txt`).
- **Corpus Domains**: Synthesizes Ancient Wisdom, Philosophy, Ethics, Human Behavior, Neuroscience, Stories & Parables, and Structured Reasoning Patterns.
- Encodes tokens using `tiktoken` (GPT-2 BPE).
- Splits data 90% training / 10% validation.
- Saves binary token arrays as `np.uint16` (`train.bin`, `val.bin`).

<br/>

### Data Loading (`dataset.py`)
- Uses NumPy memory-mapping (`np.memmap`) for zero-copy binary reading from disk.
- Pins memory on CUDA devices (`pin_memory()`) for fast host-to-GPU memory transfer.

<br/>

---

<br/>

## Training Pipeline (`train.py`)

| Hyperparameter | Setting |
|---|---|
| **Optimizer** | `AdamW` |
| **Learning Rate** | `1e-3` |
| **Max Iterations** | `1000` |
| **Batch Size** | `16` |
| **Gradient Clipping** | `max_norm = 1.0` |
| **Eval Interval** | Every `100` steps |
| **Checkpoint Output** | `checkpoint.pt` |

<br/>

---

<br/>

## Sampling & Generation (`generate.py`)

1. **Context Cropping**: Automatically crops prompt sequences exceeding `seq_len=64`.

2. **Temperature Scaling**: Logits scaled by $T = 0.8$.

3. **Top-K Filtering**: Candidate pool restricted to top $K = 40$ tokens.

4. **Multinomial Sampling**: Stochastic token sampling via `torch.multinomial`.

<br/>

---

<br/>

## Execution Guide

Run the pipeline from the project root:

```bash
# 1. Prepare raw text data into bin artifacts
python prepare.py

# 2. Train the GPT model and save weights
python train.py

# 3. Launch interactive generation session
python generate.py
```

<br/>

---

<br/>

## How to Navigate the Codebase

- **Need the main PyTorch production pipeline?**  
  Look directly at the root files (`model.py`, `train.py`, `dataset.py`, `prepare.py`, `generate.py`). They contain clean, modular execution scripts using standard PyTorch `torch.nn` modules.

- **Want step-by-step educational notebooks with detailed comments?**  
  Explore the `Optional/` directory. It contains interactive Jupyter Notebooks packed with explanations and step-by-step comments covering every Transformer concept.

- **Want raw Python implementations without comments or PyTorch abstractions?**  
  Check the `Optional/CleanCode/` directory for low-level, comment-less Python code that builds neural layers, forward/backward passes, and optimizers directly from raw tensor math.
