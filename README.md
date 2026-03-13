# HappyTorch

**A PyTorch coding practice platform — covering LLM, Diffusion, PEFT, RLHF, and more**

*Like LeetCode, but for tensors. Self-hosted. Supports both Jupyter and Web interfaces. Instant auto-grading feedback. No GPU required.*

[中文版 README](README_CN.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Problems](https://img.shields.io/badge/problems-36-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen?style=flat-square)

> **News**
> - 2026-03-12: Web UI now groups problems by category (Fundamentals, Attention, RLHF, etc.) with collapsible sections in the sidebar for easier topic-based practice.
> - 2026-03-10: Thanks to [SongHuang1](https://github.com/SongHuang1) for contributing the MLP XOR training problem (pure NumPy, manual forward + backward). Fixed Web UI issues: class-based tasks (LoRA, SwiGLU, etc.) now work correctly, added `nn`/`F`/`numpy`/`math` to execution namespace, fixed OpenMP crash on Windows, added MHA solution lookup, added 60s request timeout.
> - 2026-03-09: Thanks to [chaoyitud](https://github.com/chaoyitud) for adding ML and RLHF practice problems. Thanks to [fiberproduct](https://github.com/fiberproduct) for fixing `torch_judge/tasks/rope.py`. Welcome everybody to contribute more problems!
> - 2026-03-06: The plugin [happytorch-plugin](https://github.com/Rivflyyy/happytorch-plugin) has been released.

---

## Why HappyTorch?

If you're learning deep learning or preparing for ML interviews, you might have encountered these challenges:

- You've read many papers, but don't know where to start when it comes to implementing things from scratch
- You're asked to implement `softmax` or `MultiHeadAttention` in an interview, and your mind goes blank
- You want to deeply understand Transformer, LoRA, Diffusion, RLHF, but lack systematic practice

**HappyTorch** provides a friendly hands-on practice environment with **36 curated problems**, from basic activation functions to complete Transformer components and RLHF algorithms.

| Feature | Description |
|---------|-------------|
| **36 curated problems** | From basics to advanced, covering mainstream deep learning topics |
| **Auto-grading** | Instant feedback showing what you got right and where to improve |
| **Two interfaces** | LeetCode-like Web UI (Monaco Editor) or Jupyter notebooks |
| **Helpful hints** | Get nudges when stuck, not full spoilers |
| **Reference solutions** | Compare and learn after your own attempt |
| **Progress tracking** | Record your learning journey |

---

## Quick Start

```bash
# 1. Create and activate environment
conda create -n torchcode python=3.11 -y
conda activate torchcode

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy
pip install -e .

# 3. Prepare notebooks
python prepare_notebooks.py

# 4a. Launch Web Mode (recommended)
pip install fastapi uvicorn python-multipart
python start_web.py
# Open http://localhost:8000

# 4b. Or launch Jupyter Mode
python start_jupyter.py
# Open http://localhost:8888
```

### Docker

```bash
# Web Mode (default, recommended)
make run                # build & start → http://localhost:8000
make stop               # stop container

# Jupyter Mode
make jupyter            # build & start → http://localhost:8888

# Or pull the pre-built image directly
docker compose up -d    # pulls ghcr.io/chan/happytorch:latest
```

Progress data (`data/progress.json`) is persisted via Docker volume.

---

## Web Mode

A LeetCode-like practice interface with:

- **Monaco Editor** — VS Code's editor with Python syntax highlighting
- **Random / Sequential Mode** — Get random unsolved problems or work through them in order
- **Instant Testing** — Run tests with one click (`Ctrl+Enter`)
- **Solution Tab** — View reference solutions with markdown explanation and copyable code
- **Progress Dashboard** — Track solved / attempted / todo status
- **Dark Theme** — Modern, eye-friendly interface

```bash
pip install fastapi uvicorn python-multipart
python start_web.py
# Open http://localhost:8000
```

---

## Problem Set (36 Problems)

### Fundamentals

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 1 | ReLU | `relu(x)` | ![Easy](https://img.shields.io/badge/-Easy-4CAF50?style=flat-square) | Activation functions, element-wise ops |
| 2 | Softmax | `my_softmax(x, dim)` | ![Easy](https://img.shields.io/badge/-Easy-4CAF50?style=flat-square) | Numerical stability, exp/log tricks |
| 3 | Linear Layer | `SimpleLinear` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | y = xW^T + b, Kaiming init, nn.Parameter |
| 4 | LayerNorm | `my_layer_norm(x, g, b)` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Normalization, affine transform |
| 7 | BatchNorm | `my_batch_norm(x, g, b)` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Batch vs layer statistics, train/eval |
| 8 | RMSNorm | `rms_norm(x, weight)` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | LLaMA-style norm, simpler than LayerNorm |

### Attention Mechanisms

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 5 | Scaled Dot-Product Attention | `scaled_dot_product_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | softmax(QK^T/sqrt(d_k))V |
| 6 | Multi-Head Attention | `MultiHeadAttention` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Parallel heads, split/concat, projections |
| 9 | Causal Self-Attention | `causal_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Autoregressive masking, GPT-style |
| 10 | Grouped Query Attention | `GroupQueryAttention` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | GQA (LLaMA 2), KV sharing |
| 11 | Sliding Window Attention | `sliding_window_attention(Q, K, V, w)` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Mistral-style local attention |
| 12 | Linear Attention | `linear_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Kernel trick, O(n*d^2) |

### Full Architecture

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 13 | GPT-2 Block | `GPT2Block` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Pre-norm, causal MHA + MLP, residual |

### Modern Activation Functions *(V2)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 14 | GELU | `gelu(x)` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Gaussian CDF, erf, BERT/GPT/DiT |
| 15 | SiLU (Swish) | `silu(x)` | ![Easy](https://img.shields.io/badge/-Easy-4CAF50?style=flat-square) | x * sigmoid(x), LLaMA component |
| 16 | SwiGLU | `SwiGLU` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Gated activation, LLaMA MLP |

### Parameter-Efficient Fine-Tuning *(V2)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 17 | LoRA | `LoRALinear` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Low-rank BA, zero-init B, alpha/r scaling |
| 18 | DoRA | `DoRALinear` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Weight decomposition, magnitude + direction |

### Conditional Modulation — Diffusion *(V2)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 19 | AdaLN | `AdaLN` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Adaptive LayerNorm, DiT-style |
| 20 | AdaLN-Zero | `AdaLNZero` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Zero-init gate, stable training |
| 21 | FiLM | `FiLM` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Feature-wise modulation |

### LLM Inference *(V2)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 22 | RoPE | `apply_rotary_pos_emb(x, pos)` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Rotary embedding, 2D rotation |
| 23 | KV Cache | `KVCache` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Incremental caching for generation |

### Diffusion Training *(V2)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 24 | Sigmoid Schedule | `sigmoid_schedule(t, ...)` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | S-curve noise schedule |

### ML Fundamentals & Decoding *(V3 — Community)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 25 | K-Means Clustering | `kmeans` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Iterative centroid update, assignment |
| 26 | K-Nearest Neighbors | `knn_predict` | ![Easy](https://img.shields.io/badge/-Easy-4CAF50?style=flat-square) | Distance-based classification |
| 27 | MLP Backward | `mlp_backward` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Hand-written backprop for 2-layer MLP |
| 36 | MLP XOR Training | `mlp_xor` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Complete MLP training loop (pure NumPy), He init, MSE loss |
| 28 | Greedy Decoding | `greedy_decode` | ![Easy](https://img.shields.io/badge/-Easy-4CAF50?style=flat-square) | Argmax token selection |
| 29 | Beam Search | `beam_search_decode` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Beam search decoding strategy |
| 30 | Temperature Sampling | `temperature_sample` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Temperature-scaled softmax sampling |
| 31 | Top-k Sampling | `top_k_sample` | ![Medium](https://img.shields.io/badge/-Medium-FF9800?style=flat-square) | Truncated probability distribution |
| 32 | Top-p Sampling | `top_p_sample` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Nucleus sampling |

### RLHF *(V3 — Community)*

| # | Problem | Function / Class | Difficulty | Key Concepts |
|:-:|---------|-----------------|:----------:|--------------|
| 33 | PPO Clipped Loss | `ppo_clipped_loss` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Clipped surrogate objective |
| 34 | DPO Loss | `dpo_loss` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Direct Preference Optimization |
| 35 | GRPO Loss | `grpo_loss` | ![Hard](https://img.shields.io/badge/-Hard-F44336?style=flat-square) | Group Relative Policy Optimization |

---

## How It Works

### Workflow

```
1. Open a blank notebook / web editor    →  Read the problem description
2. Implement your solution               →  Use only basic PyTorch ops
3. Run the judge                         →  check("relu")
4. See instant colored feedback          →  ✅ pass / ❌ fail per test case
5. Stuck? Get a hint                     →  hint("relu")
6. Review the reference solution         →  01_relu_solution.ipynb
```

### In-Notebook API

```python
from torch_judge import check, hint, status

check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint without full spoiler
status()                    # Progress dashboard
```

---

## Suggested Study Plan

> **Total: ~15–20 hours spread across 4–5 weeks**

| Week | Focus | Problems | Est. Time |
|:----:|-------|----------|:---------:|
| **1** | Foundations | ReLU, Softmax, Linear, LayerNorm, BatchNorm, RMSNorm | 1–2 hrs |
| **2** | Attention | SDPA, MHA, Causal, GQA, Sliding Window, Linear Attention | 3–4 hrs |
| **3** | Modern Components | GELU, SiLU, SwiGLU, LoRA, DoRA | 2–3 hrs |
| **4** | Advanced Topics | AdaLN, FiLM, RoPE, KV Cache, GPT-2 Block | 3–4 hrs |
| **5** | ML & RLHF | K-Means, KNN, MLP Backward, MLP XOR, Decoding Strategies, PPO, DPO, GRPO | 3–4 hrs |

---

## Adding Your Own Problems

HappyTorch uses auto-discovery — just drop a new file in `torch_judge/tasks/`:

```python
# torch_judge/tasks/my_task.py
TASK = {
    "title": "My Custom Problem",
    "difficulty": "Medium",       # Easy / Medium / Hard
    "function_name": "my_function",
    "hint": "Think about broadcasting...",
    "tests": [
        {"name": "Basic test", "code": "assert ..."},
    ]
}
```

No registration needed. Then create corresponding notebooks in `templates/` and `solutions/`.

---

## FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>
<br>
No. Everything runs on CPU. The problems test correctness and understanding, not throughput.
</details>

<details>
<summary><b>How are solutions graded?</b></summary>
<br>
The judge runs your function against multiple test cases using <code>torch.allclose</code> for numerical correctness, verifies gradients flow properly via autograd, and checks edge cases specific to each operation.
</details>

<details>
<summary><b>Can I save my progress?</b></summary>
<br>
Progress is saved in <code>data/progress.json</code>. Your solutions in <code>notebooks/</code> persist between sessions. To start fresh, simply re-copy templates.
</details>

<details>
<summary><b>What's different from the original TorchCode?</b></summary>
<br>
HappyTorch extends <a href="https://github.com/duoan/TorchCode">TorchCode</a> (13 problems) with 23 additional problems covering modern activations, LoRA/DoRA, Diffusion components, LLM inference, decoding strategies, RLHF algorithms, and a manual NumPy MLP training exercise.
</details>

---

## Acknowledgments

This project is based on [TorchCode](https://github.com/duoan/TorchCode) by [@duoan](https://github.com/duoan). If you find this project helpful, please also star the [original repository](https://github.com/duoan/TorchCode).

Community contributions:
- [chaoyitud](https://github.com/chaoyitud) — ML fundamentals and RLHF practice problems
- [fiberproduct](https://github.com/fiberproduct) — RoPE task fix
- [Rivflyyy](https://github.com/Rivflyyy) — [happytorch-plugin](https://github.com/Rivflyyy/happytorch-plugin)
- [SongHuang1](https://github.com/SongHuang1) — MLP XOR training problem

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If you find it useful, a Star would be appreciated.**

</div>
