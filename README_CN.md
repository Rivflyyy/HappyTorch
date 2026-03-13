# HappyTorch

**PyTorch 编程练习平台 — 涵盖 LLM、Diffusion、PEFT、RLHF 等方向**

*类似 LeetCode，但专注于张量运算。本地部署，支持 Jupyter 和 Web 两种界面，即时自动评测，无需 GPU。*

[English README](README.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Problems](https://img.shields.io/badge/题目数-36-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-无需-brightgreen?style=flat-square)

> **动态**
> - 2026-03-13：感谢 [wavetao2010](https://github.com/wavetao2010) 贡献 Docker 镜像支持，支持 Web/Jupyter 双模式及预构建镜像。
> - 2026-03-12：Web 界面侧边栏新增题目分类展示（基础层、注意力机制、RLHF 等），支持折叠/展开，方便按专题刷题。
> - 2026-03-10：感谢 [SongHuang1](https://github.com/SongHuang1) 贡献 MLP XOR 训练题目（纯 NumPy 手写前向+反向传播）。修复 Web 界面问题：class 类题目（LoRA、SwiGLU 等）现已正常工作，执行环境添加 `nn`/`F`/`numpy`/`math` 支持，修复 Windows 上 OpenMP 冲突导致的崩溃，修复 MHA 题解查找，前端增加 60 秒请求超时保护。
> - 2026-03-09：感谢 [chaoyitud](https://github.com/chaoyitud) 新增 ML 与 RLHF 练习题目，感谢 [fiberproduct](https://github.com/fiberproduct) 修复 `torch_judge/tasks/rope.py`。欢迎大家贡献更多题目！
> - 2026-03-06：插件 [happytorch-plugin](https://github.com/Rivflyyy/happytorch-plugin) 已发布。

---

## 为什么选择 HappyTorch？

如果你正在学习深度学习或准备 ML 面试，你可能遇到过这些问题：

- 看了很多论文，真到写代码时却不知从何下手
- 面试被要求从零实现 `softmax` 或 `MultiHeadAttention`，脑子一片空白
- 想深入理解 Transformer、LoRA、Diffusion、RLHF，但缺乏系统性的动手练习

**HappyTorch** 提供一个友好的实践环境，包含 **36 道精选题目**，从基础激活函数到完整 Transformer 组件和 RLHF 算法，帮助你循序渐进地提升。

| 特性 | 说明 |
|------|------|
| **36 道精选题目** | 从基础到进阶，覆盖主流深度学习技术栈 |
| **自动评测** | 即时反馈，清晰展示每个测试用例的通过/失败状态 |
| **双界面** | LeetCode 风格的 Web 界面（Monaco 编辑器）或 Jupyter Notebook |
| **智能提示** | 卡住时给你思路，而非直接给答案 |
| **参考题解** | 自己尝试后对照学习 |
| **进度追踪** | 记录你的学习旅程 |

---

## 快速开始

```bash
# 1. 创建并激活环境
conda create -n torchcode python=3.11 -y
conda activate torchcode

# 2. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy
pip install -e .

# 3. 准备练习 Notebook
python prepare_notebooks.py

# 4a. 启动 Web 模式（推荐）
pip install fastapi uvicorn python-multipart
python start_web.py
# 浏览器打开 http://localhost:8000

# 4b. 或启动 Jupyter 模式
python start_jupyter.py
# 浏览器打开 http://localhost:8888
```

### Docker 启动

```bash
# Web 模式（默认，推荐）
make run                # 构建并启动 → http://localhost:8000
make stop               # 停止容器

# Jupyter 模式
make jupyter            # 构建并启动 → http://localhost:8888

# 或直接使用预构建镜像（无需本地构建）
docker compose up -d                        # Web 界面 → http://localhost:8000
MODE=jupyter docker compose up -d           # Jupyter → http://localhost:8888
```

做题进度（`data/progress.json`）通过 Docker volume 持久化保存。

---

## Web 模式

类似 LeetCode 的练习界面，功能包括：

- **Monaco 编辑器** — VS Code 同款编辑器，Python 语法高亮
- **随机 / 顺序模式** — 随机抽取未解决的题目，或按顺序刷题
- **即时测试** — 一键运行测试（`Ctrl+Enter`）
- **题解栏目** — 查看参考实现，支持 Markdown 说明和一键复制代码
- **进度面板** — 追踪已通过 / 已尝试 / 待完成状态
- **暗色主题** — 现代化护眼界面

```bash
pip install fastapi uvicorn python-multipart
python start_web.py
# 浏览器打开 http://localhost:8000
```

---

## 题目列表（共 36 题）

### 基础层

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 1 | ReLU | `relu(x)` | ![Easy](https://img.shields.io/badge/-简单-4CAF50?style=flat-square) | 激活函数，逐元素操作 |
| 2 | Softmax | `my_softmax(x, dim)` | ![Easy](https://img.shields.io/badge/-简单-4CAF50?style=flat-square) | 数值稳定性，exp/log 技巧 |
| 3 | 线性层 | `SimpleLinear` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | y = xW^T + b，Kaiming 初始化 |
| 4 | LayerNorm | `my_layer_norm(x, g, b)` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 归一化，仿射变换 |
| 7 | BatchNorm | `my_batch_norm(x, g, b)` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | Batch 与 Layer 统计量，训练/推理行为 |
| 8 | RMSNorm | `rms_norm(x, weight)` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | LLaMA 风格归一化 |

### 注意力机制

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 5 | 缩放点积注意力 | `scaled_dot_product_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | softmax(QK^T/sqrt(d_k))V |
| 6 | 多头注意力 | `MultiHeadAttention` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 并行头，分割/拼接，投影矩阵 |
| 9 | 因果自注意力 | `causal_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 自回归掩码，GPT 风格 |
| 10 | 分组查询注意力 | `GroupQueryAttention` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | GQA（LLaMA 2），KV 共享 |
| 11 | 滑动窗口注意力 | `sliding_window_attention(Q, K, V, w)` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | Mistral 风格局部注意力 |
| 12 | 线性注意力 | `linear_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 核技巧，O(n*d^2) |

### 完整架构

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 13 | GPT-2 Block | `GPT2Block` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | Pre-norm，因果 MHA + MLP，残差连接 |

### 现代激活函数 *(V2)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 14 | GELU | `gelu(x)` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 高斯 CDF，erf，BERT/GPT/DiT |
| 15 | SiLU (Swish) | `silu(x)` | ![Easy](https://img.shields.io/badge/-简单-4CAF50?style=flat-square) | x * sigmoid(x)，LLaMA 组件 |
| 16 | SwiGLU | `SwiGLU` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 门控激活，LLaMA MLP |

### 参数高效微调 *(V2)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 17 | LoRA | `LoRALinear` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 低秩分解 BA，B 零初始化，alpha/r 缩放 |
| 18 | DoRA | `DoRALinear` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 权重分解，幅度 + 方向 |

### 条件调制 — Diffusion *(V2)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 19 | AdaLN | `AdaLN` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 自适应 LayerNorm，DiT 风格 |
| 20 | AdaLN-Zero | `AdaLNZero` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 零初始化门控，稳定训练 |
| 21 | FiLM | `FiLM` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 特征级线性调制 |

### LLM 推理组件 *(V2)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 22 | RoPE | `apply_rotary_pos_emb(x, pos)` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 旋转位置编码，二维旋转 |
| 23 | KV Cache | `KVCache` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 增量缓存，文本生成加速 |

### 扩散模型训练 *(V2)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 24 | Sigmoid 噪声调度 | `sigmoid_schedule(t, ...)` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | S 曲线噪声调度 |

### ML 基础与解码策略 *(V3 — 社区贡献)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 25 | K-Means 聚类 | `kmeans` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 迭代质心更新，样本分配 |
| 26 | K 近邻分类 | `knn_predict` | ![Easy](https://img.shields.io/badge/-简单-4CAF50?style=flat-square) | 基于距离的分类 |
| 27 | MLP 反向传播 | `mlp_backward` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 手写两层 MLP 反向传播 |
| 36 | MLP XOR 训练 | `mlp_xor` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 完整 MLP 训练循环（纯 NumPy），He 初始化，MSE 损失 |
| 28 | 贪心解码 | `greedy_decode` | ![Easy](https://img.shields.io/badge/-简单-4CAF50?style=flat-square) | Argmax 逐步选取 token |
| 29 | 束搜索 | `beam_search_decode` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | Beam Search 解码策略 |
| 30 | 温度采样 | `temperature_sample` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 温度缩放 softmax 采样 |
| 31 | Top-k 采样 | `top_k_sample` | ![Medium](https://img.shields.io/badge/-中等-FF9800?style=flat-square) | 截断概率分布 |
| 32 | Top-p 采样 | `top_p_sample` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 核采样（Nucleus Sampling） |

### RLHF *(V3 — 社区贡献)*

| # | 题目 | 函数 / 类 | 难度 | 核心概念 |
|:-:|------|----------|:----:|----------|
| 33 | PPO 截断策略损失 | `ppo_clipped_loss` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 截断代理目标函数 |
| 34 | DPO 损失 | `dpo_loss` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 直接偏好优化 |
| 35 | GRPO 损失 | `grpo_loss` | ![Hard](https://img.shields.io/badge/-困难-F44336?style=flat-square) | 群组相对策略优化 |

---

## 使用方法

### 练习流程

```
1. 打开空白 Notebook / Web 编辑器    →  阅读题目描述
2. 实现你的解法                      →  仅使用基础 PyTorch 操作
3. 运行评测                          →  check("relu")
4. 查看即时反馈                      →  ✅ 通过 / ❌ 失败（逐测试用例）
5. 卡住了？获取提示                  →  hint("relu")
6. 查看参考题解                      →  01_relu_solution.ipynb
```

### Notebook 内 API

```python
from torch_judge import check, hint, status

check("relu")               # 评测你的实现
hint("causal_attention")    # 获取提示（不是完整答案）
status()                    # 进度面板
```

---

## 建议学习计划

> **总计约 15–20 小时，分 4–5 周完成**

| 周 | 重点 | 题目 | 预计时间 |
|:--:|------|------|:--------:|
| **1** | 基础层 | ReLU、Softmax、Linear、LayerNorm、BatchNorm、RMSNorm | 1–2 小时 |
| **2** | 注意力 | SDPA、MHA、Causal、GQA、Sliding Window、Linear Attention | 3–4 小时 |
| **3** | 现代组件 | GELU、SiLU、SwiGLU、LoRA、DoRA | 2–3 小时 |
| **4** | 进阶话题 | AdaLN、FiLM、RoPE、KV Cache、GPT-2 Block | 3–4 小时 |
| **5** | ML 与 RLHF | K-Means、KNN、MLP 反向传播、MLP XOR 训练、解码策略、PPO、DPO、GRPO | 3–4 小时 |

---

## 添加自定义题目

HappyTorch 使用自动发现机制 — 只需在 `torch_judge/tasks/` 下新增文件：

```python
# torch_judge/tasks/my_task.py
TASK = {
    "title": "我的自定义题目",
    "difficulty": "Medium",       # Easy / Medium / Hard
    "function_name": "my_function",
    "hint": "考虑一下广播机制...",
    "tests": [
        {"name": "基础测试", "code": "assert ..."},
    ]
}
```

无需手动注册，评测引擎会自动发现新题目。然后在 `templates/` 和 `solutions/` 中创建对应的 Notebook 即可。

---

## 常见问题

<details>
<summary><b>需要 GPU 吗？</b></summary>
<br>
不需要。所有题目均在 CPU 上运行，测试的是正确性和理解深度，而非计算吞吐量。
</details>

<details>
<summary><b>如何评测？</b></summary>
<br>
评测引擎使用 <code>torch.allclose</code> 验证数值正确性，通过 autograd 检查梯度是否正确流动，并针对每个操作检测特定的边界情况。
</details>

<details>
<summary><b>进度可以保存吗？</b></summary>
<br>
进度保存在 <code>data/progress.json</code> 中。你在 <code>notebooks/</code> 中的解答在不同会话之间持久保存。如需重新开始，重新复制模板即可。
</details>

<details>
<summary><b>与 TorchCode 有什么不同？</b></summary>
<br>
HappyTorch 基于 <a href="https://github.com/duoan/TorchCode">TorchCode</a>（13 题）扩展了 23 道新题目，涵盖现代激活函数、LoRA/DoRA、Diffusion 组件、LLM 推理、解码策略、RLHF 算法和纯 NumPy 手写 MLP 训练。
</details>

---

## 致谢

本项目基于 [@duoan](https://github.com/duoan) 的 [TorchCode](https://github.com/duoan/TorchCode)。如果你觉得本项目有帮助，也请给[原项目](https://github.com/duoan/TorchCode)一个 Star。

社区贡献者：
- [chaoyitud](https://github.com/chaoyitud) — ML 基础和 RLHF 练习题目
- [fiberproduct](https://github.com/fiberproduct) — RoPE 题目修复
- [Rivflyyy](https://github.com/Rivflyyy) — [happytorch-plugin](https://github.com/Rivflyyy/happytorch-plugin) 插件
- [SongHuang1](https://github.com/SongHuang1) — MLP XOR 训练题目
- [wavetao2010](https://github.com/wavetao2010) — Docker 镜像支持

## 许可证

MIT License — 详见 [LICENSE](LICENSE)。

---

<div align="center">

**如果觉得有用，欢迎点个 Star。**

</div>
