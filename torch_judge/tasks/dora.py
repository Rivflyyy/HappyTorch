"""DoRA (Weight-Decomposed Low-Rank Adaptation) task — improved LoRA variant."""

TASK = {
    "category": "参数高效微调",
    "title": "DoRA Linear Layer",
    "difficulty": "Hard",
    "function_name": "DoRALinear",
    "hint": "DoRA decomposes W into magnitude m and direction V: W = m * V/||V||. LoRA updates V: V' = V + BA. Final: W' = m * (V + BA) / ||V + BA||. This separates magnitude and direction learning.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
dora = {fn}(in_features=64, out_features=128, rank=8)
x = torch.randn(2, 10, 64)
out = dora(x)
assert out.shape == (2, 10, 128), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has required components",
            "code": """
import torch, torch.nn as nn
dora = {fn}(in_features=32, out_features=64, rank=4)
assert hasattr(dora, 'W'), 'Need self.W (base weight)'
assert hasattr(dora, 'm'), 'Need self.m (magnitude vector)'
assert hasattr(dora, 'A'), 'Need self.A (LoRA down)'
assert hasattr(dora, 'B'), 'Need self.B (LoRA up)'
assert dora.m.shape == (64,), f'm shape: {dora.m.shape}, expected (64,)'
""",
        },
        {
            "name": "Magnitude initialized from base weight",
            "code": """
import torch
dora = {fn}(in_features=32, out_features=64, rank=4)
# m should be initialized as ||W||_2 per output dimension
W_norm = dora.W.norm(dim=1)
assert torch.allclose(dora.m, W_norm, atol=1e-5), 'm should be ||W||_2'
""",
        },
        {
            "name": "B initialized to zero",
            "code": """
import torch
dora = {fn}(in_features=32, out_features=64, rank=4)
assert torch.allclose(dora.B, torch.zeros_like(dora.B)), 'B should be zero-initialized'
""",
        },
        {
            "name": "DoRA applies weight decomposition",
            "code": """
import torch
torch.manual_seed(0)
dora = {fn}(in_features=16, out_features=32, rank=2)
x = torch.randn(1, 4, 16)
# Manually compute DoRA output
V = dora.W
W_eff = dora.m.view(-1, 1) * V / V.norm(dim=1, keepdim=True)
base_out = x @ W_eff.T
out = dora(x)
assert torch.allclose(out, base_out, atol=1e-4), 'DoRA output should match decomposed W'
""",
        },
        {
            "name": "Numerical match with non-zero low-rank update",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(23)
# Non-square weights, non-default scaling and independently learned magnitudes
# distinguish DoRA from a base-only layer or a direction with the wrong norm.
for in_features, out_features, rank, alpha in ((7, 11, 2, 3.5), (9, 5, 3, 0.75)):
    dora = {fn}(in_features=in_features, out_features=out_features, rank=rank, alpha=alpha)
    with torch.no_grad():
        dora.W.copy_(torch.randn_like(dora.W) * 0.3)
        dora.A.copy_(torch.randn_like(dora.A) * 0.4)
        dora.B.copy_(torch.randn_like(dora.B) * 0.5)
        dora.m.copy_(torch.linspace(0.6, 1.8, out_features))
    direction = dora.W + (alpha / rank) * (dora.B @ dora.A)
    expected_weight = dora.m[:, None] * F.normalize(direction, p=2, dim=1)
    for shape in ((4, in_features), (2, 3, in_features)):
        x = torch.randn(*shape)
        expected = F.linear(x, expected_weight)
        out = dora(x)
        assert out.shape == expected.shape, 'DoRA output shape mismatch'
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), 'DoRA must normalize W + (alpha/rank)*BA per output, then apply magnitude'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
dora = {fn}(in_features=16, out_features=32, rank=4)
x = torch.randn(1, 4, 16, requires_grad=True)
dora(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
assert dora.m.grad is not None, 'm.grad is None'
assert dora.A.grad is not None, 'A.grad is None'
assert dora.B.grad is not None, 'B.grad is None'
""",
        },
        {
            "name": "DoRA differs from LoRA",
            "code": """
import torch
torch.manual_seed(42)
# DoRA with non-zero B should differ from LoRA due to normalization
dora = {fn}(in_features=16, out_features=32, rank=4)
with torch.no_grad():
    dora.B.fill_(0.1)
x = torch.randn(1, 4, 16)
# Compute LoRA-style output
lora_out = x @ dora.W.T + x @ dora.A.T @ dora.B.T * (dora.alpha / dora.rank)
dora_out = dora(x)
assert not torch.allclose(lora_out, dora_out, atol=1e-3), 'DoRA should differ from LoRA due to normalization'
""",
        },
    ],
}
