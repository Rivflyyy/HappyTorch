"""LoRA (Low-Rank Adaptation) task — parameter-efficient fine-tuning."""

TASK = {
    "category": "参数高效微调",
    "title": "LoRA Linear Layer",
    "difficulty": "Hard",
    "function_name": "LoRALinear",
    "hint": "LoRA: W' = W + BA where B (d_out × r) and A (r × d_in). Alpha scales: ΔW = (α/r) * BA. Initialize A with Kaiming/normal, B with zeros. Forward: x @ W.T + (x @ A.T @ B.T) * α/r.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
lora = {fn}(in_features=64, out_features=128, rank=8)
x = torch.randn(2, 10, 64)
out = lora(x)
assert out.shape == (2, 10, 128), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has required components",
            "code": """
import torch, torch.nn as nn
lora = {fn}(in_features=32, out_features=64, rank=4)
assert hasattr(lora, 'W'), 'Need self.W (base weight)'
assert hasattr(lora, 'A'), 'Need self.A (down projection)'
assert hasattr(lora, 'B'), 'Need self.B (up projection)'
assert lora.A.shape == (4, 32), f'A shape: {lora.A.shape}, expected (4, 32)'
assert lora.B.shape == (64, 4), f'B shape: {lora.B.shape}, expected (64, 4)'
""",
        },
        {
            "name": "B initialized to zero",
            "code": """
import torch
lora = {fn}(in_features=32, out_features=64, rank=4)
# B should be zero-initialized so the base layer is unchanged (W' = W)
assert torch.allclose(lora.B, torch.zeros_like(lora.B)), 'B should be zero-initialized'
""",
        },
        {
            "name": "LoRA adds to base output",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
lora = {fn}(in_features=16, out_features=32, rank=2, alpha=4.0)
x = torch.randn(1, 4, 16)
# Set A and B to known values
with torch.no_grad():
    lora.A.fill_(0.1)
    lora.B.fill_(0.2)
out = lora(x)
base = x @ lora.W.T
lora_delta = x @ lora.A.T @ lora.B.T * (lora.alpha / lora.rank)
ref = base + lora_delta
assert torch.allclose(out, ref, atol=1e-5), 'LoRA should add scaled BA to base output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
lora = {fn}(in_features=16, out_features=32, rank=4)
x = torch.randn(1, 4, 16, requires_grad=True)
out = lora(x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert lora.A.grad is not None, 'A.grad is None'
assert lora.B.grad is not None, 'B.grad is None'
""",
        },
        {
            "name": "Registered low-rank parameters",
            "code": """
import torch.nn as nn
layer = {fn}(in_features=64, out_features=128, rank=8)
assert layer.W.shape == (128, 64), 'W must use (out_features, in_features) layout'
registered = {id(p) for p in layer.parameters()}
for name in ('A', 'B'):
    p = getattr(layer, name)
    assert isinstance(p, nn.Parameter) and p.requires_grad and id(p) in registered, 'Register trainable low-rank parameters'
assert layer.A.numel() + layer.B.numel() == 8 * (64 + 128), 'Low-rank parameter count should scale with rank'
""",
        },
        {
            "name": "Zero rank leaves the base layer unchanged",
            "code": """
import torch
lora = {fn}(in_features=32, out_features=64, rank=0)
x = torch.randn(1, 4, 32)
out = lora(x)
base = x @ lora.W.T
assert torch.allclose(out, base, atol=1e-6), 'rank=0 should be pure base layer'
""",
        },
    ],
}
