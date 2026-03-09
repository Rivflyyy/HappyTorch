"""RoPE (Rotary Position Embedding) task — positional encoding used in LLaMA, Mistral, GPT-NeoX."""

TASK = {
    "title": "Rotary Position Embedding (RoPE)",
    "difficulty": "Hard",
    "function_name": "apply_rotary_pos_emb",
    "hint": "RoPE applies rotation to pairs of dimensions. For position m and dimension pair (i, i+1): x_rotated = [x_i*cos(θ) - x_{i+1}*sin(θ), x_i*sin(θ) + x_{i+1}*cos(θ)]. θ = m * θ_i where θ_i = 1/(10000^(2i/d)).",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
x = torch.randn(2, 4, 8, 16)  # (batch, heads, seq, head_dim)
pos = torch.arange(8).unsqueeze(0).expand(2, -1)  # (batch, seq)
out = {fn}(x, pos)
assert out.shape == x.shape, f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Preserves norm",
            "code": """
import torch
x = torch.randn(1, 2, 4, 16)
pos = torch.arange(4).unsqueeze(0)
out = {fn}(x, pos)
x_norm = x.norm(dim=-1)
out_norm = out.norm(dim=-1)
assert torch.allclose(x_norm, out_norm, atol=1e-5), 'RoPE should preserve vector norm'
""",
        },
        {
            "name": "Position zero gives identity",
            "code": """
import torch
x = torch.randn(1, 2, 4, 16)
pos = torch.zeros(1, 4, dtype=torch.long)
out = {fn}(x, pos)
assert torch.allclose(out, x, atol=1e-6), 'Position 0 should give identity (cos(0)=1, sin(0)=0)'
""",
        },
        {
            "name": "Different positions give different outputs",
            "code": """
import torch
x = torch.randn(1, 2, 1, 16)
pos1 = torch.zeros(1, 1, dtype=torch.long)
pos2 = torch.ones(1, 1, dtype=torch.long)
out1 = {fn}(x, pos1)
out2 = {fn}(x, pos2)
assert not torch.allclose(out1, out2, atol=1e-3), 'Different positions should give different outputs'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
x = torch.randn(1, 2, 4, 16, requires_grad=True)
pos = torch.arange(4).unsqueeze(0)
out = {fn}(x, pos)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
""",
        },
        {
            "name": "Handles even dimensions correctly",
            "code": """
import torch, math
x = torch.randn(1, 1, 1, 8)  # head_dim = 8
pos = torch.tensor([[1]])
out = {fn}(x, pos)
# Verify rotation is applied to pairs
# For pos=1, dim pair (0,1) should be rotated by θ_0 = 1/10000^0 = 1
# Manual check: cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
theta_0 = 1.0
expected_0 = x[0,0,0,0] * math.cos(theta_0) - x[0,0,0,1] * math.sin(theta_0)
expected_1 = x[0,0,0,0] * math.sin(theta_0) + x[0,0,0,1] * math.cos(theta_0)
assert torch.allclose(out[0,0,0,0], expected_0, atol=1e-4), f'Dim 0 mismatch'
assert torch.allclose(out[0,0,0,1], expected_1, atol=1e-4), f'Dim 1 mismatch'
""",
        },
    ],
}
