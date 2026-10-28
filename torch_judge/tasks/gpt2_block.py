"""GPT-2 Transformer Block task."""

TASK = {
    "category": "完整架构",
    "title": "GPT-2 Transformer Block",
    "difficulty": "Hard",
    "function_name": "GPT2Block",
    "hint": "Pre-norm: x = x + attn(ln1(x)), x = x + mlp(ln2(x)). MLP: Linear(d, 4d) -> GELU -> Linear(4d, d). Attention must be causal. Inherit from nn.Module.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
block = {fn}(d_model=64, num_heads=4)
assert isinstance(block, nn.Module), 'GPT2Block should inherit from nn.Module'
out = block(torch.randn(2, 8, 64))
assert out.shape == (2, 8, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has LayerNorm (pre-norm architecture)",
            "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'ln1') and isinstance(block.ln1, nn.LayerNorm), 'Need self.ln1 = nn.LayerNorm'
assert hasattr(block, 'ln2') and isinstance(block.ln2, nn.LayerNorm), 'Need self.ln2 = nn.LayerNorm'
""",
        },
        {
            "name": "MLP has 4x expansion with GELU",
            "code": """
import torch, torch.nn as nn
block = {fn}(d_model=32, num_heads=4)
assert hasattr(block, 'mlp'), 'Need self.mlp'
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
assert len(linears) >= 2, f'MLP needs >= 2 Linear layers, got {len(linears)}'
assert linears[0].weight.shape == (128, 32), f'MLP first layer: {linears[0].weight.shape}, expected (128, 32)'
assert linears[-1].weight.shape == (32, 128), f'MLP last layer: {linears[-1].weight.shape}, expected (32, 128)'
""",
        },
        {
            "name": "Causal masking — future doesn't affect past",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 8, 32)
out1 = block(x)
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 4, 32)
out2 = block(x2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Future tokens affected past — not causal'
""",
        },
        {
            "name": "Numerical match to pre-norm causal block",
            "code": """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(17)
D, H = 24, 3
d_k = D // H
block = {fn}(d_model=D, num_heads=H)
block.eval()
linears = [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]
for S in (1, 5):
    x = torch.randn(2, S, D) * 1.7 + 0.4
    z = F.layer_norm(x, (D,), block.ln1.weight, block.ln1.bias, block.ln1.eps)
    q = F.linear(z, block.W_q.weight, block.W_q.bias)
    k = F.linear(z, block.W_k.weight, block.W_k.bias)
    v = F.linear(z, block.W_v.weight, block.W_v.bias)
    mask = torch.ones(S, S, dtype=torch.bool).triu(1)
    heads = []
    for h in range(H):
        head_slice = slice(h * d_k, (h + 1) * d_k)
        scores = q[..., head_slice] @ k[..., head_slice].transpose(-2, -1)
        scores = scores / math.sqrt(d_k)
        heads.append(scores.masked_fill(mask, float('-inf')).softmax(-1) @ v[..., head_slice])
    attn = F.linear(torch.cat(heads, dim=-1), block.W_o.weight, block.W_o.bias)
    residual = x + attn
    z = F.layer_norm(residual, (D,), block.ln2.weight, block.ln2.bias, block.ln2.eps)
    hidden = F.gelu(F.linear(z, linears[0].weight, linears[0].bias))
    expected = residual + F.linear(hidden, linears[-1].weight, linears[-1].bias)
    out = block(x)
    assert out.shape == expected.shape, 'Block output shape mismatch'
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), 'Values differ from pre-norm causal attention + GELU MLP with both residuals'
""",
        },
        {
            "name": "Gradient flow to all parameters",
            "code": """
import torch
torch.manual_seed(0)
block = {fn}(d_model=32, num_heads=4)
x = torch.randn(1, 4, 32, requires_grad=True)
block(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
n_total = sum(1 for p in block.parameters())
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
    ],
}
