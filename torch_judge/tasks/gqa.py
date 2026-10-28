"""Grouped Query Attention task."""

TASK = {
    "category": "注意力机制",
    "title": "Grouped Query Attention",
    "difficulty": "Hard",
    "function_name": "GroupQueryAttention",
    "hint": "Like MHA but fewer KV heads. W_k/W_v project to num_kv_heads * d_k dims. Use repeat_interleave to expand KV heads to match Q heads.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
out = gqa.forward(torch.randn(2, 6, 32))
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "nn.Linear with correct shapes",
            "code": """
import torch, torch.nn as nn
gqa = {fn}(d_model=32, num_heads=8, num_kv_heads=2)
d_k = 32 // 8
assert isinstance(gqa.W_q, nn.Linear) and gqa.W_q.weight.shape == (32, 32), f'W_q wrong'
assert isinstance(gqa.W_k, nn.Linear) and gqa.W_k.weight.shape == (2 * d_k, 32), f'W_k shape: {gqa.W_k.weight.shape}'
assert isinstance(gqa.W_v, nn.Linear) and gqa.W_v.weight.shape == (2 * d_k, 32), f'W_v shape: {gqa.W_v.weight.shape}'
assert isinstance(gqa.W_o, nn.Linear), 'W_o should be nn.Linear'
""",
        },
        {
            "name": "Degenerates to MHA when kv_heads == heads",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(42)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=4)
x = torch.randn(2, 5, 16)
out = gqa.forward(x)
assert out.shape == (2, 5, 16)
assert gqa.W_k.weight.shape == (16, 16), 'Full KV when kv_heads == heads'
# Compare values using identical projection weights, not just head dimensions.
ref = nn.MultiheadAttention(16, 4, dropout=0.0, batch_first=True)
with torch.no_grad():
    projections = (gqa.W_q, gqa.W_k, gqa.W_v)
    ref.in_proj_weight.copy_(torch.cat([p.weight for p in projections]))
    ref.in_proj_bias.copy_(torch.cat([
        p.bias if p.bias is not None else p.weight.new_zeros(16)
        for p in projections
    ]))
    ref.out_proj.weight.copy_(gqa.W_o.weight)
    ref.out_proj.bias.zero_()
    if gqa.W_o.bias is not None:
        ref.out_proj.bias.copy_(gqa.W_o.bias)
expected, _ = ref(x, x, x, need_weights=True)
assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), 'GQA must match MHA values when KV heads equal Q heads'
""",
        },
        {
            "name": "KV heads are shared correctly",
            "code": """
import torch
import torch.nn.functional as F
import math
torch.manual_seed(0)
# Use distinct head/sequence dimensions to expose incorrect reshapes. Compute
# each head separately so the reference does not repeat the learner's KV expand.
D, H = 24, 6
d_k = D // H
for KV in (1, 2, 3):
    gqa = {fn}(d_model=D, num_heads=H, num_kv_heads=KV)
    x = torch.randn(2, 5, D)
    q = F.linear(x, gqa.W_q.weight, gqa.W_q.bias)
    k = F.linear(x, gqa.W_k.weight, gqa.W_k.bias)
    v = F.linear(x, gqa.W_v.weight, gqa.W_v.bias)
    heads = []
    for h in range(H):
        kv_head = h // (H // KV)
        q_h = q[..., h * d_k:(h + 1) * d_k]
        k_h = k[..., kv_head * d_k:(kv_head + 1) * d_k]
        v_h = v[..., kv_head * d_k:(kv_head + 1) * d_k]
        weights = (q_h @ k_h.transpose(-2, -1) / math.sqrt(d_k)).softmax(dim=-1)
        heads.append(weights @ v_h)
    expected = F.linear(torch.cat(heads, dim=-1), gqa.W_o.weight, gqa.W_o.bias)
    out = gqa.forward(x)
    assert out.shape == expected.shape, f'Wrong output shape with {KV} KV heads'
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), f'Incorrect attention values or KV grouping with {KV} KV heads'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(0)
gqa = {fn}(d_model=16, num_heads=4, num_kv_heads=2)
x = torch.randn(1, 4, 16, requires_grad=True)
gqa.forward(x).sum().backward()
assert x.grad is not None, 'x.grad is None'
assert gqa.W_q.weight.grad is not None and gqa.W_k.weight.grad is not None, 'Missing weight gradients'
""",
        },
    ],
}
