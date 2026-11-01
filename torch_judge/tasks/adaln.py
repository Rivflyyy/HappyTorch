"""AdaLN (Adaptive Layer Normalization) task — used in DiT, Diffusion Transformers."""

TASK = {
    "category": "条件调制 — Diffusion",
    "title": "AdaLN (Adaptive Layer Normalization)",
    "difficulty": "Hard",
    "function_name": "AdaLN",
    "hint": 'Predict scale and shift from cond using separate projections or a fused projection and chunk. Output: gamma(cond) * LayerNorm(x) + beta(cond), with population variance and eps=1e-5. Internal module names are your choice.',
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
adaln = {fn}(hidden_dim=64, cond_dim=128)
x = torch.randn(2, 10, 64)
cond = torch.randn(2, 128)
out = adaln(x, cond)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Produces per-sample modulation",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(21)
layer = {fn}(hidden_dim=32, cond_dim=64)
assert isinstance(layer, nn.Module), 'Implement an nn.Module with registered trainable conditioning parameters'
params = [p for p in layer.parameters() if p.requires_grad]
assert params, 'Conditioning must have trainable parameters'
with torch.no_grad():
    for p in params:
        p.normal_(0, 0.2)
x = torch.randn(2, 5, 32)
cond = torch.randn(2, 64)
changed = cond.clone()
# Use a nonuniform replacement: normalization may validly remove uniform offsets.
changed[1] = torch.randn_like(changed[1]) * 2
out = layer(x, cond)
out_changed = layer(x, changed)
assert torch.allclose(out[0], out_changed[0], atol=1e-5), "One sample must not use another sample's condition"
assert not torch.allclose(out[1], out_changed[1], atol=1e-5), 'Condition must affect its own sample'
for b in range(2):
    single = layer(x[b:b+1], cond[b:b+1])
    assert torch.allclose(out[b:b+1], single, atol=1e-5), 'Process each batch element independently'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
adaln = {fn}(hidden_dim=32, cond_dim=64)
x = torch.randn(1, 4, 32, requires_grad=True)
cond = torch.randn(1, 64, requires_grad=True)
out = adaln(x, cond)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert cond.grad is not None, 'cond.grad is None'
assert any(p.grad is not None for p in adaln.parameters() if p.requires_grad), 'Conditioning parameters must receive gradients'
""",
        },
        {
            "name": 'Featurewise affine formula and token independence',
            "code": """
import torch
torch.manual_seed(23)
layer = {fn}(hidden_dim=32, cond_dim=64)
with torch.no_grad():
    for p in layer.parameters():
        if p.requires_grad:
            p.normal_(0, 0.2)
x = torch.randn(2, 5, 32) * 2 + 3
cond = torch.randn(2, 64)
beta = layer(torch.zeros_like(x), cond)
probe = torch.tensor([-1., 1.]).repeat(16).expand_as(x)
gamma = (layer(probe, cond) - beta) / (torch.nn.functional.layer_norm(probe, (32,), eps=1e-5))
expected = gamma * (torch.nn.functional.layer_norm(x, (32,), eps=1e-5)) + beta
out = layer(x, cond)
assert torch.allclose(out, expected, atol=2e-5, rtol=1e-4), 'Apply scale and shift independently to each feature'
assert torch.allclose(beta, beta[:, :1].expand_as(beta), atol=1e-5), 'Shift must be shared across tokens'
assert torch.allclose(gamma, gamma[:, :1].expand_as(gamma), atol=1e-5), 'Scale must be shared across tokens'
for t in range(x.shape[1]):
    assert torch.allclose(out[:, t:t+1], layer(x[:, t:t+1], cond), atol=2e-5), 'Tokens must not interact in this layer'
""",
        },
    ],
}
