"""AdaLN-Zero task — zero-initialized AdaLN for stable Diffusion Transformer training."""

TASK = {
    "category": "条件调制 — Diffusion",
    "title": "AdaLN-Zero",
    "difficulty": "Hard",
    "function_name": "AdaLNZero",
    "hint": 'This simplified residual branch returns gate(cond) * ((1 + scale(cond)) * LayerNorm(x) + shift(cond)). Zero-initialize scale, shift, and gate projections. Then x + branch(x, cond) starts as identity, while the gate can receive gradients. Expose gate_proj as nn.Linear(cond_dim, hidden_dim).',
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
adaln = {fn}(hidden_dim=64, cond_dim=128)
x = torch.randn(2, 10, 64)
cond = torch.randn(2, 128)
out = adaln(x, cond)
assert out.shape == (2, 10, 64), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Zero initialization",
            "code": """
import torch
adaln = {fn}(hidden_dim=32, cond_dim=64)
# The final projection should be zero-initialized
# This ensures γ=0, β=0, gate=0 at initialization
x = torch.randn(1, 4, 32)
cond = torch.randn(1, 64)
out = adaln(x, cond)
# With zero init, output should be close to zero
assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), f'Zero init should give zero output, got max {out.abs().max()}'
""",
        },
        {
            "name": "Has gate parameter",
            "code": """
import torch.nn as nn
layer = {fn}(hidden_dim=32, cond_dim=64)
assert isinstance(layer.gate_proj, nn.Linear), 'Public gate_proj must be nn.Linear'
assert layer.gate_proj.weight.shape == (32, 64), 'gate_proj maps cond_dim to hidden_dim'
""",
        },
        {
            "name": "Gate controls the residual branch",
            "code": """
import torch
layer = {fn}(hidden_dim=32, cond_dim=64)
x, cond = torch.randn(2, 5, 32), torch.randn(2, 64)
# Opening only the gate exposes the initial identity modulation of LayerNorm.
with torch.no_grad():
    layer.gate_proj.weight.zero_()
    layer.gate_proj.bias.fill_(1)
expected = torch.nn.functional.layer_norm(x, (32,), eps=1e-5)
assert torch.allclose(layer(x, cond), expected, atol=1e-5), 'An open gate at initialization must expose LayerNorm(x)'
# Closing the gate must also suppress nonzero modulation after training.
with torch.no_grad():
    for p in layer.parameters():
        if p.requires_grad:
            p.normal_(0, 0.2)
    layer.gate_proj.weight.zero_()
    layer.gate_proj.bias.zero_()
assert torch.allclose(layer(x, cond), torch.zeros_like(x), atol=1e-5), 'Zero gate must suppress the branch'
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
""",
        },
        {
            "name": "Non-zero after training simulation",
            "code": """
import torch
import torch.optim as optim
torch.manual_seed(20)
layer = {fn}(hidden_dim=32, cond_dim=64)
optimizer = optim.SGD(layer.parameters(), lr=0.05)
x, cond = torch.randn(2, 5, 32), torch.randn(2, 64)
target = torch.nn.functional.layer_norm(x, (32,))
initial_loss = (layer(x, cond) - target).square().mean().item()
for step in range(5):
    optimizer.zero_grad()
    loss = (layer(x, cond) - target).square().mean()
    loss.backward()
    if step == 0:
        grad = layer.gate_proj.bias.grad
        assert grad is not None and grad.abs().sum() > 0, 'Zero initialization must allow the gate to learn'
    optimizer.step()
final_loss = (layer(x, cond) - target).square().mean()
assert torch.isfinite(final_loss) and final_loss < initial_loss, 'The residual branch must learn from zero initialization'
""",
        },
        {
            "name": 'Nonzero per-sample modulation',
            "code": """
import torch
torch.manual_seed(27)
layer = {fn}(hidden_dim=32, cond_dim=64)
with torch.no_grad():
    for p in layer.parameters():
        if p.requires_grad:
            p.normal_(0, 0.2)
    # Open the public gate independently to exercise learned scale and shift.
    layer.gate_proj.weight.zero_()
    layer.gate_proj.bias.fill_(1)
x, cond = torch.randn(2, 5, 32) * 2 + 3, torch.randn(2, 64)
shift = layer(torch.zeros_like(x), cond)
probe = torch.tensor([-1., 1.]).repeat(16).expand_as(x)
normalize = lambda value: torch.nn.functional.layer_norm(value, (32,), eps=1e-5)
scale = (layer(probe, cond) - shift) / normalize(probe)
out = layer(x, cond)
assert shift.abs().max() > 0, 'Use the learned shift when the gate is open'
assert (scale - 1).abs().max() > 0, 'Use the learned scale offset when the gate is open'
assert torch.allclose(out, scale * normalize(x) + shift, atol=2e-5, rtol=1e-4), 'Modulate normalized features before gating'
for b in range(2):
    assert torch.allclose(out[b:b+1], layer(x[b:b+1], cond[b:b+1]), atol=2e-5), 'Each sample must use its own condition'
for t in range(x.shape[1]):
    assert torch.allclose(out[:, t:t+1], layer(x[:, t:t+1], cond), atol=2e-5), 'Modulation must be independent across tokens'
""",
        },
    ],
}
