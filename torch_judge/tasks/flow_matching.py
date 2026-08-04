"""Flow Matching / Rectified Flow training objective."""

TASK = {
    "category": "扩散模型训练",
    "title": "Flow Matching Loss (Rectified Flow)",
    "difficulty": "Medium",
    "function_name": "flow_matching_loss",
    "hint": (
        "Straight-line probability path: x_t = (1 - t) * x0 + t * x1, so the target velocity is "
        "the constant u = x1 - x0 (independent of t!). Loss = mean((model(x_t, t) - u) ** 2). "
        "t has shape (B,) but x has shape (B, ...) — reshape t to (B, 1, 1, ...) so it broadcasts "
        "over every feature dimension, otherwise you silently mix samples."
    ),
    "tests": [
        {
            "name": "Scalar loss, zero-velocity model",
            "code": """
import torch
torch.manual_seed(0)
x0 = torch.randn(8, 4)
x1 = torch.randn(8, 4)
t = torch.rand(8)
loss = {fn}(lambda x, tt: torch.zeros_like(x), x0, x1, t)
assert isinstance(loss, torch.Tensor), f'Loss must be a tensor, got {type(loss)}'
assert loss.dim() == 0, f'Loss must be a scalar, got shape {loss.shape}'
expected = ((x1 - x0) ** 2).mean()
assert torch.allclose(loss, expected, atol=1e-6), f'Expected {expected.item()}, got {loss.item()}'
""",
        },
        {
            "name": "Perfect velocity model gives zero loss",
            "code": """
import torch
torch.manual_seed(1)
x0 = torch.randn(6, 3, 8, 8)
x1 = torch.randn(6, 3, 8, 8)
t = torch.rand(6)
loss = {fn}(lambda x, tt: x1 - x0, x0, x1, t)
assert loss.item() < 1e-10, f'A model that outputs the exact velocity must give ~0 loss, got {loss.item()}'
""",
        },
        {
            "name": "Interpolation path x_t = (1-t)*x0 + t*x1",
            "code": """
import torch
torch.manual_seed(2)
seen = {}
def model(x, tt):
    seen['x_t'] = x
    return torch.zeros_like(x)
x0 = torch.randn(3, 2, 4, 4)
x1 = torch.randn(3, 2, 4, 4)
t = torch.tensor([0.0, 0.5, 1.0])
{fn}(model, x0, x1, t)
assert 'x_t' in seen, 'The model was never called — you must evaluate model(x_t, t)'
x_t = seen['x_t']
assert x_t.shape == x0.shape, f'x_t shape {x_t.shape} should match x0 shape {x0.shape}'
assert torch.allclose(x_t[0], x0[0], atol=1e-6), 't=0 must give pure noise x0'
assert torch.allclose(x_t[2], x1[2], atol=1e-6), 't=1 must give pure data x1'
assert torch.allclose(x_t[1], 0.5 * (x0[1] + x1[1]), atol=1e-6), 't=0.5 must give the midpoint'
""",
        },
        {
            "name": "Per-sample t broadcasting",
            "code": """
import torch
torch.manual_seed(3)
seen = {}
def model(x, tt):
    seen['x_t'] = x
    return torch.zeros_like(x)
x0 = torch.zeros(4, 5)
x1 = torch.ones(4, 5)
t = torch.tensor([0.0, 0.25, 0.75, 1.0])
{fn}(model, x0, x1, t)
x_t = seen['x_t']
for i, ti in enumerate([0.0, 0.25, 0.75, 1.0]):
    assert torch.allclose(x_t[i], torch.full((5,), ti), atol=1e-6), \\
        f'Sample {i} should be all {ti}, got {x_t[i]} — did you broadcast t per-sample?'
""",
        },
        {
            "name": "Velocity target is constant in t",
            "code": """
import torch
torch.manual_seed(4)
x0 = torch.randn(5, 7)
x1 = torch.randn(5, 7)
const = lambda x, tt: torch.full_like(x, 0.3)
loss_a = {fn}(const, x0, x1, torch.zeros(5))
loss_b = {fn}(const, x0, x1, torch.ones(5))
assert torch.allclose(loss_a, loss_b, atol=1e-6), \\
    'The regression target x1 - x0 does not depend on t, so a t-independent model must give the same loss'
expected = ((0.3 - (x1 - x0)) ** 2).mean()
assert torch.allclose(loss_a, expected, atol=1e-6), f'Expected {expected.item()}, got {loss_a.item()}'
""",
        },
        {
            "name": "Gradient flows into the model",
            "code": """
import torch
import torch.nn as nn
torch.manual_seed(5)
net = nn.Linear(4, 4)
x0 = torch.randn(8, 4)
x1 = torch.randn(8, 4)
t = torch.rand(8)
loss = {fn}(lambda x, tt: net(x), x0, x1, t)
loss.backward()
assert net.weight.grad is not None, 'No gradient reached the model parameters'
assert net.weight.grad.abs().sum() > 0, 'Model gradient is all zeros — did you detach something?'
""",
        },
    ],
}
