"""GELU (Gaussian Error Linear Unit) task — widely used in BERT, GPT, Diffusion models."""

TASK = {
    "category": "现代激活函数",
    "title": "GELU Activation",
    "difficulty": "Medium",
    "function_name": "gelu",
    "hint": 'Implement exact GELU: x * Phi(x), with Phi(x) = 0.5 * (1 + erf(x / sqrt(2))). The tanh approximation is a different function and is not the target of this exercise.',
    "tests": [
        {
            "name": "Basic values",
            "code": """
import torch, math
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
# GELU(0) = 0, GELU(x) > 0 for x > 0, GELU(x) < 0 but small for x < 0
assert abs(out[2].item()) < 1e-6, f'GELU(0) should be ~0, got {out[2]}'
assert out[3] > 0.8, f'GELU(1) should be ~0.841, got {out[3]}'
assert out[0] < 0, f'GELU(-2) should be negative, got {out[0]}'
assert out[0] > -0.05, f'GELU(-2) should be small negative, got {out[0]}'
""",
        },
        {
            "name": "Numerical accuracy vs reference",
            "code": """
import torch, math
x = torch.linspace(-4, 4, 100)
out = {fn}(x)
ref = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
assert torch.allclose(out, ref, atol=1e-5), f'Max diff: {(out - ref).abs().max()}'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import torch, math
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'No gradient'
# GELU'(x) = Φ(x) + x * φ(x) where φ is Gaussian PDF
# At x=1: ~0.841 + 1 * 0.242 = ~1.083
assert x.grad[2] > 0.5, f'GELU\\'(1) should be > 0.5, got {x.grad[2]}'
assert x.grad[2] < 1.5, f'GELU\\'(1) should be < 1.5, got {x.grad[2]}'
""",
        },
        {
            "name": "2-D tensor",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch'
assert torch.allclose(out, torch.nn.functional.gelu(x), atol=1e-4), 'Should match nn.functional.gelu'
""",
        },
        {
            "name": "Smooth transition at zero",
            "code": """
import torch
x = torch.linspace(-0.1, 0.1, 20)
out = {fn}(x)
# GELU should be smooth (no sharp edges like ReLU)
diffs = out[1:] - out[:-1]
assert (diffs >= 0).all(), 'GELU should be increasing near zero'
""",
        },
    ],
}
