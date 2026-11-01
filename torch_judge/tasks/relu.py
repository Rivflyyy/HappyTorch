"""ReLU implementation task."""

TASK = {
    "category": "基础层",
    "title": "Implement ReLU",
    "difficulty": "Easy",
    "function_name": "relu",
    "hint": "ReLU(x) = max(0, x). Think about element-wise comparison with zero.",
    "tests": [
        {
            "name": "Basic values",
            "code": """
import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
expected = torch.tensor([0., 0., 0., 1., 2.])
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected), f'Wrong Answer: {out} vs {expected}'
""",
        },
        {
            "name": "2-D tensor",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x)
assert out.shape == x.shape, f'Shape mismatch on 2-D input'
assert (out >= 0).all(), 'ReLU output must be non-negative'
assert torch.allclose(out, x.clamp(min=0)), 'Value mismatch on random input'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import torch
x = torch.tensor([-1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'Gradient not computed'
assert x.grad[0] == 0., f'grad at x=-1 should be 0, got {x.grad[0]}'
assert x.grad[2] == 1., f'grad at x=1 should be 1, got {x.grad[2]}'
assert x.grad[3] == 1., f'grad at x=2 should be 1, got {x.grad[3]}'
assert torch.isfinite(x.grad[1]) and 0 <= x.grad[1] <= 1, f'At zero, any ReLU subgradient in [0, 1] is valid; got {x.grad[1]}'
""",
        },
    ],
}
