"""Softmax implementation task."""

TASK = {
    "category": "基础层",
    "title": "Implement Softmax",
    "difficulty": "Easy",
    "function_name": "my_softmax",
    "hint": "softmax(x)_i = exp(x_i) / sum(exp(x_j)). Subtract max(x) first for numerical stability.",
    "tests": [
        {
            "name": "Basic 1-D",
            "code": """
import torch
x = torch.tensor([1.0, 2.0, 3.0])
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), f'{out} vs {expected}'
""",
        },
        {
            "name": "2-D along dim=-1",
            "code": """
import torch
x = torch.randn(4, 8)
out = {fn}(x, dim=-1)
expected = torch.softmax(x, dim=-1)
assert out.shape == expected.shape, f'Shape mismatch'
assert torch.allclose(out, expected, atol=1e-5), 'Values differ'
assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5), 'Rows must sum to 1'
""",
        },
        {
            "name": "Numerical stability",
            "code": """
import torch
x = torch.tensor([1000., 1001., 1002.])
out = {fn}(x, dim=-1)
assert not torch.isnan(out).any(), 'NaN in output — not numerically stable'
assert not torch.isinf(out).any(), 'Inf in output — not numerically stable'
expected = torch.softmax(x, dim=-1)
assert torch.allclose(out, expected, atol=1e-5), 'Values differ on large input'
""",
        },
        {
            "name": 'Non-last dimension',
            "code": """
import torch
x = torch.randn(2, 3, 5)
for dim in (0, 1, -2):
    out = {fn}(x, dim=dim)
    assert torch.allclose(out, x.softmax(dim), atol=1e-6), 'Softmax must honor dim'
""",
        },
    ],
}
