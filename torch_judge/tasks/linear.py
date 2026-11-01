"""Simple Linear Layer task."""

TASK = {
    "category": "基础层",
    "title": "Simple Linear Layer",
    "difficulty": "Medium",
    "function_name": "SimpleLinear",
    "hint": "y = x @ W^T + b. Initialize weight with Kaiming scaling: randn * (1/sqrt(in_features)).",
    "tests": [
        {
            "name": "Weight & bias shape",
            "code": """
import torch
layer = {fn}(8, 4)
assert layer.weight.shape == (4, 8), f'Weight shape: {layer.weight.shape}'
assert layer.bias.shape == (4,), f'Bias shape: {layer.bias.shape}'
assert layer.weight.requires_grad, 'weight must require grad'
assert layer.bias.requires_grad, 'bias must require grad'
""",
        },
        {
            "name": "Forward pass",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
assert y.shape == (2, 4), f'Output shape: {y.shape}'
expected = x @ layer.weight.T + layer.bias
assert torch.allclose(y, expected, atol=1e-5), 'Forward != x @ W^T + b'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
layer = {fn}(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
dw, db = torch.autograd.grad(y.sum(), (layer.weight, layer.bias))
assert torch.allclose(dw, x.sum(0).expand_as(dw)), 'Incorrect weight gradient'
assert torch.allclose(db, torch.full_like(db, x.shape[0])), 'Incorrect bias gradient'
""",
        },
    ],
}
