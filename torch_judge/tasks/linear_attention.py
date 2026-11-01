"""Linear Self-Attention task."""

TASK = {
    "category": "注意力机制",
    "title": "Linear Self-Attention",
    "difficulty": "Hard",
    "function_name": "linear_attention",
    "hint": "Feature map: phi(x) = elu(x) + 1. Compute phi(Q) @ (phi(K)^T @ V) instead of softmax(Q @ K^T) @ V. Normalize by phi(Q) @ sum(phi(K)).",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 32))
assert out.shape == (2, 8, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "No NaN or Inf",
            "code": """
import torch
torch.manual_seed(0)
out = {fn}(torch.randn(2, 16, 8), torch.randn(2, 16, 8), torch.randn(2, 16, 8))
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(1, 4, 8, requires_grad=True)
K = torch.randn(1, 4, 8, requires_grad=True)
V = torch.randn(1, 4, 8, requires_grad=True)
{fn}(Q, K, V).sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": 'Numerical normalized feature-map attention',
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(17)
Q, K, V = torch.randn(2, 7, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 3)
# A small quadratic oracle is independent of the efficient associativity trick.
weights = (F.elu(Q) + 1) @ (F.elu(K) + 1).transpose(-1, -2)
expected = (weights @ V) / (weights.sum(-1, keepdim=True) + 1e-6)
out = {fn}(Q, K, V)
assert out.shape == expected.shape and torch.allclose(out, expected, atol=1e-5), 'Wrong normalized feature-map attention'
""",
        },
    ],
}
