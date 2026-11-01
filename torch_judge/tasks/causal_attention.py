"""Causal Self-Attention task."""

TASK = {
    "category": "注意力机制",
    "title": "Causal Self-Attention",
    "difficulty": "Hard",
    "function_name": "causal_attention",
    "hint": "Same as softmax attention but mask future positions with -inf before softmax. torch.triu(..., diagonal=1) gives the upper triangle.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 6, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16))
assert out.shape == (2, 6, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Future tokens don't affect past",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 8, 16
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V)
K2, V2 = K.clone(), V.clone()
K2[:, 4:] = torch.randn(B, 4, D)
V2[:, 4:] = torch.randn(B, 4, D)
out2 = {fn}(Q, K2, V2)
assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5), 'Changing future K/V affected past outputs'
""",
        },
        {
            "name": "First position only sees itself",
            "code": """
import torch
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V)
assert torch.allclose(out[:, 0], V[:, 0], atol=1e-5), 'Position 0 should output V[0]'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
out = {fn}(Q, K, V)
out.sum().backward()
assert Q.grad is not None and K.grad is not None and V.grad is not None, 'Missing gradients'
""",
        },
        {
            "name": 'Numerical masked attention with distinct value width',
            "code": """
import torch, math
torch.manual_seed(17)
Q, K, V = torch.randn(2, 7, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 3)
scores = Q @ K.transpose(-1, -2) / math.sqrt(4)
scores = scores.masked_fill(torch.arange(7)[None, :] > torch.arange(7)[:, None], float('-inf'))
expected = scores.softmax(-1) @ V
out = {fn}(Q, K, V)
assert out.shape == expected.shape and torch.allclose(out, expected, atol=1e-5), 'Wrong masked attention values'
""",
        },
    ],
}
