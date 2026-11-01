"""Sliding Window Attention task."""

TASK = {
    "category": "注意力机制",
    "title": "Sliding Window Attention",
    "difficulty": "Hard",
    "function_name": "sliding_window_attention",
    "hint": "Like softmax attention but position i only attends to positions j where |i-j| <= window_size. Mask the rest with -inf.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
out = {fn}(torch.randn(2, 8, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16), window_size=2)
assert out.shape == (2, 8, 16), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "window_size=0 — only sees itself",
            "code": """
import torch
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = {fn}(Q, K, V, window_size=0)
assert torch.allclose(out, V, atol=1e-5), 'window=0: each position should output V[i]'
""",
        },
        {
            "name": "Large window equals full attention",
            "code": """
import torch, math
torch.manual_seed(0)
B, S, D = 2, 6, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out_win = {fn}(Q, K, V, window_size=S)
d_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
assert torch.allclose(out_win, ref, atol=1e-5), 'Large window should equal full attention'
""",
        },
        {
            "name": "Distant tokens don't affect output",
            "code": """
import torch
torch.manual_seed(0)
B, S, D = 1, 10, 8
Q = torch.randn(B, S, D)
K = torch.randn(B, S, D)
V = torch.randn(B, S, D)
out1 = {fn}(Q, K, V, window_size=1)
K2, V2 = K.clone(), V.clone()
K2[:, 5:] = torch.randn(B, 5, D)
V2[:, 5:] = torch.randn(B, 5, D)
out2 = {fn}(Q, K2, V2, window_size=1)
assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-5), 'Distant tokens should not affect output'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
Q = torch.randn(2, 4, 8, requires_grad=True)
K = torch.randn(2, 4, 8, requires_grad=True)
V = torch.randn(2, 4, 8, requires_grad=True)
{fn}(Q, K, V, window_size=1).sum().backward()
assert Q.grad is not None, 'Q.grad is None'
""",
        },
        {
            "name": 'Numerical masked attention with distinct value width',
            "code": """
import torch, math
torch.manual_seed(17)
Q, K, V = torch.randn(2, 7, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 3)
scores = Q @ K.transpose(-1, -2) / math.sqrt(4)
scores = scores.masked_fill((torch.arange(7)[None, :] - torch.arange(7)[:, None]).abs() > 2, float('-inf'))
expected = scores.softmax(-1) @ V
out = {fn}(Q, K, V, window_size=2)
assert out.shape == expected.shape and torch.allclose(out, expected, atol=1e-5), 'Wrong masked attention values'
""",
        },
    ],
}
