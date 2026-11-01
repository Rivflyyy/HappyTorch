"""Top-k sampling task."""

TASK = {
    "category": "ML 基础与解码策略",
    "title": "Top-k Sampling",
    "difficulty": "Medium",
    "function_name": "top_k_sample",
    "hint": (
        "Find the top-k logits, mask all other logits to -inf, apply softmax over the "
        "remaining tokens, then sample from that truncated distribution."
    ),
    "tests": [
        {
            "name": "Samples only from top-k tokens",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.tensor([[5.0, 4.0, 1.0, -2.0]])
out = {fn}(logits, k=2)
assert out.item() in (0, 1), f'Sampled token must be in the top-2 set, got {out.item()}'
""",
        },
        {
            "name": "Samples the specified distribution",
            "code": """
import torch
torch.manual_seed(42)
base = torch.tensor([[0.6, 0.2, 0.15], [0.15, 0.2, 0.6]]).log()
n = 8192
# Alternate distinct distributions to catch accidental sharing between rows.
logits = base.repeat(n, 1)
for k in (2, 9):
    out = {fn}(logits, k=k)
    assert out.shape == (2*n,), 'Return one token index per row'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 3)).all(), 'Token index outside vocabulary'
    expected = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.25, 0.75]]) if k == 2 else base.softmax(-1)
    for row in range(2):
        counts = torch.bincount(out[row::2].long(), minlength=3)
        assert (counts[expected[row] == 0] == 0).all(), 'Sampled a masked token'
        frequencies = counts.float() / n
        # More than seven binomial standard errors at the worst-case variance.
        assert torch.allclose(frequencies, expected[row], atol=0.04, rtol=0), 'Sample frequencies do not match the specified distribution'
""",
        },
        {
            "name": "k=1 equals greedy",
            "code": """
import torch
logits = torch.tensor([[0.2, 1.7, 0.8], [3.0, 2.0, 1.0]])
out = {fn}(logits, k=1)
expected = logits.argmax(dim=-1)
assert torch.equal(out, expected), f'k=1 should be greedy: {out} vs {expected}'
""",
        },
        {
            "name": 'Scalar and batched token shapes',
            "code": """
import torch
for shape in ((4,), (3, 4)):
    out = {fn}(torch.randn(*shape), k=2)
    assert out.shape == shape[:-1], 'Remove only the vocabulary dimension'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 4)).all(), 'Token index outside vocabulary'
""",
        },
    ],
}
