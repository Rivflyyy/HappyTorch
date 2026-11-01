"""Top-p (nucleus) sampling task."""

TASK = {
    "category": "ML 基础与解码策略",
    "title": "Top-p Sampling",
    "difficulty": "Hard",
    "function_name": "top_p_sample",
    "hint": (
        "Sort tokens by probability descending, take the smallest prefix whose cumulative "
        "probability reaches p, mask the rest, renormalize, and sample from that nucleus."
    ),
    "tests": [
        {
            "name": "Samples only from nucleus set",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.tensor([[4.0, 3.5, 1.0, -2.0]], dtype=torch.float32)
out = {fn}(logits, p=0.8)
assert out.item() in (0, 1), f'Sampled token must come from the nucleus set, got {out.item()}'
""",
        },
        {
            "name": "Samples the specified distribution",
            "code": """
import torch
torch.manual_seed(42)
base = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]]).log()
n = 8192
# Alternate distinct distributions to catch accidental sharing between rows.
logits = base.repeat(n, 1)
for p in (0.75, 1.0):
    out = {fn}(logits, p=p)
    assert out.shape == (2*n,), 'Return one token index per row'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 3)).all(), 'Token index outside vocabulary'
    expected = torch.tensor([[2/3, 1/3, 0.0], [0.0, 1/3, 2/3]]) if p < 1 else base.softmax(-1)
    for row in range(2):
        counts = torch.bincount(out[row::2].long(), minlength=3)
        assert (counts[expected[row] == 0] == 0).all(), 'Sampled a masked token'
        frequencies = counts.float() / n
        # More than seven binomial standard errors at the worst-case variance.
        assert torch.allclose(frequencies, expected[row], atol=0.04, rtol=0), 'Sample frequencies do not match the specified distribution'
""",
        },
        {
            "name": "Small p still keeps at least one token",
            "code": """
import torch
logits = torch.tensor([1.0, 5.0, 0.5], dtype=torch.float32)
torch.manual_seed(0)
out = {fn}(logits, p=0.01)
assert out.item() == 1, f'At least the top token must remain available, got {out.item()}'
""",
        },
        {
            "name": 'Stops at an exact cumulative boundary',
            "code": """
import torch
torch.manual_seed(0)
logits = torch.tensor([2., 1., 1.]).log().expand(512, -1)
out = {fn}(logits, p=0.5)
assert out.shape == (512,) and (out == 0).all(), 'The first token already reaches p; exclude subsequent tokens'
""",
        },
        {
            "name": 'Scalar and batched token shapes',
            "code": """
import torch
for shape in ((4,), (3, 4)):
    out = {fn}(torch.randn(*shape), p=0.8)
    assert out.shape == shape[:-1], 'Remove only the vocabulary dimension'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 4)).all(), 'Token index outside vocabulary'
""",
        },
    ],
}
