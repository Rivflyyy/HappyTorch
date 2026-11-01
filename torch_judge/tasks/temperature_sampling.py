"""Temperature sampling task."""

TASK = {
    "category": "ML 基础与解码策略",
    "title": "Temperature Sampling",
    "difficulty": "Medium",
    "function_name": "temperature_sample",
    "hint": 'Divide logits by positive temperature, apply softmax, and sample from that categorical distribution. torch.multinomial is one option; other correct sampling algorithms are accepted.',
    "tests": [
        {
            "name": "Returns one token per row",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
out = {fn}(logits, temperature=0.7)
assert out.shape == (2,), f'Expected one sample per row, got {out.shape}'
assert out.dtype in (torch.int32, torch.int64), f'Output should be integer typed, got {out.dtype}'
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
for temperature in (0.5, 2.0):
    out = {fn}(logits, temperature=temperature)
    assert out.shape == (2*n,), 'Return one token index per row'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 3)).all(), 'Token index outside vocabulary'
    expected = torch.softmax(base / temperature, -1)
    for row in range(2):
        counts = torch.bincount(out[row::2].long(), minlength=3)
        assert (counts[expected[row] == 0] == 0).all(), 'Sampled a masked token'
        frequencies = counts.float() / n
        # More than seven binomial standard errors at the worst-case variance.
        assert torch.allclose(frequencies, expected[row], atol=0.04, rtol=0), 'Sample frequencies do not match the specified distribution'
""",
        },
        {
            "name": "Very low temperature behaves greedily",
            "code": """
import torch
logits = torch.tensor([0.0, 5.0, 1.0], dtype=torch.float32)
torch.manual_seed(0)
out = {fn}(logits, temperature=1e-4)
assert out.item() == 1, f'Low temperature should almost always pick argmax, got {out.item()}'
""",
        },
        {
            "name": 'Scalar and batched token shapes',
            "code": """
import torch
for shape in ((4,), (3, 4)):
    out = {fn}(torch.randn(*shape), temperature=1.0)
    assert out.shape == shape[:-1], 'Remove only the vocabulary dimension'
    assert out.dtype in (torch.int32, torch.int64), 'Return integer token indices'
    assert ((out >= 0) & (out < 4)).all(), 'Token index outside vocabulary'
""",
        },
    ],
}
