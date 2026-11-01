"""SiLU (Swish) activation task — core component of SwiGLU, used in many modern architectures."""

TASK = {
    "category": "现代激活函数",
    "title": "SiLU (Swish) Activation",
    "difficulty": "Easy",
    "function_name": "silu",
    "hint": "SiLU(x) = x * sigmoid(x). Also known as Swish. Non-monotonic activation that outperforms ReLU in deep networks.",
    "tests": [
        {
            "name": "Basic values",
            "code": """
import torch
x = torch.tensor([-2., -1., 0., 1., 2.])
out = {fn}(x)
# SiLU(-x) = -x * sigmoid(-x) = -x * (1 - sigmoid(x)) ≈ small negative
assert abs(out[2].item()) < 1e-6, f'SiLU(0) should be 0, got {out[2]}'
assert out[3] > 0.5, f'SiLU(1) should be ~0.731, got {out[3]}'
assert out[0] < 0, f'SiLU(-2) should be negative, got {out[0]}'
""",
        },
        {
            "name": "Matches torch.nn.functional.silu",
            "code": """
import torch, torch.nn.functional as F
x = torch.randn(4, 8)
out = {fn}(x)
ref = F.silu(x)
assert torch.allclose(out, ref, atol=1e-6), 'Should match F.silu'
""",
        },
        {
            "name": "Gradient check",
            "code": """
import torch
x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
out = {fn}(x)
out.sum().backward()
assert x.grad is not None, 'No gradient'
# SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
# At x=1: sigmoid(1) ≈ 0.731, SiLU'(1) ≈ 0.928
assert x.grad[3] > 0.5, f'SiLU\\'(1) should be > 0.5, got {x.grad[3]}'
""",
        },
        {
            "name": "Non-monotonic property",
            "code": """
import torch
x = torch.linspace(-5, 5, 100)
out = {fn}(x)
# SiLU has a dip for negative values (non-monotonic)
neg_mask = x < 0
neg_out = out[neg_mask]
assert neg_out.min() < -0.1, 'SiLU should have minimum < -0.1 for negative inputs'
""",
        },
        {
            "name": "Bounded below for negative inputs",
            "code": """
import torch
x = torch.linspace(-100, -1, 100)
out = {fn}(x)
# As x → -∞, SiLU(x) ≈ x * e^x → 0 (approaches 0 from below)
assert out.min() > -1, f'SiLU should be bounded below > -1, got {out.min()}'
""",
        },
    ],
}
