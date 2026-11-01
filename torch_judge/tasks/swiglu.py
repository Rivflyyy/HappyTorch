"""SwiGLU activation task — used in LLaMA, PaLM and modern LLMs."""

TASK = {
    "category": "现代激活函数",
    "title": "SwiGLU Activation",
    "difficulty": "Hard",
    "function_name": "SwiGLU",
    "hint": "SwiGLU(x) = Swish(gate) * up where gate and up are projections. Swish(x) = x * sigmoid(x). Use nn.Linear for gate_proj and up_proj. Both projections map d_in to the constructor's d_out.",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
glu = {fn}(d_in=32, d_out=32)
x = torch.randn(2, 6, 32)
out = glu(x)
assert out.shape == (2, 6, 32), f'Shape mismatch: {out.shape}'
""",
        },
        {
            "name": "Has correct linear projections",
            "code": """
import torch, torch.nn as nn
glu = {fn}(d_in=64, d_out=64)
assert hasattr(glu, 'gate_proj'), 'Need self.gate_proj = nn.Linear'
assert hasattr(glu, 'up_proj'), 'Need self.up_proj = nn.Linear'
assert isinstance(glu.gate_proj, nn.Linear), 'gate_proj should be nn.Linear'
assert isinstance(glu.up_proj, nn.Linear), 'up_proj should be nn.Linear'
""",
        },
        {
            "name": "SwiGLU vs GLU difference",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
glu = {fn}(d_in=16, d_out=16)
x = torch.randn(1, 4, 16)
out = glu(x)
gate = glu.gate_proj(x)
up = glu.up_proj(x)
swish = gate * torch.sigmoid(gate)
ref = swish * up
assert torch.allclose(out, ref, atol=1e-5), 'SwiGLU = Swish(gate) * up'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
glu = {fn}(d_in=16, d_out=16)
x = torch.randn(1, 4, 16, requires_grad=True)
out = glu(x)
out.sum().backward()
assert x.grad is not None, 'x.grad is None'
assert glu.gate_proj.weight.grad is not None, 'gate_proj.weight.grad is None'
assert glu.up_proj.weight.grad is not None, 'up_proj.weight.grad is None'
""",
        },
        {
            "name": "Different from simple linear",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(42)
glu = {fn}(d_in=32, d_out=32)
linear = nn.Linear(32, 32)
x = torch.randn(1, 4, 32)
glu_out = glu(x)
# SwiGLU should have different output than simple linear
assert not torch.allclose(glu_out, linear(x), atol=0.1), 'SwiGLU should differ from simple linear'
""",
        },
    ],
}
