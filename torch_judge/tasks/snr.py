"""SNR (Sigmoid Noise Schedule) task — noise scheduling for diffusion models."""

TASK = {
    "category": "扩散模型训练",
    "title": "Sigmoid Noise Schedule",
    "difficulty": "Medium",
    "function_name": "sigmoid_schedule",
    "hint": 'Use beta(t) = beta_start + (beta_end - beta_start) * sigmoid(gamma * (t - 0.5)). Gamma controls steepness; t=0.5 gives the midpoint of the beta bounds.',
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
t = torch.linspace(0, 1, 100)
betas = {fn}(t, beta_start=0.0001, beta_end=0.02)
assert betas.shape == t.shape, f'Shape mismatch: {betas.shape}'
""",
        },
        {
            "name": "Bounds at endpoints",
            "code": """
import torch
t = torch.tensor([0.0, 1.0])
betas = {fn}(t, beta_start=0.0001, beta_end=0.02)
assert betas[0] < 0.001, f'beta at t=0 should be near beta_start, got {betas[0]}'
assert betas[1] > 0.015, f'beta at t=1 should be near beta_end, got {betas[1]}'
""",
        },
        {
            "name": "Monotonic increase",
            "code": """
import torch
t = torch.linspace(0, 1, 100)
betas = {fn}(t, beta_start=0.0001, beta_end=0.02)
diffs = betas[1:] - betas[:-1]
assert (diffs >= -1e-6).all(), 'Betas should be monotonically non-decreasing'
""",
        },
        {
            "name": "Numerical sigmoid schedule",
            "code": """
import torch
t = torch.tensor([0., 0.25, 0.5, 0.75, 1.])
for gamma in (2.0, 10.0):
    expected = 0.001 + 0.039 * torch.sigmoid(gamma * (t - 0.5))
    out = {fn}(t, beta_start=0.001, beta_end=0.04, gamma=gamma)
    assert torch.allclose(out, expected, atol=1e-7), 'Use the stated sigmoid beta schedule and honor gamma'
""",
        },
        {
            "name": "Different parameters give different schedules",
            "code": """
import torch
t = torch.linspace(0, 1, 50)
betas1 = {fn}(t, beta_start=0.0001, beta_end=0.02)
betas2 = {fn}(t, beta_start=0.0001, beta_end=0.05)
assert not torch.allclose(betas1, betas2, atol=1e-3), 'Different beta_end should give different schedule'
""",
        },
    ],
}
