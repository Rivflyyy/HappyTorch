"""DMD2 — the distribution matching (score-difference) loss used to distill a one-step generator."""

TASK = {
    "category": "扩散模型训练",
    "title": "DMD2 Distribution Matching Loss",
    "difficulty": "Hard",
    "function_name": "dmd_loss",
    "hint": (
        "The reverse-KL gradient w.r.t. a generated sample is the score difference "
        "(s_fake - s_real), which in x0-prediction space is proportional to (pred_fake - pred_real). "
        "Normalize it per sample by |pred_real - x_gen|.mean() so the scale is dimension-free, then "
        "turn the gradient into a differentiable surrogate: loss = 0.5 * mse(x_gen, stopgrad(x_gen - grad)). "
        "Differentiating that gives exactly d(loss)/d(x_gen) = grad / numel."
    ),
    "tests": [
        {
            "name": "Scalar, non-negative loss",
            "code": """
import torch
torch.manual_seed(0)
x = torch.randn(4, 3, 8, 8, requires_grad=True)
pred_real = torch.randn(4, 3, 8, 8)
pred_fake = torch.randn(4, 3, 8, 8)
loss = {fn}(x, pred_real, pred_fake)
assert isinstance(loss, torch.Tensor), f'Loss must be a tensor, got {type(loss)}'
assert loss.dim() == 0, f'Loss must be a scalar, got shape {loss.shape}'
assert loss.item() >= 0, f'Loss must be non-negative, got {loss.item()}'
""",
        },
        {
            "name": "Zero loss when the two scores agree",
            "code": """
import torch
torch.manual_seed(1)
x = torch.randn(2, 3, 4, 4, requires_grad=True)
pred = torch.randn(2, 3, 4, 4)
loss = {fn}(x, pred, pred)
assert loss.item() < 1e-12, f'If pred_fake == pred_real the distributions match, loss must be 0, got {loss.item()}'
loss.backward()
assert x.grad.abs().max() < 1e-10, 'Gradient must vanish when the two scores agree'
""",
        },
        {
            "name": "Gradient equals the normalized score difference",
            "code": """
import torch
torch.manual_seed(2)
x = torch.randn(4, 3, 8, 8, requires_grad=True)
pred_real = torch.randn(4, 3, 8, 8)
pred_fake = torch.randn(4, 3, 8, 8)
loss = {fn}(x, pred_real, pred_fake)
loss.backward()
normalizer = (pred_real - x.detach()).abs().mean(dim=(1, 2, 3), keepdim=True)
expected = (pred_fake - pred_real) / (normalizer + 1e-8) / x.numel()
assert x.grad is not None, 'No gradient reached x_gen'
assert torch.allclose(x.grad, expected, atol=1e-6), \\
    'd(loss)/d(x_gen) must equal (pred_fake - pred_real) / normalizer / numel'
""",
        },
        {
            "name": "No gradient leaks into the score predictions",
            "code": """
import torch
torch.manual_seed(3)
x = torch.randn(2, 3, 4, 4, requires_grad=True)
pred_real = torch.randn(2, 3, 4, 4, requires_grad=True)
pred_fake = torch.randn(2, 3, 4, 4, requires_grad=True)
loss = {fn}(x, pred_real, pred_fake)
loss.backward()
assert pred_real.grad is None or pred_real.grad.abs().max() < 1e-12, \\
    'The regression target must be detached — the frozen teacher must not receive gradient'
assert pred_fake.grad is None or pred_fake.grad.abs().max() < 1e-12, \\
    'The fake score model is trained by its own diffusion loss, not by this one — detach the target'
""",
        },
        {
            "name": "Scale invariance from the normalizer",
            "code": """
import torch
torch.manual_seed(4)
x = torch.randn(3, 2, 4, 4)
pred_real = torch.randn(3, 2, 4, 4)
pred_fake = torch.randn(3, 2, 4, 4)
c = 50.0
loss_a = {fn}(x, pred_real, pred_fake)
loss_b = {fn}(x * c, pred_real * c, pred_fake * c)
assert torch.allclose(loss_a, loss_b, rtol=1e-3, atol=1e-6), \\
    f'Dividing by |pred_real - x_gen|.mean() makes the loss scale-invariant: {loss_a.item()} vs {loss_b.item()}'
""",
        },
        {
            "name": "Normalizer is per-sample",
            "code": """
import torch
torch.manual_seed(5)
x = torch.randn(2, 6)
pred_real = torch.randn(2, 6)
pred_fake = torch.randn(2, 6)
base = {fn}(x, pred_real, pred_fake)
x2, pr2, pf2 = x.clone(), pred_real.clone(), pred_fake.clone()
x2[0] *= 20
pr2[0] *= 20
pf2[0] *= 20
scaled = {fn}(x2, pr2, pf2)
assert torch.allclose(base, scaled, rtol=1e-3, atol=1e-6), \\
    'Rescaling one sample changed the loss — the normalizer must be computed per sample, not over the whole batch'
""",
        },
        {
            "name": "Finite when the normalizer degenerates",
            "code": """
import torch
torch.manual_seed(6)
x = torch.randn(2, 4, requires_grad=True)
pred_real = x.detach().clone()   # normalizer becomes exactly 0
pred_fake = torch.randn(2, 4)
loss = {fn}(x, pred_real, pred_fake)
assert not torch.isnan(loss).any(), 'Loss became NaN — guard the division with eps and/or torch.nan_to_num'
loss.backward()
assert not torch.isnan(x.grad).any(), 'Gradient became NaN on a degenerate normalizer'
""",
        },
    ],
}
