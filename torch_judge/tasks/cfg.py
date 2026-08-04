"""Classifier-Free Guidance (CFG) with the optional std-rescale trick."""

TASK = {
    "category": "扩散模型训练",
    "title": "Classifier-Free Guidance",
    "difficulty": "Medium",
    "function_name": "classifier_free_guidance",
    "hint": (
        "eps = eps_uncond + w * (eps_cond - eps_uncond). With this convention w=1 means 'no guidance' "
        "(pure conditional) and w=0 means unconditional. For rescale: compute per-sample std over all "
        "non-batch dims, scale the guided prediction by std_cond / std_cfg, then blend: "
        "phi * rescaled + (1 - phi) * eps_cfg."
    ),
    "tests": [
        {
            "name": "Basic combination formula",
            "code": """
import torch
torch.manual_seed(0)
eps_u = torch.randn(4, 3, 8, 8)
eps_c = torch.randn(4, 3, 8, 8)
out = {fn}(eps_u, eps_c, 7.5)
assert out.shape == eps_u.shape, f'Shape mismatch: {out.shape}'
expected = eps_u + 7.5 * (eps_c - eps_u)
assert torch.allclose(out, expected, atol=1e-6), 'Guided prediction does not match eps_u + w*(eps_c - eps_u)'
""",
        },
        {
            "name": "w=1 is pure conditional, w=0 is unconditional",
            "code": """
import torch
torch.manual_seed(1)
eps_u = torch.randn(2, 4)
eps_c = torch.randn(2, 4)
out1 = {fn}(eps_u, eps_c, 1.0)
assert torch.allclose(out1, eps_c, atol=1e-6), 'guidance_scale=1.0 must return the conditional prediction'
out0 = {fn}(eps_u, eps_c, 0.0)
assert torch.allclose(out0, eps_u, atol=1e-6), 'guidance_scale=0.0 must return the unconditional prediction'
""",
        },
        {
            "name": "Extrapolation direction is amplified",
            "code": """
import torch
torch.manual_seed(2)
eps_u = torch.randn(3, 5)
eps_c = torch.randn(3, 5)
w = 5.0
out = {fn}(eps_u, eps_c, w)
delta = out - eps_u
assert torch.allclose(delta, w * (eps_c - eps_u), atol=1e-6), \\
    'CFG extrapolates along (cond - uncond); the step must be exactly w times that direction'
""",
        },
        {
            "name": "Negative / fractional guidance scales",
            "code": """
import torch
torch.manual_seed(3)
eps_u = torch.randn(2, 3, 4, 4)
eps_c = torch.randn(2, 3, 4, 4)
for w in [-2.0, 0.5, 3.0, 12.0]:
    out = {fn}(eps_u, eps_c, w)
    expected = eps_u + w * (eps_c - eps_u)
    assert torch.allclose(out, expected, atol=1e-6), f'Wrong result for guidance_scale={w}'
""",
        },
        {
            "name": "rescale=1.0 restores the conditional std",
            "code": """
import torch
torch.manual_seed(4)
eps_u = torch.randn(4, 3, 8, 8)
eps_c = torch.randn(4, 3, 8, 8)
out = {fn}(eps_u, eps_c, 7.5, rescale=1.0)
dims = (1, 2, 3)
std_out = out.std(dim=dims)
std_cond = eps_c.std(dim=dims)
assert torch.allclose(std_out, std_cond, atol=1e-4), \\
    f'With rescale=1.0 the per-sample std must match the conditional std: {std_out} vs {std_cond}'
""",
        },
        {
            "name": "rescale blends between guided and rescaled",
            "code": """
import torch
torch.manual_seed(5)
eps_u = torch.randn(3, 2, 6, 6)
eps_c = torch.randn(3, 2, 6, 6)
w, phi = 6.0, 0.7
out = {fn}(eps_u, eps_c, w, rescale=phi)
dims = (1, 2, 3)
cfg = eps_u + w * (eps_c - eps_u)
ratio = eps_c.std(dim=dims, keepdim=True) / cfg.std(dim=dims, keepdim=True)
expected = phi * (cfg * ratio) + (1 - phi) * cfg
assert torch.allclose(out, expected, atol=1e-5), 'rescale must linearly blend the rescaled and raw guided predictions'
""",
        },
        {
            "name": "rescale is per-sample, not global",
            "code": """
import torch
torch.manual_seed(6)
eps_u = torch.randn(2, 3, 8, 8)
eps_c = torch.randn(2, 3, 8, 8)
out = {fn}(eps_u, eps_c, 7.5, rescale=1.0)
# Scale only sample 0 — sample 1 must be untouched
eps_u2, eps_c2 = eps_u.clone(), eps_c.clone()
eps_u2[0] *= 10
eps_c2[0] *= 10
out2 = {fn}(eps_u2, eps_c2, 7.5, rescale=1.0)
assert torch.allclose(out[1], out2[1], atol=1e-5), \\
    'Statistics must be computed per sample — changing sample 0 changed sample 1'
assert torch.allclose(out2[0], out[0] * 10, atol=1e-4), 'Sample 0 should scale linearly with its inputs'
""",
        },
        {
            "name": "rescale=0.0 is a no-op",
            "code": """
import torch
torch.manual_seed(7)
eps_u = torch.randn(2, 3, 4, 4)
eps_c = torch.randn(2, 3, 4, 4)
a = {fn}(eps_u, eps_c, 7.5)
b = {fn}(eps_u, eps_c, 7.5, rescale=0.0)
assert torch.allclose(a, b, atol=1e-6), 'rescale=0.0 must equal the plain guided prediction'
""",
        },
    ],
}
