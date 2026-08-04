"""2-D convolution backward pass — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "Conv2D Backward (NumPy)",
    "difficulty": "Hard",
    "function_name": "conv2d_backward",
    "hint": (
        "db = dout.sum(axis=(0, 2, 3)) — every spatial position shares the same bias. "
        "Loop over output positions again; let d = dout[:, :, i, j] of shape (N, F) and "
        "patch = x_pad[:, :, i*s:i*s+KH, j*s:j*s+KW] of shape (N, C, KH, KW). Then "
        "dw += tensordot(d, patch, axes=([0],[0])) and the same window of dx_pad gets "
        "+= tensordot(d, w, axes=([1],[0])). Windows overlap when stride < kernel, so ACCUMULATE (+=) "
        "instead of assigning. Finally crop the padding off dx_pad."
    ),
    "tests": [
        {
            "name": "Gradient shapes",
            "code": """
import numpy as np
np.random.seed(0)
x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(4, 3, 3, 3)
dout = np.random.randn(2, 4, 6, 6)
dx, dw, db = {fn}(dout, x, w, stride=1, padding=0)
assert dx.shape == x.shape, f'dx shape {dx.shape} != x shape {x.shape}'
assert dw.shape == w.shape, f'dw shape {dw.shape} != w shape {w.shape}'
assert db.shape == (4,), f'db shape {db.shape} != (4,)'
""",
        },
        {
            "name": "Bias gradient sums over batch and space",
            "code": """
import numpy as np
np.random.seed(1)
x = np.random.randn(2, 2, 6, 6)
w = np.random.randn(3, 2, 3, 3)
dout = np.random.randn(2, 3, 4, 4)
_, _, db = {fn}(dout, x, w, stride=1, padding=0)
assert np.allclose(db, dout.sum(axis=(0, 2, 3)), atol=1e-10), 'db must be dout summed over N, H_out, W_out'
""",
        },
        {
            "name": "Matches autograd (stride 1, padding 1)",
            "code": """
import numpy as np
import torch
np.random.seed(2)
x = np.random.randn(3, 2, 7, 7)
w = np.random.randn(4, 2, 3, 3)
dout = np.random.randn(3, 4, 7, 7)
dx, dw, db = {fn}(dout, x, w, stride=1, padding=1)

tx = torch.tensor(x, requires_grad=True)
tw = torch.tensor(w, requires_grad=True)
tb = torch.zeros(4, dtype=torch.float64, requires_grad=True)
out = torch.nn.functional.conv2d(tx, tw, tb, stride=1, padding=1)
out.backward(torch.tensor(dout))
assert np.allclose(dx, tx.grad.numpy(), atol=1e-8), 'dx mismatch'
assert np.allclose(dw, tw.grad.numpy(), atol=1e-8), 'dw mismatch'
assert np.allclose(db, tb.grad.numpy(), atol=1e-8), 'db mismatch'
""",
        },
        {
            "name": "Matches autograd (stride 2, padding 2)",
            "code": """
import numpy as np
import torch
np.random.seed(3)
x = np.random.randn(2, 3, 9, 11)
w = np.random.randn(5, 3, 5, 5)
tx = torch.tensor(x, requires_grad=True)
tw = torch.tensor(w, requires_grad=True)
tb = torch.zeros(5, dtype=torch.float64, requires_grad=True)
out = torch.nn.functional.conv2d(tx, tw, tb, stride=2, padding=2)
dout = np.random.randn(*out.shape)
dx, dw, db = {fn}(dout, x, w, stride=2, padding=2)
out.backward(torch.tensor(dout))
assert np.allclose(dx, tx.grad.numpy(), atol=1e-8), 'dx mismatch with stride=2'
assert np.allclose(dw, tw.grad.numpy(), atol=1e-8), 'dw mismatch with stride=2'
assert np.allclose(db, tb.grad.numpy(), atol=1e-8), 'db mismatch with stride=2'
""",
        },
        {
            "name": "Overlapping windows accumulate",
            "code": """
import numpy as np
x = np.zeros((1, 1, 4, 4))
w = np.ones((1, 1, 2, 2))
dout = np.ones((1, 1, 3, 3))          # stride 1 -> interior pixels are covered 4 times
dx, _, _ = {fn}(dout, x, w, stride=1, padding=0)
assert np.allclose(dx[0, 0, 0, 0], 1.0), f'Corner pixel is used once, expected 1.0, got {dx[0, 0, 0, 0]}'
assert np.allclose(dx[0, 0, 1, 1], 4.0), \\
    f'Interior pixel is used by 4 windows, expected 4.0, got {dx[0, 0, 1, 1]} — accumulate with += instead of ='
""",
        },
        {
            "name": "Padding is cropped off dx",
            "code": """
import numpy as np
import torch
np.random.seed(4)
x = np.random.randn(1, 2, 5, 5)
w = np.random.randn(3, 2, 3, 3)
dout = np.random.randn(1, 3, 5, 5)
dx, _, _ = {fn}(dout, x, w, stride=1, padding=1)
assert dx.shape == x.shape, f'dx must have the shape of x, not of the padded input: {dx.shape}'
tx = torch.tensor(x, requires_grad=True)
out = torch.nn.functional.conv2d(tx, torch.tensor(w), None, stride=1, padding=1)
out.backward(torch.tensor(dout))
assert np.allclose(dx, tx.grad.numpy(), atol=1e-8), 'dx mismatch — did you crop the padded border correctly?'
""",
        },
        {
            "name": "Pure NumPy — no torch",
            "code": """
import numpy as np
np.random.seed(5)
x = np.random.randn(1, 1, 4, 4)
w = np.random.randn(1, 1, 2, 2)
dout = np.random.randn(1, 1, 3, 3)
dx, dw, db = {fn}(dout, x, w)
assert isinstance(dx, np.ndarray) and isinstance(dw, np.ndarray), 'Must return NumPy arrays'
assert not type(dx).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
