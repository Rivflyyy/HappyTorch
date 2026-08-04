"""2-D convolution forward pass — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "Conv2D Forward (NumPy)",
    "difficulty": "Hard",
    "function_name": "conv2d_forward",
    "hint": (
        "H_out = (H + 2*padding - KH) // stride + 1 (same for W). Pad only the spatial dims with "
        "np.pad(x, ((0,0),(0,0),(p,p),(p,p))). Then for each output position (i, j) slice the patch "
        "x[:, :, i*s:i*s+KH, j*s:j*s+KW] of shape (N, C, KH, KW) and contract it with w (F, C, KH, KW) "
        "over the last three axes — np.tensordot(patch, w, axes=([1,2,3],[1,2,3])) gives (N, F). "
        "Deep learning 'convolution' is really cross-correlation: no kernel flipping."
    ),
    "tests": [
        {
            "name": "Output shape with stride and padding",
            "code": """
import numpy as np
np.random.seed(0)
x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(4, 3, 3, 3)
b = np.random.randn(4)
out = {fn}(x, w, b, stride=1, padding=0)
assert isinstance(out, np.ndarray), f'Must return np.ndarray, got {type(out)}'
assert out.shape == (2, 4, 6, 6), f'Expected (2, 4, 6, 6), got {out.shape}'
out = {fn}(x, w, b, stride=2, padding=1)
assert out.shape == (2, 4, 4, 4), f'Expected (2, 4, 4, 4) for stride=2 padding=1, got {out.shape}'
""",
        },
        {
            "name": "Matches F.conv2d",
            "code": """
import numpy as np
import torch
np.random.seed(1)
x = np.random.randn(3, 2, 7, 9)
w = np.random.randn(5, 2, 3, 3)
b = np.random.randn(5)
out = {fn}(x, w, b, stride=1, padding=1)
ref = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=1, padding=1)
assert np.allclose(out, ref.numpy(), atol=1e-8), 'Output does not match torch.nn.functional.conv2d'
""",
        },
        {
            "name": "Matches F.conv2d with stride 2",
            "code": """
import numpy as np
import torch
np.random.seed(2)
x = np.random.randn(2, 4, 10, 10)
w = np.random.randn(3, 4, 5, 5)
b = np.random.randn(3)
out = {fn}(x, w, b, stride=2, padding=2)
ref = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=2, padding=2)
assert out.shape == tuple(ref.shape), f'Shape {out.shape} != {tuple(ref.shape)}'
assert np.allclose(out, ref.numpy(), atol=1e-8), 'Mismatch with stride=2, padding=2'
""",
        },
        {
            "name": "Identity kernel copies the input",
            "code": """
import numpy as np
np.random.seed(3)
x = np.random.randn(1, 1, 5, 5)
w = np.zeros((1, 1, 3, 3))
w[0, 0, 1, 1] = 1.0          # center tap only
b = np.zeros(1)
out = {fn}(x, w, b, stride=1, padding=1)
assert np.allclose(out, x, atol=1e-10), 'A center-tap kernel with padding=1 must reproduce the input'
""",
        },
        {
            "name": "No kernel flipping (cross-correlation)",
            "code": """
import numpy as np
x = np.zeros((1, 1, 3, 3))
x[0, 0, 0, 0] = 1.0
w = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
b = np.zeros(1)
out = {fn}(x, w, b, stride=1, padding=0)
assert out.shape == (1, 1, 1, 1)
assert np.allclose(out[0, 0, 0, 0], 0.0), \\
    f'Expected w[0,0] * x[0,0] = 0 (cross-correlation). Got {out[0, 0, 0, 0]} — you flipped the kernel'
""",
        },
        {
            "name": "Bias is added per output channel",
            "code": """
import numpy as np
np.random.seed(4)
x = np.random.randn(2, 3, 6, 6)
w = np.random.randn(4, 3, 3, 3)
zero_b = np.zeros(4)
b = np.array([1.0, -2.0, 0.5, 10.0])
out0 = {fn}(x, w, zero_b, stride=1, padding=0)
out1 = {fn}(x, w, b, stride=1, padding=0)
diff = out1 - out0
for f in range(4):
    assert np.allclose(diff[:, f], b[f], atol=1e-9), f'Bias for channel {f} not added correctly'
""",
        },
        {
            "name": "Pure NumPy — no torch",
            "code": """
import numpy as np
np.random.seed(5)
x = np.random.randn(1, 1, 4, 4)
w = np.random.randn(1, 1, 2, 2)
b = np.zeros(1)
out = {fn}(x, w, b)
assert isinstance(out, np.ndarray), f'Must return np.ndarray, got {type(out)}'
assert out.shape == (1, 1, 3, 3), f'Expected (1, 1, 3, 3) with the default stride=1, padding=0, got {out.shape}'
assert not type(out).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
