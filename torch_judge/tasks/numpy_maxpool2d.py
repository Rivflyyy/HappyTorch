"""Max pooling forward + backward closure — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "MaxPool2D Forward & Backward (NumPy)",
    "difficulty": "Medium",
    "function_name": "maxpool2d",
    "hint": (
        "Return (out, backward) where backward(dout) -> dx. Pooling has no parameters, so all the "
        "backward pass needs is WHERE each maximum came from: record the argmax inside every window "
        "during the forward pass (flatten the window and use argmax, then unflatten with divmod). "
        "Backward is pure routing — each dout value is added to its winner position, everything else "
        "gets 0. Use += so overlapping windows (stride < kernel_size) accumulate."
    ),
    "tests": [
        {
            "name": "Forward shape and returned closure",
            "code": """
import numpy as np
np.random.seed(0)
x = np.random.randn(2, 3, 8, 8)
out, backward = {fn}(x, kernel_size=2, stride=2)
assert isinstance(out, np.ndarray), f'First return must be np.ndarray, got {type(out)}'
assert out.shape == (2, 3, 4, 4), f'Expected (2, 3, 4, 4), got {out.shape}'
assert callable(backward), 'Second return value must be a callable backward(dout) -> dx'
dx = backward(np.ones_like(out))
assert dx.shape == x.shape, f'dx shape {dx.shape} != x shape {x.shape}'
""",
        },
        {
            "name": "Forward matches F.max_pool2d",
            "code": """
import numpy as np
import torch
np.random.seed(1)
x = np.random.randn(3, 4, 10, 10)
out, _ = {fn}(x, kernel_size=2, stride=2)
ref = torch.nn.functional.max_pool2d(torch.tensor(x), kernel_size=2, stride=2)
assert np.allclose(out, ref.numpy(), atol=1e-10), 'Forward output does not match torch max_pool2d'
""",
        },
        {
            "name": "Non-square windows / stride 3",
            "code": """
import numpy as np
import torch
np.random.seed(2)
x = np.random.randn(2, 2, 9, 9)
out, _ = {fn}(x, kernel_size=3, stride=3)
ref = torch.nn.functional.max_pool2d(torch.tensor(x), kernel_size=3, stride=3)
assert out.shape == tuple(ref.shape), f'Shape {out.shape} != {tuple(ref.shape)}'
assert np.allclose(out, ref.numpy(), atol=1e-10), 'Mismatch with kernel_size=3, stride=3'
""",
        },
        {
            "name": "Backward routes gradient to the argmax only",
            "code": """
import numpy as np
x = np.array([[[[1.0, 2.0], [3.0, 9.0]]]])     # (1, 1, 2, 2), max at (1, 1)
out, backward = {fn}(x, kernel_size=2, stride=2)
assert np.allclose(out, 9.0), f'Max should be 9.0, got {out}'
dx = backward(np.array([[[[5.0]]]]))
expected = np.array([[[[0.0, 0.0], [0.0, 5.0]]]])
assert np.allclose(dx, expected), f'Gradient must go only to the max position, got {dx}'
assert np.count_nonzero(dx) == 1, 'Exactly one element per window may receive gradient'
""",
        },
        {
            "name": "Backward matches autograd",
            "code": """
import numpy as np
import torch
np.random.seed(3)
x = np.random.randn(2, 3, 8, 8)
out, backward = {fn}(x, kernel_size=2, stride=2)
dout = np.random.randn(*out.shape)
dx = backward(dout)
tx = torch.tensor(x, requires_grad=True)
ref = torch.nn.functional.max_pool2d(tx, kernel_size=2, stride=2)
ref.backward(torch.tensor(dout))
assert np.allclose(dx, tx.grad.numpy(), atol=1e-10), 'Backward does not match autograd'
""",
        },
        {
            "name": "Overlapping windows accumulate",
            "code": """
import numpy as np
import torch
np.random.seed(4)
x = np.random.randn(1, 1, 5, 5)
out, backward = {fn}(x, kernel_size=3, stride=1)
dout = np.ones_like(out)
dx = backward(dout)
tx = torch.tensor(x, requires_grad=True)
ref = torch.nn.functional.max_pool2d(tx, kernel_size=3, stride=1)
ref.backward(torch.tensor(dout))
assert np.allclose(dx, tx.grad.numpy(), atol=1e-10), \\
    'With stride < kernel_size a pixel can win several windows — accumulate with += instead of ='
assert dx.sum() > 0, 'Gradient vanished entirely'
""",
        },
        {
            "name": "Pure NumPy — no torch",
            "code": """
import numpy as np
np.random.seed(5)
x = np.random.randn(1, 1, 4, 4)
out, backward = {fn}(x, kernel_size=2, stride=2)
dx = backward(np.ones_like(out))
assert not type(out).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
assert not type(dx).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
