"""Softmax + cross-entropy loss and its gradient — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "Softmax Cross-Entropy + Gradient (NumPy)",
    "difficulty": "Medium",
    "function_name": "softmax_cross_entropy",
    "hint": (
        "Stable softmax: subtract the row max before exp. loss = -mean(log p[i, y_i]). "
        "The famous shortcut: fusing softmax with cross-entropy collapses the whole Jacobian into "
        "dlogits = (p - onehot(y)) / N. Build it by copying p and subtracting 1 at the label positions."
    ),
    "tests": [
        {
            "name": "Return types and shapes",
            "code": """
import numpy as np
np.random.seed(0)
logits = np.random.randn(6, 4)
labels = np.array([0, 1, 2, 3, 0, 1])
loss, dlogits = {fn}(logits, labels)
assert np.isscalar(loss) or np.ndim(loss) == 0, f'loss must be a scalar, got {np.shape(loss)}'
assert isinstance(dlogits, np.ndarray), f'dlogits must be np.ndarray, got {type(dlogits)}'
assert dlogits.shape == logits.shape, f'dlogits shape {dlogits.shape} != logits shape {logits.shape}'
""",
        },
        {
            "name": "Loss matches the reference",
            "code": """
import numpy as np
np.random.seed(1)
logits = np.random.randn(8, 5)
labels = np.random.randint(0, 5, size=8)
loss, _ = {fn}(logits, labels)
z = logits - logits.max(axis=1, keepdims=True)
logZ = np.log(np.exp(z).sum(axis=1))
ref = float(np.mean(logZ - z[np.arange(8), labels]))
assert abs(float(loss) - ref) < 1e-9, f'Loss {float(loss)} != reference {ref} (did you average over the batch?)'
""",
        },
        {
            "name": "Uniform logits give log(C)",
            "code": """
import numpy as np
logits = np.zeros((4, 10))
labels = np.array([0, 3, 7, 9])
loss, dlogits = {fn}(logits, labels)
assert abs(float(loss) - np.log(10)) < 1e-9, f'Uniform logits over 10 classes must give log(10)={np.log(10):.4f}, got {float(loss)}'
""",
        },
        {
            "name": "Gradient matches autograd",
            "code": """
import numpy as np
import torch
np.random.seed(2)
logits = np.random.randn(7, 6)
labels = np.random.randint(0, 6, size=7)
_, dlogits = {fn}(logits, labels)
t = torch.tensor(logits, requires_grad=True)
ref_loss = torch.nn.functional.cross_entropy(t, torch.tensor(labels))
ref_loss.backward()
assert np.allclose(dlogits, t.grad.numpy(), atol=1e-8), 'dlogits does not match PyTorch autograd'
""",
        },
        {
            "name": "Gradient rows sum to zero",
            "code": """
import numpy as np
np.random.seed(3)
logits = np.random.randn(5, 4)
labels = np.random.randint(0, 4, size=5)
_, dlogits = {fn}(logits, labels)
row_sums = dlogits.sum(axis=1)
assert np.allclose(row_sums, 0, atol=1e-10), \\
    f'Each row of (p - onehot)/N sums to 0 because probabilities sum to 1: {row_sums}'
assert np.allclose(dlogits.sum(), 0, atol=1e-10), 'Total gradient must be 0'
""",
        },
        {
            "name": "Numerically stable on large logits",
            "code": """
import numpy as np
logits = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
labels = np.array([1, 2])
loss, dlogits = {fn}(logits, labels)
assert np.isfinite(loss), f'Loss overflowed to {loss} — subtract the row max before exp'
assert np.isfinite(dlogits).all(), 'Gradient overflowed — subtract the row max before exp'
""",
        },
        {
            "name": "Shift invariance",
            "code": """
import numpy as np
np.random.seed(4)
logits = np.random.randn(4, 3)
labels = np.array([0, 1, 2, 1])
loss_a, grad_a = {fn}(logits, labels)
loss_b, grad_b = {fn}(logits + 7.0, labels)
assert abs(float(loss_a) - float(loss_b)) < 1e-9, 'Softmax is shift-invariant — adding a constant must not change the loss'
assert np.allclose(grad_a, grad_b, atol=1e-9), 'Gradient must be shift-invariant too'
""",
        },
    ],
}
