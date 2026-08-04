"""Full backpropagation through an L-layer ReLU MLP with softmax cross-entropy — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "MLP Backpropagation (NumPy)",
    "difficulty": "Hard",
    "function_name": "mlp_loss_and_grads",
    "hint": (
        "Forward once and CACHE (A_prev, Z) for every layer. Start the backward pass from the fused "
        "softmax-CE gradient dZ_last = (p - onehot(y)) / N. Then for each layer, walking backwards: "
        "dW = A_prev.T @ dZ, db = dZ.sum(axis=0), dA_prev = dZ @ W.T, and dZ_prev = dA_prev * (Z_prev > 0). "
        "Every gradient has exactly the shape of the thing it differentiates."
    ),
    "tests": [
        {
            "name": "Return structure and shapes",
            "code": """
import numpy as np
np.random.seed(0)
X = np.random.randn(6, 4)
labels = np.random.randint(0, 3, size=6)
params = [(np.random.randn(4, 5), np.random.randn(5)),
          (np.random.randn(5, 3), np.random.randn(3))]
loss, grads = {fn}(X, labels, params)
assert np.ndim(loss) == 0, f'loss must be a scalar, got shape {np.shape(loss)}'
assert len(grads) == len(params), f'Need one (dW, db) pair per layer, got {len(grads)}'
for (dW, db), (W, b) in zip(grads, params):
    assert dW.shape == W.shape, f'dW shape {dW.shape} != W shape {W.shape}'
    assert db.shape == b.shape, f'db shape {db.shape} != b shape {b.shape}'
""",
        },
        {
            "name": "Loss matches cross-entropy",
            "code": """
import numpy as np
import torch
np.random.seed(1)
X = np.random.randn(8, 5)
labels = np.random.randint(0, 4, size=8)
params = [(np.random.randn(5, 7), np.random.randn(7)), (np.random.randn(7, 4), np.random.randn(4))]
loss, _ = {fn}(X, labels, params)
t = torch.tensor(X)
for i, (W, b) in enumerate(params):
    t = t @ torch.tensor(W) + torch.tensor(b)
    if i < len(params) - 1:
        t = torch.relu(t)
ref = torch.nn.functional.cross_entropy(t, torch.tensor(labels)).item()
assert abs(float(loss) - ref) < 1e-9, f'Loss {float(loss)} != reference {ref}'
""",
        },
        {
            "name": "Gradients match autograd (3 layers)",
            "code": """
import numpy as np
import torch
np.random.seed(2)
X = np.random.randn(9, 6)
labels = np.random.randint(0, 4, size=9)
shapes = [(6, 8), (8, 5), (5, 4)]
params = [(np.random.randn(*s), np.random.randn(s[1])) for s in shapes]
loss, grads = {fn}(X, labels, params)

ts = [(torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)) for W, b in params]
t = torch.tensor(X)
for i, (W, b) in enumerate(ts):
    t = t @ W + b
    if i < len(ts) - 1:
        t = torch.relu(t)
torch.nn.functional.cross_entropy(t, torch.tensor(labels)).backward()

for i, ((dW, db), (W, b)) in enumerate(zip(grads, ts)):
    assert np.allclose(dW, W.grad.numpy(), atol=1e-8), f'dW mismatch at layer {i}'
    assert np.allclose(db, b.grad.numpy(), atol=1e-8), f'db mismatch at layer {i}'
""",
        },
        {
            "name": "Matches numerical gradient",
            "code": """
import numpy as np
np.random.seed(3)
X = np.random.randn(5, 3)
labels = np.array([0, 1, 2, 1, 0])
params = [(np.random.randn(3, 4), np.random.randn(4)), (np.random.randn(4, 3), np.random.randn(3))]
loss, grads = {fn}(X, labels, params)
eps = 1e-6
W = params[0][0]
num = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        orig = W[i, j]
        W[i, j] = orig + eps
        lp, _ = {fn}(X, labels, params)
        W[i, j] = orig - eps
        lm, _ = {fn}(X, labels, params)
        W[i, j] = orig
        num[i, j] = (float(lp) - float(lm)) / (2 * eps)
assert np.allclose(grads[0][0], num, atol=1e-5), 'Analytic dW1 disagrees with the finite-difference gradient'
""",
        },
        {
            "name": "Dead ReLU units receive no gradient",
            "code": """
import numpy as np
X = np.ones((3, 2))
W1 = np.array([[-5.0, 1.0], [-5.0, 1.0]])
b1 = np.array([-5.0, 0.0])          # unit 0 is always negative -> dead
W2 = np.array([[1.0, -1.0], [2.0, 0.5]])
b2 = np.zeros(2)
labels = np.array([0, 1, 0])
loss, grads = {fn}(X, labels, [(W1, b1), (W2, b2)])
dW1, db1 = grads[0]
assert np.allclose(dW1[:, 0], 0), f'Dead hidden unit must get zero weight gradient, got {dW1[:, 0]}'
assert np.allclose(db1[0], 0), f'Dead hidden unit must get zero bias gradient, got {db1[0]}'
assert np.abs(dW1[:, 1]).sum() > 0, 'The active hidden unit must receive gradient'
""",
        },
        {
            "name": "Gradient descent actually reduces the loss",
            "code": """
import numpy as np
np.random.seed(4)
X = np.random.randn(40, 4)
labels = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
params = [(np.random.randn(4, 16) * 0.5, np.zeros(16)), (np.random.randn(16, 2) * 0.5, np.zeros(2))]
first = None
for step in range(300):
    loss, grads = {fn}(X, labels, params)
    if first is None:
        first = float(loss)
    for (W, b), (dW, db) in zip(params, grads):
        W -= 0.1 * dW
        b -= 0.1 * db
assert float(loss) < first * 0.3, f'Loss barely moved: {first:.4f} -> {float(loss):.4f}; check the gradient signs'
""",
        },
        {
            "name": "Pure NumPy — no torch",
            "code": """
import numpy as np
np.random.seed(5)
X = np.random.randn(4, 3)
labels = np.array([0, 1, 0, 1])
params = [(np.random.randn(3, 2), np.random.randn(2))]
loss, grads = {fn}(X, labels, params)
assert not type(grads[0][0]).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
