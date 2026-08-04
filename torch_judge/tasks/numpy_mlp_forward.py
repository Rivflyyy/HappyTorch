"""Forward pass of an L-layer ReLU MLP — pure NumPy."""

TASK = {
    "category": "NumPy 手写神经网络",
    "title": "MLP Forward Pass (NumPy)",
    "difficulty": "Medium",
    "function_name": "mlp_forward",
    "hint": (
        "A = X; for every (W, b): Z = A @ W + b, then A = relu(Z) — except after the LAST layer, "
        "which stays linear (logits). relu is np.maximum(Z, 0). Note W has shape (in, out) here, "
        "so it is A @ W, not A @ W.T. Broadcasting adds b of shape (out,) to every row."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": """
import numpy as np
np.random.seed(0)
X = np.random.randn(5, 4)
params = [(np.random.randn(4, 6), np.random.randn(6)),
          (np.random.randn(6, 3), np.random.randn(3))]
out = {fn}(X, params)
assert isinstance(out, np.ndarray), f'Must return np.ndarray, got {type(out)}'
assert out.shape == (5, 3), f'Expected shape (5, 3), got {out.shape}'
""",
        },
        {
            "name": "Single layer is a plain affine map",
            "code": """
import numpy as np
np.random.seed(1)
X = np.random.randn(3, 4)
W = np.random.randn(4, 2)
b = np.random.randn(2)
out = {fn}(X, [(W, b)])
assert np.allclose(out, X @ W + b), 'With one layer there is no hidden activation — output must be X @ W + b'
""",
        },
        {
            "name": "Matches a reference 3-layer network",
            "code": """
import numpy as np
np.random.seed(2)
X = np.random.randn(7, 5)
params = [(np.random.randn(5, 8), np.random.randn(8)),
          (np.random.randn(8, 6), np.random.randn(6)),
          (np.random.randn(6, 3), np.random.randn(3))]
out = {fn}(X, params)
a = X
for i, (W, b) in enumerate(params):
    z = a @ W + b
    a = np.maximum(z, 0) if i < len(params) - 1 else z
assert np.allclose(out, a, atol=1e-10), 'Output does not match the reference forward pass'
""",
        },
        {
            "name": "ReLU is applied to hidden layers only",
            "code": """
import numpy as np
# Layer 1 always produces negative pre-activations -> ReLU kills them -> output is exactly b2
X = np.ones((2, 3))
W1 = -np.ones((3, 4))
b1 = -np.ones(4)
W2 = np.ones((4, 2))
b2 = np.array([0.5, -1.5])
out = {fn}(X, [(W1, b1), (W2, b2)])
assert np.allclose(out, np.tile(b2, (2, 1))), \\
    f'Dead hidden layer should leave only the output bias, got {out} — is ReLU applied to the hidden layer?'
assert (out < 0).any(), 'The final layer must stay linear — do not apply ReLU to the logits'
""",
        },
        {
            "name": "Does not mutate its inputs",
            "code": """
import numpy as np
np.random.seed(3)
X = np.random.randn(4, 3)
params = [(np.random.randn(3, 5), np.random.randn(5)), (np.random.randn(5, 2), np.random.randn(2))]
X_copy = X.copy()
saved = [(W.copy(), b.copy()) for W, b in params]
out = {fn}(X, params)
assert isinstance(out, np.ndarray) and out.shape == (4, 2), f'Expected an (4, 2) ndarray, got {out}'
assert np.allclose(X, X_copy), 'X was modified in place'
for (W, b), (W0, b0) in zip(params, saved):
    assert np.allclose(W, W0) and np.allclose(b, b0), 'Parameters were modified in place'
""",
        },
        {
            "name": "Pure NumPy — no torch",
            "code": """
import numpy as np
np.random.seed(4)
X = np.random.randn(3, 3)
params = [(np.random.randn(3, 3), np.random.randn(3))]
out = {fn}(X, params)
assert isinstance(out, np.ndarray), f'Must return np.ndarray, got {type(out)}'
assert not type(out).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
