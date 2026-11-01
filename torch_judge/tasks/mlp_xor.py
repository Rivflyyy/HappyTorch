"""Train a 2-layer MLP on XOR using only NumPy (manual forward + backward)."""

TASK = {
    "category": "ML 基础与解码策略",
    "title": "Train a 2-Layer MLP on XOR (Manual NumPy)",
    "difficulty": "Hard",
    "function_name": "mlp_xor",
    "hint": (
        "Forward: Z1 = X @ W1 + b1, A1 = relu(Z1), Z2 = A1 @ W2 + b2 (linear output). "
        "Loss = mean((Z2 - Y)^2). Backward: dZ2 = 2*(Z2-Y)/n, dW2 = A1.T @ dZ2, "
        "db2 = sum(dZ2), dA1 = dZ2 @ W2.T, dZ1 = dA1 * (Z1 > 0), dW1 = X.T @ dZ1, "
        "db1 = sum(dZ1). Use He initialization: W * sqrt(2/fan_in)."
    ),
    "tests": [
        {
            "name": "Returns numpy array with correct shape",
            "code": """
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
preds = {fn}(X, Y)
assert isinstance(preds, np.ndarray), f'Expected np.ndarray, got {type(preds)}'
assert preds.shape == (4, 1), f'Expected shape (4, 1), got {preds.shape}'
""",
        },
        {
            "name": "Solves XOR (MSE < 0.01)",
            "code": """
import numpy as np
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
preds = {fn}(X, Y, hidden_size=4, lr=0.1, epochs=10000)
mse = np.mean((preds - Y) ** 2)
assert mse < 0.01, f'MSE too high: {mse:.4f}, predictions: {preds.ravel()}'
""",
        },
        {
            "name": "Works with different hidden_size",
            "code": """
import numpy as np
np.random.seed(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
preds = {fn}(X, Y, hidden_size=8, lr=0.1, epochs=10000)
mse = np.mean((preds - Y) ** 2)
assert mse < 0.01, f'MSE too high with hidden_size=8: {mse:.4f}'
""",
        },
        {
            "name": "Returns NumPy predictions",
            "code": """
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
preds = {fn}(X, Y, hidden_size=4, lr=0.1, epochs=1000)
assert isinstance(preds, np.ndarray), f'Must return np.ndarray, got {type(preds).__module__}.{type(preds).__name__}'
assert not type(preds).__module__.startswith('torch'), 'Must use only NumPy, no PyTorch'
""",
        },
    ],
}
