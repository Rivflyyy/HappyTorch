"""KV Cache task — essential for efficient LLM inference."""

TASK = {
    "category": "LLM 推理组件",
    "title": "KV Cache for Attention",
    "difficulty": "Hard",
    "function_name": "KVCache",
    "hint": "KV Cache stores past K, V tensors to avoid recomputation during generation. Implement append(k, v) to grow cache, and get() to return cached K, V. Shape: (batch, num_heads, seq_len, head_dim). Handle incremental updates efficiently.",
    "tests": [
        {
            "name": "Initialize and shape",
            "code": """
import torch
cache = {fn}(num_heads=8, head_dim=64, max_seq_len=256, batch_size=2)
k, v = cache.get()
assert k.shape == v.shape == (2, 8, 0, 64), 'A new cache must expose zero stored tokens via get()'
""",
        },
        {
            "name": "Append and get",
            "code": """
import torch
cache = {fn}(num_heads=4, head_dim=16, max_seq_len=32, batch_size=1)
k1 = torch.randn(1, 4, 3, 16)  # seq_len=3
v1 = torch.randn(1, 4, 3, 16)
cache.append(k1, v1)
k, v = cache.get()
assert k.shape == (1, 4, 3, 16), f'K shape: {k.shape}'
assert v.shape == (1, 4, 3, 16), f'V shape: {v.shape}'
""",
        },
        {
            "name": "Incremental append",
            "code": """
import torch
cache = {fn}(num_heads=4, head_dim=16, max_seq_len=32, batch_size=1)
k1 = torch.randn(1, 4, 3, 16)
v1 = torch.randn(1, 4, 3, 16)
cache.append(k1, v1)
k2 = torch.randn(1, 4, 2, 16)  # append 2 more tokens
v2 = torch.randn(1, 4, 2, 16)
cache.append(k2, v2)
k, v = cache.get()
assert k.shape == (1, 4, 5, 16), f'K shape after append: {k.shape}'
assert v.shape == (1, 4, 5, 16), f'V shape after append: {v.shape}'
""",
        },
        {
            "name": "Get returns correct values",
            "code": """
import torch
cache = {fn}(num_heads=4, head_dim=16, max_seq_len=32, batch_size=1)
k1 = torch.ones(1, 4, 2, 16) * 1.0
v1 = torch.ones(1, 4, 2, 16) * 2.0
cache.append(k1, v1)
k2 = torch.ones(1, 4, 1, 16) * 3.0
v2 = torch.ones(1, 4, 1, 16) * 4.0
cache.append(k2, v2)
k, v = cache.get()
assert torch.allclose(k[:, :, :2], k1), 'First K values incorrect'
assert torch.allclose(k[:, :, 2:], k2), 'Second K values incorrect'
assert torch.allclose(v[:, :, :2], v1), 'First V values incorrect'
assert torch.allclose(v[:, :, 2:], v2), 'Second V values incorrect'
""",
        },
        {
            "name": "Clear cache",
            "code": """
import torch
cache = {fn}(num_heads=4, head_dim=16, max_seq_len=32, batch_size=1)
k = torch.randn(1, 4, 5, 16)
v = torch.randn(1, 4, 5, 16)
cache.append(k, v)
cache.clear()
# After clear, cache should be empty (seq_len = 0 or get returns empty)
assert cache.seq_len == 0, 'seq_len should be 0 after clear'
assert cache.get()[0].shape == cache.get()[1].shape == (1, 4, 0, 16), 'get() must be empty after clear'
new_k, new_v = torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16)
cache.append(new_k, new_v)
assert torch.equal(cache.get()[0], new_k) and torch.equal(cache.get()[1], new_v), 'Reusing a cleared cache must not expose old tokens'
""",
        },
        {
            "name": "Current sequence length",
            "code": """
import torch
cache = {fn}(num_heads=4, head_dim=16, max_seq_len=32, batch_size=1)
assert cache.seq_len == 0, 'Initial seq_len should be 0'
cache.append(torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))
assert cache.seq_len == 3, f'seq_len after append: {cache.seq_len}'
cache.append(torch.randn(1, 4, 2, 16), torch.randn(1, 4, 2, 16))
assert cache.seq_len == 5, f'seq_len after second append: {cache.seq_len}'
""",
        },
    ],
}
