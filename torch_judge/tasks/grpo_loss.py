"""GRPO loss task."""

TASK = {
    "category": "RLHF",
    "title": "GRPO Loss",
    "difficulty": "Hard",
    "function_name": "grpo_loss",
    "hint": (
        "First normalize rewards within each group to get group-relative advantages: "
        "(r - mean(group)) / (std(group, unbiased=False) + 1e-8). Then reuse the PPO clipped objective "
        "with ratio = exp(new_log_probs - old_log_probs)."
    ),
    "tests": [
        {
            "name": "Matches reference formula",
            "code": """
import torch
old_log_probs = torch.log(torch.tensor([0.3, 0.4, 0.2, 0.5], dtype=torch.float32))
new_log_probs = torch.log(torch.tensor([0.33, 0.36, 0.25, 0.45], dtype=torch.float32))
rewards = torch.tensor([1.0, 3.0, 2.0, 6.0], dtype=torch.float32)
group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
loss = {fn}(old_log_probs, new_log_probs, rewards, group_ids, clip_eps=0.2)

advantages = torch.zeros_like(rewards)
for gid in torch.unique(group_ids):
    mask = group_ids == gid
    group_rewards = rewards[mask]
    advantages[mask] = (group_rewards - group_rewards.mean()) / (group_rewards.std(unbiased=False) + 1e-8)

ratio = torch.exp(new_log_probs - old_log_probs)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
expected = -torch.minimum(unclipped, clipped).mean()
assert torch.allclose(loss, expected, atol=1e-6), f'Loss mismatch: {loss} vs {expected}'
""",
        },
        {
            "name": "Invariant to per-group reward shifts",
            "code": """
import torch
old_log_probs = torch.log(torch.tensor([0.2, 0.4, 0.5, 0.1], dtype=torch.float32))
new_log_probs = torch.log(torch.tensor([0.25, 0.35, 0.45, 0.12], dtype=torch.float32))
rewards = torch.tensor([1.0, 2.0, 10.0, 11.0], dtype=torch.float32)
shifted = torch.tensor([101.0, 102.0, -5.0, -4.0], dtype=torch.float32)
group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
loss_a = {fn}(old_log_probs, new_log_probs, rewards, group_ids, clip_eps=0.2)
loss_b = {fn}(old_log_probs, new_log_probs, shifted, group_ids, clip_eps=0.2)
assert torch.allclose(loss_a, loss_b, atol=1e-6), f'Per-group shifts should not change the loss: {loss_a} vs {loss_b}'
""",
        },
        {
            "name": "Single-item groups have zero advantage",
            "code": """
import torch
old_log_probs = torch.zeros(3)
new_log_probs = torch.tensor([0.1, -0.2, 0.3])
rewards = torch.tensor([5.0, 6.0, 7.0])
group_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
loss = {fn}(old_log_probs, new_log_probs, rewards, group_ids, clip_eps=0.2)
assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f'Single-item groups should produce zero normalized advantage, got {loss}'
""",
        },
    ],
}
