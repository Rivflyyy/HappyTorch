"""DPO loss task."""

TASK = {
    "category": "RLHF",
    "title": "DPO Loss",
    "difficulty": "Hard",
    "function_name": "dpo_loss",
    "hint": (
        "DPO compares the policy preference gap against the reference preference gap. "
        "Compute logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)), "
        "then return -logsigmoid(logits).mean()."
    ),
    "tests": [
        {
            "name": "Matches reference formula",
            "code": """
import torch
import torch.nn.functional as F
policy_chosen = torch.tensor([-0.2, -0.4, -0.1])
policy_rejected = torch.tensor([-1.0, -0.5, -0.8])
ref_chosen = torch.tensor([-0.3, -0.4, -0.2])
ref_rejected = torch.tensor([-0.7, -0.4, -0.6])
loss = {fn}(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.5)
logits = 0.5 * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
expected = -F.logsigmoid(logits).mean()
assert torch.allclose(loss, expected, atol=1e-6), f'Loss mismatch: {loss} vs {expected}'
""",
        },
        {
            "name": "Stronger policy preference lowers the loss",
            "code": """
import torch
weak = {fn}(
    torch.tensor([-0.5]),
    torch.tensor([-0.7]),
    torch.tensor([-0.4]),
    torch.tensor([-0.6]),
    beta=1.0,
)
strong = {fn}(
    torch.tensor([-0.1]),
    torch.tensor([-1.2]),
    torch.tensor([-0.4]),
    torch.tensor([-0.6]),
    beta=1.0,
)
assert strong < weak, f'Stronger preferred-vs-rejected gap should reduce loss: {strong} vs {weak}'
""",
        },
        {
            "name": "Larger beta reduces loss for a favorable preference gap",
            "code": """
import torch
low_beta = {fn}(
    torch.tensor([-0.3]),
    torch.tensor([-0.4]),
    torch.tensor([-0.3]),
    torch.tensor([-0.2]),
    beta=0.1,
)
high_beta = {fn}(
    torch.tensor([-0.3]),
    torch.tensor([-0.4]),
    torch.tensor([-0.3]),
    torch.tensor([-0.2]),
    beta=2.0,
)
assert high_beta < low_beta, f'When the policy is better than reference, larger beta should reduce loss: {high_beta} vs {low_beta}'
""",
        },
    ],
}
