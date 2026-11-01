"""Equivalent implementations must pass; plausible mathematical mistakes must fail."""
import unittest

import torch
from torch import nn

from test_numerical_judges import judge_failures, load_solution


class FusedFiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.conditioner = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * feature_dim))

    def forward(self, x, cond):
        scale, shift = self.conditioner(cond).chunk(2, dim=-1)
        return scale[:, None] * x + shift[:, None]


class RenamedAdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.standardize = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        return self.scale(cond)[:, None] * self.standardize(x) + self.shift(cond)[:, None]


class PrivateCache:
    def __init__(self, num_heads, head_dim, max_seq_len, batch_size):
        self._keys = torch.empty(batch_size, num_heads, 0, head_dim)
        self._values = torch.empty_like(self._keys)

    @property
    def seq_len(self):
        return self._keys.shape[2]

    def append(self, k, v):
        self._keys = torch.cat((self._keys, k), dim=2)
        self._values = torch.cat((self._values, v), dim=2)

    def get(self):
        return self._keys, self._values

    def clear(self):
        self._keys = self._keys[:, :, :0]
        self._values = self._values[:, :, :0]


def inverse_cdf(probs):
    u = torch.rand_like(probs[..., :1])
    return (u > probs.cumsum(-1)).sum(-1).clamp_max(probs.shape[-1] - 1)


def alternative_temperature(logits, temperature):
    return inverse_cdf(torch.softmax(logits / temperature, -1))


def compact_top_k(logits, k):
    values, indices = logits.topk(min(k, logits.shape[-1]), dim=-1)
    draw = torch.multinomial(torch.softmax(values, -1), 1)
    return indices.gather(-1, draw).squeeze(-1)


def alternative_top_p(logits, p):
    probs, indices = logits.softmax(-1).sort(descending=True, dim=-1)
    keep = torch.cat((torch.ones_like(probs[..., :1], dtype=torch.bool),
                      probs.cumsum(-1)[..., :-1] < p), -1)
    probs = probs * keep
    draw = inverse_cdf(probs / probs.sum(-1, keepdim=True))
    return indices.gather(-1, draw.unsqueeze(-1)).squeeze(-1)


class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_fused_film_with_arbitrary_module_name(self):
        self.assertEqual(judge_failures('film', FusedFiLM), [])

    def test_adaln_with_separate_renamed_projections(self):
        self.assertEqual(judge_failures('adaln', RenamedAdaLN), [])

    def test_cache_with_private_storage(self):
        self.assertEqual(judge_failures('kv_cache', PrivateCache), [])

    def test_relu_accepts_maximum_subgradient(self):
        self.assertEqual(judge_failures('relu', lambda x: torch.maximum(x, torch.zeros_like(x))), [])

    def test_linear_accepts_differentiable_nonleaf_weights(self):
        class NonleafLinear:
            def __init__(self, in_features, out_features):
                self.weight = torch.randn(out_features, in_features, requires_grad=True) / in_features**0.5
                self.bias = torch.zeros(out_features, requires_grad=True)
            def forward(self, x):
                return x @ self.weight.T + self.bias
        self.assertEqual(judge_failures('linear', NonleafLinear), [])

    def test_mha_does_not_require_distinct_initial_queries(self):
        base = load_solution('mha')
        class ZeroQueryMHA(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                nn.init.zeros_(self.W_q.weight)
                nn.init.zeros_(self.W_q.bias)
        self.assertEqual(judge_failures('mha', ZeroQueryMHA), [])

    def test_valid_alternative_sampling_algorithms(self):
        for task, fn in [('temperature_sampling', alternative_temperature),
                         ('top_k_sampling', compact_top_k), ('top_p_sampling', alternative_top_p)]:
            with self.subTest(task=task):
                self.assertEqual(judge_failures(task, fn), [])

    def test_sampling_rejects_greedy_and_uniform_answers(self):
        for task, arg in [('temperature_sampling', 'temperature'), ('top_k_sampling', 'k'), ('top_p_sampling', 'p')]:
            with self.subTest(task=task, mistake='greedy'):
                self.assertTrue(judge_failures(task, lambda logits, **kw: logits.argmax(-1)))
            with self.subTest(task=task, mistake='uniform'):
                self.assertTrue(judge_failures(task, lambda logits, **kw: torch.randint(logits.shape[-1], logits.shape[:-1])))

    def test_film_rejects_nonlinearity_in_x(self):
        class Nonlinear(FusedFiLM):
            def forward(self, x, cond):
                return super().forward(x.square(), cond)
        self.assertTrue(judge_failures('film', Nonlinear))

    def test_film_rejects_condition_shared_across_batch(self):
        class SharedCondition(FusedFiLM):
            def forward(self, x, cond):
                return super().forward(x, cond[:1].expand_as(cond))
        self.assertTrue(judge_failures('film', SharedCondition))

    def test_adaln_rejects_missing_normalization(self):
        class Unnormalized(RenamedAdaLN):
            def forward(self, x, cond):
                return self.scale(cond)[:, None] * x + self.shift(cond)[:, None]
        self.assertTrue(judge_failures('adaln', Unnormalized))

    def test_cache_rejects_wrong_values_and_stale_clear(self):
        class WrongValues(PrivateCache):
            def append(self, k, v):
                super().append(k, k)
        class StaleClear(PrivateCache):
            def clear(self):
                pass
        for cls in (WrongValues, StaleClear):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(judge_failures('kv_cache', cls))

    def test_attention_rejects_passthrough_values(self):
        def passthrough(Q, K, V):
            return V + (Q.sum() + K.sum()) * 0
        for task in ('causal_attention', 'linear_attention'):
            with self.subTest(task=task):
                self.assertTrue(judge_failures(task, passthrough))

    def test_lora_allows_frozen_base_buffer(self):
        cls = load_solution('lora', [
            ('self.W = nn.Parameter(torch.empty(out_features, in_features))',
             'self.register_buffer("W", torch.empty(out_features, in_features))'),
        ])
        self.assertEqual(judge_failures('lora', cls), [])

    def test_sigmoid_schedule_rejects_linear_interpolation(self):
        def linear(t, beta_start, beta_end, gamma=10.0):
            return beta_start + (beta_end - beta_start) * t
        self.assertTrue(judge_failures('snr', linear))

    def test_relu_reference_preserves_float_dtype_and_handles_negative_infinity(self):
        relu = load_solution('relu')
        for dtype in (torch.float16, torch.float32, torch.float64):
            x = torch.tensor([float('-inf'), -1., 0., 2.], dtype=dtype)
            out = relu(x)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(torch.equal(out, torch.tensor([0., 0., 0., 2.], dtype=dtype)))

    def test_adaln_zero_rejects_dead_zero_branch(self):
        cls = load_solution('adaln_zero', [
            ('return gate * ((1 + gamma) * x_norm + beta)',
             'return gate * (gamma * x_norm + beta)'),
        ])
        self.assertIn('Gate controls the residual branch', judge_failures('adaln_zero', cls))

    def test_top_p_rejects_extra_token_at_exact_boundary(self):
        def extra_token(logits, p):
            probs, indices = logits.softmax(-1).sort(descending=True, dim=-1)
            keep = torch.cat((torch.ones_like(probs[..., :1], dtype=torch.bool),
                              probs.cumsum(-1)[..., :-1] <= p), -1)
            probs = probs * keep
            draw = inverse_cdf(probs / probs.sum(-1, keepdim=True))
            return indices.gather(-1, draw.unsqueeze(-1)).squeeze(-1)
        self.assertIn('Stops at an exact cumulative boundary', judge_failures('top_p_sampling', extra_token))

    def test_conditioners_may_normalize_the_condition(self):
        class NormalizedFiLM(FusedFiLM):
            def forward(self, x, cond):
                cond = nn.functional.layer_norm(cond, (cond.shape[-1],))
                return super().forward(x, cond)
        class NormalizedAdaLN(RenamedAdaLN):
            def forward(self, x, cond):
                cond = nn.functional.layer_norm(cond, (cond.shape[-1],))
                return super().forward(x, cond)
        for task, cls in [('film', NormalizedFiLM), ('adaln', NormalizedAdaLN)]:
            with self.subTest(task=task):
                self.assertEqual(judge_failures(task, cls), [])

    def test_adaln_zero_rejects_shared_conditions_and_unused_modulation(self):
        shared = load_solution('adaln_zero', [
            ('gamma, beta = self.cond_mlp(cond).chunk(2, dim=-1)',
             'cond = cond[:1].expand_as(cond); gamma, beta = self.cond_mlp(cond).chunk(2, dim=-1)'),
        ])
        unused = load_solution('adaln_zero', [
            ('return gate * ((1 + gamma) * x_norm + beta)', 'return gate * x_norm'),
        ])
        for cls in (shared, unused):
            with self.subTest(cls=cls):
                self.assertIn('Nonzero per-sample modulation', judge_failures('adaln_zero', cls))

    def test_adaln_zero_gate_can_learn_at_initialization(self):
        layer = load_solution('adaln_zero')(hidden_dim=8, cond_dim=4)
        x = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
        cond = torch.randn(2, 4)
        target = nn.functional.layer_norm(x, (8,))
        out = layer(x, cond)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        (out - target).square().mean().backward()
        self.assertGreater(layer.gate_proj.bias.grad.abs().sum().item(), 0)

    def test_top_p_stops_when_prefix_reaches_p_exactly(self):
        fn = load_solution('top_p_sampling')
        # Non-tied probabilities .5, .25, .25: p=.5 must only keep token 0.
        torch.manual_seed(0)
        draws = fn(torch.tensor([2., 1., 1.]).log().expand(512, -1), p=0.5)
        self.assertTrue(torch.equal(draws, torch.zeros_like(draws)))


if __name__ == '__main__':
    unittest.main()
