# Notebook and judge contracts

The audit covers all 37 templates (`00`–`36`), all 36 reference solutions,
and their task definitions. Public entry points and explicitly documented
parameter attributes are part of an exercise's interface. Private helper and
module names are not. Tests should compare mathematical behavior wherever
possible, and any parameters read or modified by a test must be documented.

| Notebook | Audit outcome |
| --- | --- |
| 00 Welcome | Explain public interfaces, implementation freedom, and limits of automated grading. |
| 01 ReLU | Accept subgradients in [0, 1] at zero; remove hardware timing gate; preserve reference dtype and handle negative infinity. |
| 02 Softmax | Existing contract aligned; add a non-last-axis numerical check. |
| 03 Linear | Keep documented weight/bias interface; test derivatives without requiring leaf-tensor `.grad` storage. |
| 04 LayerNorm | Specify population variance and epsilon placement. |
| 05 Attention | Contract aligned; remove an example that printed an unchecked literal `True`. |
| 06 Multihead attention | Remove random-initialization assumption; check cross-attention numerically; explain public projections and differentiability. |
| 07 BatchNorm | Contract aligned; use population std in the reference demonstration. |
| 08 RMSNorm | Already aligned; no change required. |
| 09 Causal attention | State Q/K versus V widths and autograd; verify numerical masked attention. |
| 10 GQA | Preserve existing numerical oracle; clarify grouping, divisibility, shapes, and implementation freedom. |
| 11 Sliding window | Explain symmetric windows, value width, autograd, and dense-mask cost; verify intermediate window values. |
| 12 Linear attention | Specify denominator stabilization; replace timing gate with a numerical feature-map oracle. |
| 13 GPT-2 block | Preserve numerical oracle; disclose dimensions, exact GELU, and dropout convention; check gradients only on trainable parameters. |
| 14 GELU | Clearly require exact erf formula; label tanh approximation as comparison only. |
| 15 SiLU | Contract aligned; correct the negative-tail explanation. |
| 16 SwiGLU | Clarify the already-public projections and constructor-defined output width. |
| 17 LoRA | Document public tensors/scalars, shapes, and rank zero; allow a frozen base buffer; clarify unchanged base versus identity mapping. |
| 18 DoRA | Document public tensors/scalars, shapes, scaling, row normalization, and autograd; preserve the nonzero-update oracle. |
| 19 AdaLN | Remove private module names and misleading identity test; check normalization and featurewise modulation behavior. |
| 20 AdaLN-Zero | Define the simplified residual branch; fix dead zero initialization and unused gate; verify gate control, learning, and nonzero per-sample modulation. |
| 21 FiLM | Accept fused or separate projections with arbitrary private names; check featurewise affine behavior and batch/token independence. |
| 22 RoPE | Clarify even head width and adjacent-pair/position conventions. |
| 23 KV cache | Remove storage-name constraints; document public `seq_len`; check both K and V, empty state, clear, and reuse. |
| 24 Sigmoid schedule | Align hint with the beta formula; use a real midpoint in examples; check nondefault steepness numerically. |
| 25 K-Means | Clarify that assignments correspond to final returned centroids. |
| 26 KNN | State Euclidean distance, label domain, shapes, and valid k. |
| 27 MLP backward | Explicitly document dictionary keys, parameter shapes, and mean reduction. |
| 28 Greedy | Already aligned; no change required. |
| 29 Beam search | Explain why prefix-independent input also admits the greedy optimum. |
| 30 Temperature sampling | Grade the distribution, not one RNG algorithm's seeded draw; document shapes and valid inputs. |
| 31 Top-k sampling | Accept compact-vocabulary sampling; grade distribution/support; document k clipping and ties. |
| 32 Top-p sampling | Grade distribution/support; stop at exact cumulative equality; document shapes, ties, and p range. |
| 33 PPO | Clarify matching input shapes and scalar mean output. |
| 34 DPO | Clarify sequence log-probabilities and favorable-gap test wording. |
| 35 GRPO | Specify population std, epsilon placement, singleton behavior, and sample averaging. |
| 36 NumPy XOR | Make demonstrations reproducible and explain initialization sensitivity; name the output-type check accurately. |

## Validation

From the repository root, with PyTorch and NumPy installed:

```sh
python -m unittest discover -s tests -v
```

The regression suite loads only imports and definitions from every solution,
then executes the actual grader snippets. It never calls notebook `check()` or
writes learner progress. It tests equivalent implementations (renamed/fused
conditioners including normalized conditions, private cache storage, alternative sampling algorithms) and
incorrect variants (wrong formulas, shared batch conditions, wrong cache
values, greedy/uniform sampling, and a dead zero-initialized branch).

Sampling tests alternate distinct input distributions across 8,192 trials per
row and use a fixed seed for reproducibility. An absolute frequency tolerance
of 0.04 is over seven worst-case binomial standard errors; disallowed tokens
must never appear. A seed does not require identical tokens across algorithms.
These are statistical checks, not a proof of randomness or complete correctness.

The tests do not inspect source to enforce every educational restriction (for
example, avoiding built-in activations), nor do they infer asymptotic complexity
from a hardware-dependent timer. Those requirements remain in the exercises.
