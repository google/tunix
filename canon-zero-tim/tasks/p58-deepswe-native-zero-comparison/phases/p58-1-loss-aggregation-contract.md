# P58.1 — Loss aggregation and shared-recipe contract

Status: completed locally on 2026-08-21. This phase is closed by the pinned
exact-image marker recorded in `../state.md`; no target TPU claim is implied.

## Purpose

Resolve the DeepSWE loss-normalization ambiguity before implementing or
launching either comparison arm. The exit gate certifies a mathematical and
distributed-reduction contract, not training quality.

## Tunix implementation under review

For trajectory `i`, completion token `t`, PPO surrogate token loss
`ell[i,t]`, completion mask `m[i,t]`, raw batch row count `B_raw`, effective
row count `B_eff = sum_i 1[sum_t m[i,t] > 0]`, and fixed response width `L`,
the pinned quality-fix `sequence-mean-token-scale` contract computes:

```text
L_scale = sum_i,t m[i,t] * ell[i,t] / (B_eff * L)
```

With the P58 recipe, `B_raw=128` over the complete update and `L=16,384`, so
the intended full-update objective is:

```text
L_scale = sum_i,t m[i,t] * ell[i,t] / (B_eff * 16,384)
```

This is not an average over each sequence's actual valid-token length. A
longer completion contributes more total gradient because it contains more
valid policy tokens, while every valid token receives the same fixed `1/L`
scale. That is the intended Dr.GRPO length-normalization behavior: it removes
the incentive created by normalizing every sequence by its own length.

For comparison:

| Mode | Formula | Consequence |
|---|---|---|
| `token-mean` | `sum ell / sum valid_tokens` | fixed token weighting but a batch-dependent denominator |
| `sequence-mean-token-mean` | mean of each sequence's actual-token mean | each sequence has equal total weight; short-sequence tokens are heavier |
| `sequence-mean-token-scale` | mean of token sums divided by fixed `L` | fixed token scale; sequence contribution grows with valid length |
| `seq-mean-token-sum` | mean of valid sequences' token sums | same direction as scale but approximately `L` times larger gradient |

The PPO token loss is formed before aggregation: Tunix computes the clipped
surrogate from the sequence-level RLOO advantage broadcast across completion
tokens, applies the completion mask in aggregation, and applies sampler-IS
weights only when that optional correction exists. P58 disables sampler-IS,
so no extra token weighting is admitted.

## Match assessment

| Reference | Observed contract | Match? |
|---|---|---|
| Together DeepSWE algorithm description | divide surrogate loss by maximum context length | yes |
| DeepSWE Hugging Face model card | same fixed maximum-context normalization | yes |
| Tunix `yuxzhang/deepswe-quality-fix@023978b...` notebook | default `sequence-mean-token-scale` | yes |
| Current Tunix DeepSWE notebook and P34 contract | default/signed `sequence-mean-token-scale` | yes |
| Public rLLM `train_deepswe_32b.sh` | `seq-mean-token-sum` | no, unless an undocumented external scale compensated it |
| Tunix `run_deepswe_disagg_v5p_32.sh` | `seq-mean-token-mean` spelling/path | no; this is not the notebook launcher used by the prior JobSet |

Primary-source references:

- https://www.together.ai/blog/deepswe
- https://huggingface.co/agentica-org/DeepSWE-Preview
- https://github.com/agentica-project/rllm/blob/main/examples/swe/train_deepswe_32b.sh
- https://github.com/rllm-org/rllm/issues/354

The issue above records the public script/blog inconsistency and was closed
without an authoritative reconciliation. P58 therefore distinguishes
“algorithm-description match” from “historical launcher exactness.”

## Two implementation hazards to close

### 1. Compact-filtered versus structurally invalid rows

The pinned quality-fix implementation divides by the count of non-empty policy
mask rows. The current operator branch divides by the static row count, so a
legitimate compact-filtered row dilutes the gradient. The current `origin/main`
aggregate again excludes empty rows but also contains unrelated segmented
packing work. P58 imports only the minimal effective-row denominator behavior;
it does not merge `main` or enable packing.

P58 retains the official compact filter. A trajectory whose terminal status is
`MAX_STEPS_REACHED`, `MAX_CONTEXT_LIMIT_REACHED`, `TIMEOUT`, `ENV_TIMEOUT`,
`MODEL_TIMEOUT`, or `REWARD_TIMEOUT` remains in the raw trajectory journal but
receives an all-zero policy-loss mask. That row is expected and excluded from
`B_eff`. In contrast, a missing, duplicated, parser-invalid, or structurally
empty trajectory is a hard input failure. The batch must always contain exactly
128 raw records. If `B_eff=0`, the step logs a no-signal receipt and performs no
optimizer commit or resampling.

### 2. Implicit norm and distributed accumulation

`grpo_loss_fn` currently calls the aggregator without `norm`, so `L` silently
defaults to `per_token_loss.shape[-1]`. That is correct only while every
unpacked training tensor is padded to exactly 16,384. Packing, truncation, or
a changed compiled width could silently change the objective.

P58 must bind `loss_norm=16,384` explicitly and assert the compiled response
width equals 16,384 before the first gradient. Eight raw-equal trajectory
micro-batches of 16 must reproduce the single 128-row reference loss and
gradient within the registered precision tolerance. Because compact filtering
can make per-microbatch effective counts unequal, accumulation must weight each
microbatch numerator by its `B_eff` and divide once by global `B_eff`; an
unweighted mean of local means is forbidden. The DP8-sharded result must
reproduce that same global objective.

## Local tests required for exit

1. Formula oracle with unequal completion lengths proves exact fixed-16K
   normalization and distinguishes all four modes in the table above.
2. Partial and all-compact-filtered mask cases prove effective-row aggregation,
   status attribution, and all-filtered no-commit behavior; separate malformed
   rows fail closed.
3. Eight 16-row raw-equal micro-batches with intentionally unequal effective
   counts match one 128-row loss and gradient.
4. A missing raw micro-batch or an unweighted mean-of-means negative control
   fails closed.
5. DP8-sharded and unsharded CPU/simulated-array reductions match for numerator,
   denominator, scalar loss, and gradient checksum.
6. Runtime receipt prints `loss_agg_mode`, explicit `loss_norm`, observed
   tensor width, raw/effective/compact-filtered/structurally-invalid row counts,
   valid-token count, filter-status histogram, accumulation depth, and DP size
   before optimizer admission.
7. Negative controls prove existing P34, P44, and P46 contracts are unchanged.

## Exit decision

Keep `sequence-mean-token-scale` only if all tests above pass with explicit
`L=16,384`. Otherwise P58 remains blocked; changing to token mean or token sum
is a new algorithm and requires a separately approved phase revision.

Passing P58.1 proves normalization and reduction wiring only. It does not prove
rollout quality, convergence, native mismatch dose, zero-TIM exactness, HBM
capacity, sandbox throughput, or 128-chip execution.

## Completion receipt

The implementation binds `loss_scale_factor=16384`, asserts the compiled
response width, excludes all-zero policy-mask rows, and carries the unreduced
numerator plus effective-row denominator through eight trajectory
micro-batches. Both the stock trainer and canonical segmented path discard a
globally zero-denominator transaction without optimizer mutation. Tests cover
unequal per-microbatch effective counts, all-filtered no-commit, fixed-norm
drift, DP/reduced aggregate equality, and journal continuation after a skipped
commit.

Pinned-image terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```
