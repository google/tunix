# Shared-policy swarm validation

Date: November 11, 2026. Scope: the first local-transport slice of the
[system design](swarm_training_design.md). Entry point and folder rationale:
[shared math recipe](../examples/swarm_math/README.md).

Initial implementation acceptance: **292 tests passed**, including **43 swarm tests**, with
**58 additional subtests passed**. The real-tokenizer check was enabled. The
three existing HTTP OPD tests required a separate run with localhost binding
permitted; they passed there. The other 289 tests passed in the sandbox.

After merging `origin/main` at `73b21537a` (merge `9e3547c29`), all **768 selected
regression tests** passed across the suite and targeted reruns. The local native
packing extension was rebuilt from the merged source; its **157 batch-assembly
tests** and **46 swarm/OPD training tests** passed after rebuilding. The swarm
path now uses `GRPOConfig`, shared loss/input registration, and plain dataclass
payloads. The two-round CPU recipe retained a parameter L2 change of `0.0648973`
and maximum sampler/actor logprob error of `3.21e-7`.

## Evidence and boundaries

| Layer | Executed check | Result |
| --- | --- | --- |
| Reference control flow | Execute the pinned upstream orchestration and verdict functions with scripted calls; compare exact role order, prompts, final answer and validity | Approval, rejection, whitespace, missing/unknown verdict, both tags, reversed tags and Chinese-character validity agree |
| Credit | Native NumPy role/episode credit against extracted upstream PyTorch advantage function | Agrees within `2e-6` for mixed rewards, unequal call counts, invalid-call penalties, missing roles, multiple tasks and singletons |
| Zero variance | Constant decimal scores and unequal call counts | Explicit zero advantage; avoids an upstream float32 rounding artifact |
| Loss and gradients | Native JAX GRPO versus original PyTorch dual-clip loss with independent autoregressive scoring | Loss and all parameter gradients agree within `2e-6`; finite-difference checks also pass |
| Masks | Active EOS equal to pad, padding rows, loss-masked observations, separate attention masks | Correct gradients and preserved context; padded assembler checked in both compiled and Python paths |
| Categorical training | Two real native optimizer updates under role normalization and episode baseline | Each updated parameter tensor agrees with the upstream full-batch gradient oracle within `2e-6`; recorded updates reduce that batch's surrogate loss |
| Qwen3 integration | Random 1-layer Qwen3, 16 hidden dimensions, 8-token fixture vocabulary, native KV sampler and native trainer | Two successful updates; sampler/actor logprob maximum error `4.77e-7`; parameter L2 changes approximately `0.315` and `0.111` |
| Real Qwen tokenizer | Official pinned Qwen3-4B tokenizer/config, unchanged Solver prompt and golden token IDs | Exact 101-token prompt; non-thinking suffix and middle truncation verified |
| Artifacts | CPU CLI output, exact trace JSONL and actual native checkpoint writer | Config, report, parameters and model/optimizer checkpoint emitted |
| Input boundaries | Duplicate IDs, mixed/stale policy versions, partial groups, bad sampler responses, logprob mismatch and all-zero signal | Fail before update, or skip optimizer/publication for zero signal |

These are correctness and integration checks. The pretrained Qwen3-4B weights
were not loaded or trained on this CPU host. Random tiny-Qwen and categorical
accuracy are not benchmark results or evidence of collaboration gains. No remote
episode RPC, Raiden, accelerator throughput, convergence or recovery result is
claimed. The single-Solver evaluation is a diagnostic, not a matched-budget
trained baseline.

## Independent reference

DrMAS commit: `c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9`.
[Original repository](https://github.com/langfengQ/DrMAS/tree/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9).
The test fixture contains exact extracted functions and source-file hashes;
tests execute those functions instead of reimplementing a second NumPy formula.
[Attribution and copied files](../examples/swarm_math/THIRD_PARTY.md).

The port preserves `ddof=1`, epsilon `1e-6`, singleton mean/std `0/1`, and the
source's distinction between per-role call normalization and per-task episode
normalization. The shared variant is explicitly enabled; the original launcher
defaults to separate models. The native loss matches the full-batch token-mean
objective, including positive/negative clipping and negative dual clipping.
Upstream microbatch averaging may weight unequal token counts differently; the
native trainer intentionally normalizes one accumulated numerator by one global
active-token denominator. This is not a bitwise reproduction of the upstream
distributed optimizer schedule.

One explicit numerical correction is applied: repeated equal float32 decimal
scores can have a rounded mean and nonzero computed standard deviation. For
example, the pinned PyTorch function on this host produces an advantage around
`0.00739` for eight identical `-0.1` rewards. Equal-score multi-call groups now
return zero, and episode means accumulate in float64 before conversion to
float32. The source singleton convention remains unchanged. Zero-signal rounds
also skip AdamW, preventing weight decay from creating an unintended update.

## Fixes found by integration

1. `PaddedBatchAssembler` discarded `completion_attention_mask`. It now pads
   and preserves that field independently of `completion_mask` through both
   implementations.
2. Native sampling omitted the sampled EOS. An opt-in `include_eos` flag
   preserves termination tokens and aligned scores, including immediate EOS;
   existing sampler callers keep their previous default behavior.
3. `nnx.clone` shares array buffers. Local serving snapshots now copy storage
   and retain each destination leaf's sharding, preventing donated optimizer
   buffers from invalidating inference state.
4. Constant decimal rewards could create artificial advantages. Explicit
   equal-score handling and wider episode-mean accumulation remove this signal.

## Reproduce

The isolated local environment used Python 3.13.1, JAX/JAXlib 0.11.0, Flax
0.12.8, Optax 0.2.8, PyTorch 2.15.0, Transformers 4.57.1 and safetensors 0.7.0
on one Apple Silicon CPU device. No TPU/GPU was available.

For the distributed-engine regression tests, generate RPC stubs using the
[distributed setup instructions](../tunix/dist_rl/distributed/examples/README.md)
and rebuild native extensions after pulling changes to their C++ sources.

```bash
JAX_PLATFORMS=cpu python -m pytest tests/dist_rl/swarm -q

JAX_PLATFORMS=cpu python -m examples.swarm_math.train \
  --backend cpu --output /tmp/new-swarm-cpu-run

# Optional: point at the downloaded tokenizer-only snapshot or full snapshot.
TUNIX_SWARM_QWEN_TOKENIZER=/path/to/pinned/tokenizer \
  JAX_PLATFORMS=cpu python -m pytest tests/dist_rl/swarm/recipe_test.py -q
```

The real tokenizer check skips without that environment variable; all core
numerical and training tests are offline. Existing sampler, batch assembly,
payload builder and OPD gradient/training regressions are included in the
acceptance run because the mask and EOS changes touch shared infrastructure.

```bash
TUNIX_SWARM_QWEN_TOKENIZER=/path/to/pinned/tokenizer JAX_PLATFORMS=cpu \
  python -m pytest tests/dist_rl/swarm \
  tests/dist_rl/rollout/vanilla_sampler_test.py \
  tests/dist_rl/orchestrator/batch_assembly_test.py \
  tests/dist_rl/orchestrator/payload_builder_test.py \
  tests/dist_rl/distillation/on_policy_test.py \
  tests/dist_rl/distillation/opd_gradient_test.py \
  tests/dist_rl/distillation/opd_training_test.py -q
```

The reference training corpus was also prepared successfully from
`aaabiao/dapo_filter` revision `0b10fea35812902fe4f6985f2064424ec460f9c0`:
13,044 raw rows, 12,613 retained unique problems, 10 conflicting problem groups
excluded, and a deterministic split of 12,357 train / 256 dev. The data manifest
records excluded problem hashes and SHA-256 hashes for both split files.
This development split is an explicit adaptation; it does not consume the
original recipe's published benchmark validation questions.

## Next acceptance gate

1. Run the pinned 4B profile on accelerators; measure memory/throughput and
   sampler/actor logprob error before longer training.
2. Add worker-side episode collection and the distributed engine adapter; verify
   the same loss/gradient invariants with actual transport and weight sync.
3. Implement committed round/data-cursor resume and failure recovery, including
   failures between optimizer update and policy publication.
4. Run role/global normalization, untrained-team and matched-budget single-agent
   baselines on locked tests with multiple seeds. Report accuracy, tokens, calls
   and uncertainty before drafting a performance claim.
