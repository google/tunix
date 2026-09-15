# Qwen3.5-35B-A3B distributed GRPO on TPU v5p — reproduction report (v6)

Supersedes `qwen35_report_v4.md` / `_v5.md`. Written against
`gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8`
(Yixuan's e2e image) plus a 7-file overlay.

Everything here was measured on cluster `bodaborg-v5p-nap`, project
`cloud-tpu-shared-capacity`, region `europe-west4`, namespace `trellis`.

---

## 0. TL;DR for someone reproducing this

```bash
export KUBECONFIG=~/.kube/config.cloud-tpu-shared-capacity.europe-west4.bodaborg-v5p-nap
cd ~/git/tunix && ./build_qwen35_overlay.sh            # builds + pushes the overlay image
cd tunix/experimental/examples/math_gsm8k_dist
MAX_STEPS=2 RUN_TAG=smoke ./run_qwen35_repro.sh start  # smoke test first
./run_qwen35_repro.sh stop
```

Two things to know before you start:

1. **`$USER` is `igorts_google_com` on these VMs and underscores are illegal in
   k8s names.** `run_qwen35_repro.sh` sanitizes it (`cut -d_ -f1`). If you
   invoke the launcher directly you must do the same or every `kubectl apply`
   fails.
2. **`KUBECONFIG` is not the default path.** Without the export above, `kubectl`
   tries `localhost:8080` and refuses the connection.

---

## 1. What actually limits this run

This is the headline result and it is not what I expected going in.

Measured on a clean 2-step run (8-chip trainer, 32 rollout chips, packing on):

| Phase | Step 0 | Step 1 (steady state) |
|---|---|---|
| Weight sync | 258.25 s + 234.55 s | 233.39 s |
| Rollout + train | ~460 s (incl. 149 s compile) | ~160 s |
| **Total `step_time`** | **696.45 s** | **393.46 s** |

**Weight sync is 59% of a steady-state step.** Sequence packing cut a step from
32 microbatches to 5 and bought about 4 seconds, which means gradient
computation was never the bottleneck and never was. Of the ~160 s of non-sync
time, essentially all of it is rollout generation.

### The 100-steps-in-2-hours target

2 hours over 100 steps is a 72 s/step budget. Weight sync alone is 233 s. Even
with rollout and training at literally zero, 100 steps costs **6.5 hours**. The
target is not reachable by tuning batch shapes or by scaling the trainer for
throughput; it requires making weight sync itself cheaper, and probably
overlapping it.

Options, in descending order of expected payoff:

- **Overlap sync with rollout (`max_staleness > 0`).** Currently
  `max_staleness=0`, so generation waits for the sync to land. Allowing one
  step of staleness lets rollout run against the previous policy while the
  sync proceeds, hiding most of 233 s. This changes the algorithm from strictly
  on-policy GRPO to slightly off-policy — a legitimate and common choice, but a
  *semantic* change, not a tuning knob. Flagging rather than doing it.
- **Fewer rollout destinations.** Raiden fans the model out to every rollout
  worker; if the cost is per-destination, 8 workers is 8 copies. Memory on the
  rollout side is slack (see §5), so fewer, busier replicas may cost little
  generation throughput. Untested.
- ~~More trainer hosts.~~ **Measured, and it goes the wrong way — see §1.1.**

### 1.1 Scaling the trainer makes it *worse* (measured)

The intuition was that more trainer hosts would give Raiden's D2H more parallel
egress. It does not. Same run, same everything else, trainer slice swapped:

| Trainer | Chips | Hosts | FSDP | Weight sync |
|---|---|---|---|---|
| `tpuv5p:2x2x2` | 8 | 2 | 8 | **233–258 s** |
| `tpuv5p:2x2x4` | 16 | 4 | 16 | **345 s** |

Doubling the trainer made the dominant cost ~48% worse. More FSDP shards means
more, smaller per-shard transfers, and the per-shard overhead dominates any
gain in parallel egress.

This is convenient, because 8 chips is also the memory floor: full-parameter
Adam on 35B needs ~560 GB of params + grads + fp32 moments + master weights
against 95 GB × 8 = 760 GB. **8 trainer chips is both the minimum that fits and
the fastest for weight sync. Leave `TRAINER_TPU_SLICE` alone.**

Trainer chip count is therefore *not* the lever. Do not spend capacity there.

---

## 2. Image and overlay

Base: `yixuannwang_google_com-runner:yixuann-e2e-0912head-v8` — jax 0.11.0,
libtpu 0.0.44, vllm 0.28.1rc1 with the TPU backend, MaxText at `/app/maxtext`,
the Raiden FFI bridge, and a tunix checkout at `/app/tunix` that
`PYTHONPATH=/app` makes win over the pip-installed copy.

The overlay patches **7 files — 6 under `/app/tunix`, 1 under `/app/maxtext`**.
Keep it this small: the base image's tunix is a different lineage from
`github.com/google/tunix` HEAD, so copying the whole tree in would silently
downgrade Yixuan's fixes.

| # | File | Why |
|---|---|---|
| 1 | `maxtext/src/maxtext/training_engine/maxtext_engine.py` | MaxText PR #5219 — stop per-microbatch recompiles |
| 2 | `orchestrator/distributed_rl_engine.py` | round-robin rollout routing |
| 3 | `examples/math_gsm8k_dist/run_gsm8k_dist_grpo.py` | stop setting `prefix_hash` |
| 4 | `examples/math_gsm8k_dist/gsm8k.py` | log completions at INFO (diagnostic) |
| 5 | `examples/common/run_rollout_node.py` | `data_parallel_size` for the `vllm` sampler |
| 6 | `utils/maxtext_utils.py` | let `max_seq_token_per_tpu` raise `max_target_length` |
| 7 | `examples/common/run_trainer_node.py` | plumb `--max_seq_token_per_tpu` |

The Dockerfile asserts each patch is present at build time, so a moved upstream
path fails the build instead of the job.

### 2.1 The recompilation fix (MaxText PR #5219)

**Symptom:** the trainer recompiled the entire 35B graph on every microbatch.

**Cause:** `RLTrainerPayload` is a `flax.struct.dataclass` and `metadata` is
declared `pytree_node=False`, i.e. *static* aux data in the PyTreeDef. Its keys
change from microbatch to microbatch, so every microbatch presented a new
treedef and missed the jit cache.

`MaxTextTrainingEngine._prepare_batch` already excluded `"metadata"` on its
fallback dataclass→dict path, but forwarded the payload intact when a
`_gen_model_input_fn` was set — which is exactly our path
(`_algo_model_input`). PR #5219 clears it to `{}` via `dataclasses.replace`
before either branch.

Cherry-picked into `~/git/maxtext` as `96ccd858b` (`git cherry-pick -x
59420af4`). **Verified:** step 0 compiles once (149 s) then runs 31 × 2.45 s;
step 1 runs straight through.

> An earlier iteration of this work patched the equivalent fix into tunix's
> `orchestrator/algorithm_adapter.py`. That has been reverted in favour of
> #5219, which is merged upstream and sits one layer down. Do not reapply it.

### 2.2 A note on `639cbe7be`

The image's `maxtext_engine.py` is byte-identical to our local checkout *except*
for local commit `639cbe7be` (drain async checkpoint saves under Pathways).
Since the overlay ships the local file wholesale, that commit now rides along
with #5219. It is **inert here** — with `CHECKPOINT_SAVE_INTERVAL_STEPS=0` the
save path never fires — and it was kept deliberately: shipping a file that
matches no git commit is worse for reproducibility than shipping one extra
unreachable change.

---

## 3. Sequence packing

**Symptom:** packing appeared to be enabled but did nothing.

**Cause, and it is a nasty one:** `validate_packing_budget` requires
`max_seq_token_per_tpu >= max_prompt_length + max_response_length`. Stock
`maxtext_utils.build_maxtext_config` *pins* MaxText's `max_target_length` to
that same sum. So the budget is squeezed to exactly one maximal trajectory from
both sides at once, and a "packed" row holds exactly one sequence.

**Fix (overlay files 6 and 7):** add a `max_seq_token_per_tpu` kwarg to
`build_maxtext_config` and take `max(max_seq_token_per_tpu, max_prompt +
max_response)` for `max_target_length`, then plumb `--max_seq_token_per_tpu`
through `run_trainer_node.py`. (`k8s_launcher.sh` already passed the value to
the orchestrator — only the trainer was missing it.)

**Measured**, same step, 256 rollouts:

| `MAX_SEQ_TOKEN_PER_TPU` | trajectories/microbatch | microbatches/step |
|---|---|---|
| 1024 (the floor) | 8–16 (~1.14/row) | 32 |
| 4096 | 61, 55, 57, 48, 35 (~7.6/row) | **5** |

Worth doing, but see §1: it bought ~4 s of a 393 s step.

---

## 4. `train_micro_batch_size` propagation

A known failure mode, and worth stating explicitly because a teammate hit it:
`GRPOAdapter` used to default `train_micro_batch_size` to 1, and
`rl_program.py:206` reads it back via `getattr(algo, "train_micro_batch_size",
1)` — so the flag was validated and logged but never applied, and packing
collapsed to one sequence per pass.

**This is already fixed in the base image** (`run_gsm8k_dist_grpo.py:249`); it
is not something this overlay adds. Two confirmations on our runs:

- Trainer side, by construction: `maxtext_utils.py:161` raises if
  `train_micro_batch_size % mesh_fsdp != 0`. With `mesh_fsdp=8`, a leaked
  default of 1 gives `1 % 8 = 1` → hard startup failure. We start fine, so 8
  arrives. `per_device_batch_size = 8/8 = 1.0`.
- Orchestrator side, logged: `train_micro_batch_size=8`, `pack_size: 8`.

Note the asymmetry: the trainer *crashes* on a bad value, so a silently
sub-optimal step time from this bug comes from the orchestrator/packing half.

---

## 5. Topology, and why the architecture is left alone

**Trainer:** Pathways, `tpuv5p:2x2x2` = 8 chips, FSDP=8 / TP=1 / EP=1.
v5p packs 4 chips per host, so this is 2 hosts.

**Rollout:** 8 × `tpuv5p:2x2x1` = 4 chips each, 32 total, each filled as
**TP=2 × DP=2**.

TP must stay at 2, and this is the one place where a tuning knob would have
quietly changed the science:

- At **TP=4**, `maxtext_utils` replicates `base_num_kv_heads` 2 → 4, and GMM_v2
  pads the MoE MLP dim 512 → 1024 (512/4 is not a multiple of 2·128). The MoE
  experts are ~32B of this 35B model, so that padding roughly doubles what the
  trainer holds — and it means training a *different model*.
- At **TP=2**, 512/2 = 256 is already aligned and `num_kv_heads` is untouched.
  Verified in-run: `kv_heads=2`, `moe_mlp_tp_size=2`,
  `padded_base_moe_mlp_dim=512`.

`base_num_kv_heads: 2` is therefore what pins the rollout's atomic unit at 2
chips; the only free variable on that side is replica count.

A sizing note: "A3B" (3B active) cuts compute, not optimizer state — all 35B
params carry it. Full-parameter Adam needs roughly 70 GB bf16 params + 70 GB
grads + fp32 moments + fp32 master ≈ 560 GB plus activations, against 95 GB ×
8 = 760 GB. 8 chips is the working floor, not a comfortable choice.

### `inprocess_vllm`, not `vllm`

**This is mandatory, and the failure is ugly.** On the plain `vllm` path the
Raiden destination registry is never populated with MaxText-named variables, so
weight sync dies in preflight before step 0:

```
WeightSyncError: round 0 ... manifest preflight failed before any destination
was quiesced (1266 problems, first: preflight: source variable
'decoder.decoder_norm.scale' has no destination counterpart)
```

1266 = 2 × 633 variables — *nothing* matched. `inprocess_vllm` logs `Using local
registration for destination metadata` and matches all 633. Root-caused by
controlled experiment: same image, same mesh, same everything else, sampler
flipped. `prefuse_moe_weights` on the trainer was investigated first and
produced a byte-identical failure, so it is not the cause (it is still set, to
match the rollout).

---

## 6. Open issues

### 6.0 SOLVED: `prefuse_moe_weights` must *differ* between trainer and rollout

**The reward-0 blocker is fixed.** One line in `k8s_launcher.sh`: the trainer
must run `--prefuse_moe_weights=false` while the rollout runs `true`. No image
change, no code change, no MaxText patch.

Before (both `true`), completions were 10–58 characters of digit soup and
`reward_mean` was exactly 0 on every step of every run. After
(`TRAINER_PREFUSE_MOE_WEIGHTS=false`), the first three logged completions of
step 0:

```
[gsm8k] completion 1/3: chars=824 reward=1.00 format_ok=True answer_ok=True extracted='13' gold='13' closes_reasoning=True
  tail='...Total Time = 2 + 6 + 5\n\nCalculation:\n2 + 6 = 8\n8 + 5 = 13\n\nSo, the total time spent waiting is 13 minutes.\n</reasoning>\n<answer>\\boxed{13}</answer><|im_end|>'
[gsm8k] completion 2/3: chars=802 reward=1.00 format_ok=True answer_ok=True extracted='13' gold='13' closes_reasoning=True
[gsm8k] completion 3/3: chars=730 reward=1.00 format_ok=True answer_ok=True extracted='7'  gold='7'  closes_reasoning=True
```

Correct arithmetic, correct `<reasoning>`/`<answer>` format, correct gold
answer, clean `<|im_end|>` stop. Full VTC reward 1.00.

Trainer-side confirmation that the flag landed:

```
[TrainerNode] MaxText param base.decoder.layers.layer_0.mlp.routed_experts.wi_0 shape=(256, 10, 2048, 512)
[TrainerNode] Trainer prepared weight sync for step 0: registered 1 work unit(s) with 633 variables
```

`wi_0` present with the unfused last dim 512 (not the fused 1024), and the
converter still hands the rollout its 633 fused variables.

Note for anyone grepping: the `[gsm8k] completion` lines are emitted on the
**rollout** pods, not the orchestrator.

The rest of this section is the derivation and the two wrong turns taken along
the way; §6.0.3 has the mechanism.

#### 6.0.0 Why this was hard to see

Every diagnostic available pointed away from the real cause:

- Raiden's weight-sync verification passed **633/633 tensors** with matching
  element counts. Its `checksums()` are per-tensor float32 **abs-sums**, which
  are invariant under permutation — and the corruption is exactly a
  permutation.
- Both ends independently compute the same `moe_mlp_tp_size=2`
  (`adapter.py:139`), so every shard-count check came back clean.
- The trainer demonstrably held the real checkpoint (`decoder_norm.scale`
  abs-sum 3232.55 over `(2048,)` = mean 1.578, not the 1.0 of a fresh
  ones-init).
- The one config the launcher forced to agree on both ends is the one that had
  to disagree, and it was forced under a comment asserting the opposite.

The two 4-chip experiments below also pointed in opposite directions; both are
recorded because one of them nearly sent me the wrong way.

#### 6.0.1 The 4-chip harness does NOT reproduce the blocker — its generation runs on dummy weights

`run_qwen35_validate_converter.sh` (§6.5) does produce digit soup:

```
Generation test after weight transfer:
['aros制备方法aros.vxarosarosabeearosabeeabeearos…inado зреcalararosabee зре vestiaros…']
validate_converter completed successfully
EXIT_CODE=0
```

but it is an artifact, and I nearly drew the wrong conclusion from it. Three
runs using **three different sync paths** —

| run | path taken | output |
|---|---|---|
| `TRAINER_PREFUSE=true` | `Qwen35MaxTextToVLLMConverter` | soup |
| `TRAINER_PREFUSE=false ROLLOUT_PREFUSE=true` | `Qwen35MaxTextToVLLMConverter` | **byte-identical** soup |
| `USE_CONVERTER=false` | `legacy tunix sync` (`transfer_state_directly`, `converter=None`) | **byte-identical** soup |

— produced the *same 300 tokens*. Three independent conversion implementations
cannot corrupt weights identically. The corroborating numbers were in the log
all along and I read past them:

```
[weight-sync] sampler.update_params via <any of the three>
              (convert+reshard+assign): 0.37 s | HBM in_use 57.95 GiB (delta +0.00) | peak 57.95 GiB
ASSIGNMENT COMPLETE: synced 785 weight leaves via sampler.update_params
```

0.37 s and **+0.00 GiB** for a 35B convert+reshard+assign. `sampler.update_params()`
writes into a state object the running vLLM model does not read, so generation
falls back to vLLM's `load_format: 'dummy'` weights — deterministic at `seed=0`,
hence bit-identical across arms.

**Retracted:** the claim that this localizes the blocker to a single process and
kills the cross-mesh hypotheses. It does not. Nothing about the distributed path
is eliminated. §6.5's harness needs `sampler.update_params` to actually land
before any of its generations mean anything; until then only its
`debug_converter=true` arm (which calls `converter.convert()` directly, see
§6.0.2) is trustworthy.

#### 6.0.2 …but the debug arm proved the prefuse premise outright

`DEBUG_CONVERTER=true` bypasses `update_params` and calls
`converter.convert(model_state, …)` directly at `validate_converter.py:950`.
With `TRAINER_PREFUSE=true` it dies immediately:

```
[rank0]:   File ".../torchax_converter/qwen35_moe.py", line 258, in _convert_moe
[rank0]:     wi_0 = jnp.transpose(routed["wi_0"], (1, 0, 2, 3))
[rank0]:   File ".../flax/nnx/statelib.py", line 258, in __getitem__
[rank0]: KeyError: 'wi_0'
```

**There is no `wi_0` in the trainer's `model_state`.** `prefuse_moe_weights=true`
really does fuse the trainer model, which is the premise §6.0.3 rests on and
which the null A/B had made me doubt. (My earlier inference — "it didn't
`KeyError`, so the source must be unfused" — was wrong for the non-debug runs
for the reason in §6.0.1: those never reached the converter's MoE code at all.)

So the two arms say: the trainer *is* fused when told to be, and the harness's
generation can't see it either way.

Worth recording as an elimination: that converter's interleave

```python
w1_chunks = wi_0.reshape(num_reps, num_experts, d_model, tp_size, chunk_size)
w3_chunks = wi_1.reshape(...)
combined_shards = jnp.stack([w1_chunks, w3_chunks], axis=-2)   # (…, tp, 2, chunk)
gate_up = combined_shards.reshape(num_reps, num_experts, d_model, -1)
```

produces exactly `[g_s0|u_s0|g_s1|u_s1]` at `tp_size = 2` — the same layout
`_interleave_moe_weights(lane_size=0)` produces, and the same layout `gmm_v2`
wants. **Two independently written converters agree on the MoE layout**, which
weakens the MoE-interleave family of hypotheses considerably.

#### 6.0.2b The harness does not exercise production's converter

An important caveat discovered while reading the null result, and a gap worth
reporting upstream:

```
[weight-sync] sampler.update_params via Qwen35MaxTextToVLLMConverter (convert+reshard+assign)
```

Production's trainer builds `WeightConverter` directly
(`maxtext_engine.py:649-668`). The harness got `Qwen35MaxTextToVLLMConverter`
instead, despite `direct_maxtext_sync=True, use_weight_converter=True` — the
combination `validate_converter.py:889` documents as "mode 1 new
(`WeightConverter(rules=None)`)". The reason is
`maxtext_vllm_rollout.py:56-64`:

```python
def _rule_table_for(model_name: str):
  if model_name == "qwen3-0.6b":
    return MODEL_TO_CONVERSION_RULES["qwen3"]
  return _NO_RULE_TABLE
```

Only `qwen3-0.6b` clears the `rule_table is not _NO_RULE_TABLE` guard at line
99, so for **every** other model — including all of qwen3.5 — the
`WeightConverter` returns at lines 103/111 are unreachable and control falls
through to the per-model torchax converter at line 116. So
`_create_model_converter` can never return a `WeightConverter` for
qwen3.5-35b-a3b, and `validate_converter`'s documented mode-1-new arm is not
selectable for this model.

Consequence: the harness reproduces the *symptom* but through a sibling
conversion path. That two independent converters both yield soup is itself
informative, but the harness cannot yet A/B production's own converter.

#### 6.0.3 The prefuse hypothesis, as argued from the code

Still worth stating, because the code reading stands on its own and the
launcher was in fact misconfigured relative to the flags' documented defaults.

**`prefuse_moe_weights` names two incompatible layouts, and the launcher was
setting both ends to the same value.**

The fused MoE kernel `wi` has last dim `2 × padded_moe_mlp_dim` = 1024. Which
half is gate and which is up depends on who reads it:

| end | attention | who reads `wi` | layout it assumes |
|---|---|---|---|
| trainer | `dot_product` (default) | `moe.py:3692`: `n = wi.shape[-1]//2; w0 = wi[..., :n]; w1 = wi[..., n:]` | **global** concat `[gate \| up]` |
| rollout | `vllm_rpa` (`configs/inference/vllm.yml:15`) | `moe.py:3690` → `tpu_inference.fused_moe_func` → tokamax `gmm_v2`, `rhs_up_ref = rhs[..., out_size_n:]` on the **local shard** | **per-shard** concat `[g_s0\|u_s0\|g_s1\|u_s1]` |

At TP=1 these coincide. At the rollout's `moe_mlp_tp_size=2` they do not.

Three code facts close the loop:

1. **The trainer builds the global layout at checkpoint load.**
   `model_creation_utils.py:221 _fuse_moe_weights` fuses the checkpoint's
   `wi_0`/`wi_1` into the model's `wi`, taking `n_shards` from *the trainer's
   own* sharding of `wi`'s last axis (lines 270-275). The validator log shows
   that spec as `P(None, None, 'fsdp', None)` — last axis unsharded — so
   **`n_shards = 1`**, i.e. plain global concat. Correct for the trainer; wrong
   for a TP=2 destination.

2. **The converter's per-shard interleave only runs on an *unfused* source.**
   `convert_utils.py:149`:
   ```python
   if not src_key or src_key[-1] != "wi_0":
     continue
   ```
   With `prefuse_moe_weights=true` on the trainer there is no `wi_0` in the
   source tree — only `wi` — so `_fuse_moe_bulk` / `_interleave_moe_weights`
   are skipped entirely and `wi` is copied **verbatim**. All the machinery for
   getting `n_shards` right (`resolve_rollout_tp`, `moe_mlp_tp_size`, the
   `_get_n_shards` fallback) is dead code on this path. That is why every
   attempt to fix `n_shards` in §6.1 found it already correct: the code that
   uses it was never reached.

3. **The defaults already encode the right answer; the launcher overrode them.**
   `run_trainer_node.py:212` → `default=False`, help text *"Off for the
   trainer."*; `run_rollout_node.py:218` → `default=True`. But
   `k8s_launcher.sh` passed the single env `PREFUSE_MOE_WEIGHTS` (default
   `true`) to **both** — line 264 (trainer) and line 410 (rollout) — under a
   comment asserting they had to match. `recipes/trellis_gsm8k_qwen3p5_35b.sh:46`
   sets the same global `true`, so this is upstream, not something introduced
   here.

**The resulting corruption is exactly the observed symptom.** `wi` is
`(256, 10, 2048, 1024)` = global `[gate(512) | up(512)]`, sharded 2-way on the
last axis: shard 0 receives all 512 gate columns, shard 1 all 512 up columns.
`gmm_v2` on shard 0 then computes `act(gate[:256]) * gate[256:]` and on shard 1
`act(up[:256]) * up[256:]`. Every expert MLP is garbage — but it is a pure
**permutation of the same elements**, so Raiden's per-tensor float32 abs-sums
match 633/633 with identical element counts, and the "transport is faithful"
evidence that blocked §6.1 for two sessions is simply measuring the wrong
thing. Fluent-looking digit soup from a coherent attention stack with broken
MLPs is the expected output.

It also explains, in retrospect, the failed `noprefuse` A/B in §6.4: that run
set `PREFUSE_MOE_WEIGHTS=false` on *both*, so the rollout went to 673 unfused
variables while the converter — which forces `prefuse=True` for its own plan
(`resolve_prefuse_moe_weights`, §6.4a) — produced a fused `wi`, giving
`preflight: source variable '...routed_experts.wi' has no destination
counterpart`. The §6.4a dead-code path is *benign* for the fixed configuration:
the converter should always fuse. Only the trainer's model config needs to be
false.

**The fix (config only, no image change, no code change):**

```
TRAINER_PREFUSE_MOE_WEIGHTS=false   # trainer keeps wi_0/wi_1
PREFUSE_MOE_WEIGHTS=true            # rollout wants fused wi  (unchanged)
```

`k8s_launcher.sh` now has the two as separate variables with those defaults, so
the recipe needs no change. With the trainer unfused, the converter's
`_fuse_moe_bulk` runs, `n_shards` resolves to `self.tp = 2` via
`rollout_tensor_parallelism=2`, and `_interleave_moe_weights(lane_size=0)`
emits the per-shard concat the kernel wants — which matches the
independently-computed `moe_mlp_tp_size=2` the destination reports at
`adapter.py:139`.

**Confirmation status: CONFIRMED end-to-end** on the real 40-chip topology —
see the completions at the top of §6.0. The `KeyError: 'wi_0'` of §6.0.2
establishes the one step that was in doubt — the trainer really is fused under
`prefuse_moe_weights=true`. The rest of the chain is code reading, now backed
by the end-to-end result:

- fused trainer ⇒ `convert_utils.py:149` (`src_key[-1] != "wi_0" → continue`)
  skips the fuse ⇒ `wi` copied verbatim. Corroborated negatively: neither
  4-chip run logged the `"Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d"`
  line from `convert_utils.py:175`.
- destination wants per-shard at 2 shards — agreed on independently by
  `gmm_v2`, by `adapter.py:139` (`moe_mlp_tp_size=2`), and by
  `qwen35_moe.py:271-283`.

The end-to-end effect could not be measured in the 4-chip harness — it cannot
generate meaningfully (§6.0.1) and cannot reach `WeightConverter` for this model
(§6.0.2b) — so it was measured on the real topology: a 2-step run with
`TRAINER_PREFUSE_MOE_WEIGHTS=false`. Result at the top of §6.0.

The change also aligns the launcher with the flags' own documented defaults
(`run_trainer_node.py:212` → `default=False`, *"Off for the trainer."*;
`run_rollout_node.py:218` → `default=True`), which the single shared env had
been overriding.

### 6.1 BLOCKER: reward is pinned at exactly 0

`reward_mean = 0.0000`, `std = 0.0000` on every step of every run so far
(including the older v4 runs, so this predates all changes described here).
With `BETA=0` and GRPO advantage `(r − mean)/std`, zero reward fully explains
`loss = 0.0000` and `perplexity = 1.0000`: **there is no learning signal at
all, and a 100-step run in this state would be worthless.**

Two false starts are worth recording, because both were reasonable and both
were wrong, and the second one wasted a run.

**False start 1 — "the completions are truncated."** I read the zero-variance
reward as a truncation signature and raised `MAX_RESPONSE_LENGTH` 512 → 1024.
Per-completion logging (overlay file 4) disproved it: the logged completions
were 10–58 characters and ended by emitting `<|endoftext|>` / `<|im_end|>` on
their own, nowhere near the cap.

**False start 2 — "the model emits garbage, so weight sync must be broken."**
That is what those 10–58 character completions look like, and with vLLM started
on `load_format: 'dummy'` (random weights, installed for real only by Raiden
sync) a bad sync is a natural suspect. But **the sample was not
representative**, and this is the actual finding:

> Overlay file 4 logs from `GSM8KEnv._step_impl` — i.e. from inside
> `env.step`. `trajectory_collect_engine._one_step` only calls `env.step` when
> `_check_and_set_context_limit_reached()` is False. **A trajectory that spends
> its entire response budget in one turn never reaches `env.step` at all**, and
> `collect()` additionally skips `_append_final_reward()` for it. So it scores a
> hard 0.0 regardless of content, *and* it is invisible to the example's own
> completion logging.

On rollout worker 0 of the 512-budget run, **27 of ~30 trajectories were
clipped that way** (`trajectory clipped: MAX_CONTEXT_LIMIT_REACHED`, 27
occurrences). The vLLM progress lines confirm they were real, full-length
generations — 47.9 s at 10.7 tok/s ≈ 512 tokens, exactly the budget. The only
completions I had actually been looking at were the *rare* ones where the model
emitted EOS after ~1 token, which is why they all looked degenerate and
near-identical.

So the mechanism behind `mean = 0.0000, std = 0.0000` is now established and is
mundane: **~84% of trajectories were dropped before grading**, and a dropped
trajectory contributes a hard zero. Zero variance does not require a collapsed
model; it only requires that almost nothing gets graded.

Correspondingly, the earlier claim that "all 8 workers emit byte-identical
completions, so the distribution has collapsed" should be **withdrawn on both
counts**: it rested on 1-token `<|im_end|>` samples, and in any case vLLM is
configured with `seed=0`, so identical output across workers is expected
determinism, not evidence of anything.

#### What the raw generations actually show (image v11, `RUN_TAG=gendiag`)

Overlay file 8 logs generations from `_one_step`, before the overlong filter can
drop them. With `TUNIX_LOG_GENERATIONS=8` and a 1024 budget, every logged
trajectory used the **entire** budget — `tokens=1024`, ~1100 chars — and the
content is:

```
head='\nS\n*    1\n2 1 1: 6 6 6 0  \n\n  *   . \n\n,\n\n\n\n \n\n 8 0 3 6  0 a ...'
first_ids=[198, 50, 198, 9, 256, 256, 16, 198, 17, 220, 16, 220, 16, 25, 220, 21]
```

**The model is genuinely broken.** Two things sharpen this beyond "it's garbage":

- The token IDs are overwhelmingly `198` (`\n`), `220` (space), `16`/`17`/… —
  i.e. the *unigram frequency* distribution, plus digits, which are contextually
  frequent for a math prompt. That is the signature of a network whose
  embedding and unembedding are intact while the 40 transformer layers
  contribute noise: the residual stream carries no information, so the logits
  fall back to token frequency. It is *not* the uniform-random-unicode
  signature of fully random weights.
- The prompt is correct. The logged `chat_completions` is the proper VTC
  template — "put your detailed step-by-step reasoning process inside
  `<reasoning>`…", the GSM8K problem, and the deliberately-open `<reasoning>\n`.
  **So the `chat_parser=auto` hypothesis is dead**; a prompt problem cannot
  produce digit soup.

#### Weight sync transports the right values — and is still the problem

With `VERIFY_WEIGHTS=true` (which only works from v11 on, see below), source and
destination checksums are the same to within float accumulation order:

| | trainer (source) | rollout (destination) |
|---|---|---|
| `decoder_norm.scale` | 3232.55078125 | 3232.55078125 |
| `layers_0.attention.A_log` | 105.5625 | 105.5625 |
| `layers_0.attention.conv1d.kernel` | 561.030517578125 | 561.030517578125 |
| `__grand_total__` | 304540019.06 | 304540018.28 |
| `__tensor_count__` | 633 | 633 |
| `__element_count__` | 34,660,610,688 | 34,660,610,688 |

So: all 633 tensors, all 34.66B elements, right values, and the transport writes
in place into the live serving state (`RaidenWeightSyncDelegate` docstring:
*"the transport binds the live transformer state directly, so weight_sync writes
into the serving copy rather than a shadow buffer"*).

**The key inference:** these checksums are per-tensor *absolute sums*, which are
invariant under permutation and reshape. Right values + broken model therefore
points squarely at **the arrangement, not the transport** — the elements arrive
but land in the wrong positions.

Ruled out:

- **Not missing tensors.** `raiden bind dropped 2 unbindable leaves` names only
  `routed_experts.rngs.params.count` and `.key` — RNG state, correctly excluded.
- **Not the base checkpoint.** The trainer restores it cleanly
  (`[sync] Finished load in 95.96 seconds`), and its own source checksums look
  like trained weights, not init. `decoder_norm.scale` has abs-sum 3232.55 over
  a `(2048,)` tensor (`base_emb_dim=2048`, confirmed by the trainer's own
  `wi_0 shape=(256, 10, 2048, 512)` log), i.e. mean 1.578. A freshly
  initialised RMSNorm scale is exactly ones, which would sum to exactly
  2048.0. The trainer is demonstrably holding the real checkpoint.
- **Not response length, not the prompt, not the chat parser**, per above.
- **Not caused by anything in this overlay** — v4 showed the same zero reward.

#### Why source-vs-destination comparison cannot solve this

Worth stating plainly, because it cost me a diagnostic run. The checksums are
taken on the trainer *after* conversion and on the rollout *after* receipt. The
conversion is upstream of both. So no comparison between the two sides — not
abs-sums, not a permutation-sensitive element-order hash, not a full bit
compare — can tell you whether the converted arrangement is the one the rollout
kernel actually wants. A perfect match only proves the transport is faithful,
and we already know it is. Answering the real question needs **ground truth**:
convert the weights and then *generate*, and read the text.

That is what `run_qwen35_validate_converter.sh` (new, §6.5) does, on 4 chips.

#### What the MoE layout is actually required to be

I traced this end to end rather than guessing, because the fuse is the leading
suspect and it is parameterised by a shard count that is derived, not passed.

- The rollout serves MaxText through vLLM, which layers
  `maxtext/configs/inference/vllm.yml` on top — that is where `attention:
  "vllm_rpa"` comes from. The rollout node does **not** set it
  (`maxtext_attention=''` in its parsed args).
- Under `vllm_rpa` + `prefuse_moe_weights=true`, `moe.py:3690` hands `self.wi`
  to `tpu_inference`'s `fused_moe_func` **untouched** — it does not take the
  `wi[..., :n] / wi[..., n:]` split at `moe.py:3692`. So the required layout is
  the kernel's, not MaxText's.
- The kernel is tokamax `gmm_v2`. With `fuse_act` set it does
  `rhs_up_ref = rhs[..., out_size_n:]` where `out_size_n = size_n // 2`, on the
  **local shard's** rhs, and then lane-interleaves gate/up into VMEM itself
  (`FusedWeightsRef.get_weight() -> interleave_lane(...)`).

So each TP shard must receive a plain `[local_gate | local_up]`, i.e. exactly
what `_interleave_moe_weights` produces with `lane_size=0` and
`n_shards == the destination's shard count on the fused axis`. Pre-interleaving
in HBM would double-interleave.

The destination's shard count: `moe.py:602` sets the MoE kernel's TP axes to
`("model", "attn_dp")` under `vllm_rpa`. The rollout log reports
`ShardingStrategy(tensor_parallelism=2, ..., attention_data_parallelism=1)` over
4 devices, so the fused axis is split **2** ways.

**The converter arrives at 2, but by luck.** In `WeightConverter.__init__`:

```python
self.tp           = resolve_rollout_tp(config, tp)
self.kv_tp_size   = kv_tp_size   or getattr(config, "kv_tp_size", 1)   or self.tp
self.moe_mlp_tp_size = moe_mlp_tp_size or getattr(config, "moe_mlp_tp_size", 1) or self.tp
```

MaxText defaults both `kv_tp_size` and `moe_mlp_tp_size` to **1**, which is
truthy, so the `or self.tp` fallback is unreachable — and nothing ever sets
them, see §6.4. Hence the trainer's log line `kv_tp_size=1, moe_mlp_tp_size=1`.
For the MoE the code then falls through to `self.tp`:

```python
n_shards = (self.moe_mlp_tp_size if self.moe_mlp_tp_size > 1
            else (self.tp if self.tp > 1 else _get_n_shards(wi_0, scan_fused_axis)))
```

and `self.tp` *does* resolve to 2, via `rollout_tensor_parallelism=2`. So
`n_shards=2`, which matches. But note the last fallback: with no rollout TP
configured at all, this reads the shard count off the **trainer's** sharding of
`wi_0`, which on an FSDP=8/TP=1 mesh is 1 — silently producing a global
`[all_gate | all_up]` concat. Anyone reproducing this on a different topology
must keep `ROLLOUT_MESH_TP` set.

Remaining hypotheses, now narrowed to layout:

1. **MoE interleave.** Still open, but weakened: the required layout and the
   produced layout both work out to per-shard concat with `n_shards=2`, per the
   trace above. What is *not* yet verified is that the rollout's MaxText state
   really shards `wi`'s fused axis 2 ways and not 4 (`data_parallelism=2` also
   appears in its sharding config). The MoE experts are ~32B of this 35B model,
   so this stays first on the list until a generation test clears it.
2. **Cross-mesh shard mapping.** The trainer is FSDP=8 over 2 hosts; each rollout
   is `data:2 × model:2` with `num_shards=4`. Raiden pairs tensors by name and
   ships shards; if the 8-way source sharding is mapped onto the 4-way
   destination incorrectly, every device gets the wrong slice and the global
   abs-sum is still exactly preserved.
3. **Hybrid Gated-DeltaNet tensors.** 30 of 40 layers are GDN (`A_log`,
   `dt_bias`, `conv1d.kernel`, `in_proj_qkvz`, `in_proj_ba`) — packed layouts
   whose wrong unpacking would corrupt three-quarters of the network exactly as
   observed. Argument *against*: both ends are MaxText and the converter copies
   these by name with no restructuring, so the two sides should agree by
   construction. The MoE is special precisely because the trainer stores
   `wi_0`/`wi_1` separately and the rollout wants one kernel-shaped `wi`.

The `validate_converter` harness in §6.5 discriminates (1)+(2) from (3) in one
4-chip run: it converts and generates in a single process with no Raiden and no
cross-mesh transfer, so coherent text there localises the fault to the
distributed transport, and garbage there localises it to the conversion.

Two practical notes for whoever picks this up:

- `VERIFY_WEIGHTS` did not reach the container before v11:
  `run_qwen35_repro.sh` hard-coded `export VERIFY_WEIGHTS=false`, silently
  overriding the caller. One diagnostic run was lost to this. It is now
  `${VERIFY_WEIGHTS:-false}`. Confirm any diagnostic flag actually landed with
  `kubectl get pod ... -o jsonpath='{.spec.containers[0].command}'`.
- `DEBUG=1` is *not* a usable alternative at this scale — it routes
  httpx/httpcore through the node loggers and buries the run in HTTP headers.
  That is why overlay files 4 and 8 log at INFO with per-worker budgets
  (`GSM8K_LOG_COMPLETIONS`, `TUNIX_LOG_GENERATIONS`).

### 6.2 Trajectories that use the whole response budget are never graded

Worth stating separately from 6.1, because it is a real defect that will bite
any long-reasoning model even once the weights are fixed. In
`trajectory_collect_engine`:

- `_one_step` calls `env.step` only `if not
  _check_and_set_context_limit_reached()`, and
- `collect()` skips `_append_final_reward()` when the status is in
  `filter_statuses` and `overlong_filter` is on (it defaults to **True** in
  `TrajectoryCollectEngine.collect_batch`, line 347).

So a trajectory that spends its full budget scores a hard 0.0 rather than being
graded on what it produced. Under the VTC recipe that is the difference between
0.0 and the 0.1 a correctly-formatted-but-wrong answer earns — i.e. it removes
exactly the gradient signal GRPO needs. At `MAX_RESPONSE_LENGTH=512` this hit
27 of ~30 trajectories on one worker; at 1024 it still hit every logged one.

Raising the budget only postpones this. The real options are to grade clipped
trajectories on their partial output, or to leave `overlong_filter` on but
accept that the budget must exceed what the model actually emits. Decide this
deliberately — it is a change to what the reward means, so I have not made it.

### 6.3 Resolved this session

**JobSet restart-into-deadlock — fixed.** When the orchestrator died,
`RestartJobSetFailurePolicyAction` + `BackoffLimitExceeded` recreated the pod,
but the rollout workers stayed registered with the *dead* discovery server. The
new orchestrator then waited forever while 40 v5p chips sat idle and every pod
read `1/1 Running`. Fixed by parameterising `maxRestarts` in
`jobset.cpu.yaml` and defaulting the orchestrator to 0 (workers stay at 3), via
`ORCHESTRATOR_MAX_RESTARTS`. Confirmed working: the `gendiag` run's orchestrator
died and stayed `0/1 Error` instead of deadlocking the cluster.
**Monitor the `Program End: EXIT_CODE=` line,
not pod status.**

**Benign but misleading:** `No checkpoint found, skipping restore.` is the tunix
RL-resume probe against the empty run directory. Base weights load fine —
look for `[sync] Finished load in 95.96 seconds @ gs://hengtaoguo-maxtext-logs/...`.

### 6.4 Two config flags that silently never reach the converter

Both found while trying to run the `PREFUSE_MOE_WEIGHTS=false` A/B. Neither is
fixed here — flagging first, per the standing instruction.

**(a) `prefuse_moe_weights=false` cannot be expressed on the trainer.**
`convert_utils.resolve_prefuse_moe_weights` is:

```python
if prefuse_moe_weights is not None:            return bool(prefuse_moe_weights)   # caller passes None
if "ROLLOUT_PREFUSE_MOE_WEIGHTS" in os.environ: return ...
if getattr(config, "rollout_prefuse_moe_weights", None) is not None: return ...   # field does not exist
rollout_backend = ... or "maxtext"
if rollout_backend == "maxtext":               return True        # <-- our path, unconditional
if getattr(config, "prefuse_moe_weights", None) is not None:  return bool(...)    # dead code
```

The `config.prefuse_moe_weights` check sits *below* the unconditional
`return True`, and `rollout_prefuse_moe_weights` is not a MaxText config field
at all. So on the MaxText rollout backend the flag is pinned to True no matter
what the config says. Observed directly:

```
[TrainerNode] Config param prefuse_moe_weights: False
[TrainerNode] MaxTextToMaxTextConverter: ... moe_fused_layout=per_shard_interleave, prefuse_moe=True, ...
[TrainerNode] MaxText param base.decoder.layers.layer_0.mlp.routed_experts.wi_0 shape=(256, 10, 2048, 512)
```

The rollout *did* honour the flag (`bind prepared 673 arrays` instead of 633),
so the two ends disagreed and the run died at preflight:

```
WeightSyncError: round 0 ... manifest preflight failed before any destination was quiesced;
no rollback needed (120 problems, first: preflight: source variable
'decoder.layers.0.mlp.routed_experts.wi' has no destination counterpart)
```

**So the `noprefuse` A/B did not test anything** — it failed before a single
weight moved, and the MoE-interleave hypothesis is untested, not refuted. The
only working lever today is the env var `ROLLOUT_PREFUSE_MOE_WEIGHTS=false` on
the trainer pod. The fix upstream is to move the `config.prefuse_moe_weights`
check above the `rollout_backend == "maxtext"` default.

**(b) `kv_tp_size` / `moe_mlp_tp_size` never reach the MaxText config.**
`tunix/utils/maxtext_utils.build_maxtext_config` derives both from
`rollout_mesh_tp` (lines 128–140) but the argv it emits contains only
`rollout_tensor_parallelism=`; neither `kv_tp_size=` nor `moe_mlp_tp_size=` is
ever appended. MaxText then uses its defaults of 1, and because
`WeightConverter.__init__` writes `kv_tp_size or getattr(config,"kv_tp_size",1)
or self.tp` — with 1 being truthy — the `or self.tp` fallback is dead. Result:
`kv_tp_size=1, moe_mlp_tp_size=1` in the converter's log even though
`--rollout_mesh_tp=2` was passed.

The MoE path survives this by falling through to `self.tp` (see §6.1). The KV
path does not: `kv_replication = kv_tp_size // base_num_kv_heads` is computed
from the wrong number. Here it happens to be harmless — `base_num_kv_heads=2`
and rollout TP=2 means one KV head per shard and no replication is needed — but
it would be wrong on any topology where TP exceeds the KV head count.

### 6.5 New: a 4-chip ground-truth harness

`tunix/experimental/examples/math_gsm8k_dist/run_qwen35_validate_converter.sh`
(host-side only, no image change) wraps
`maxtext.integration.vllm.validate_converter` as a single-slice JobSet:

```
./run_qwen35_validate_converter.sh start | logs | stop | yaml
```

It loads the real Orbax checkpoint into MaxText, converts with the *same*
`WeightConverter` production uses (via `MaxTextVllmSampler.update_params`),
assigns into vLLM, and generates greedily (`temperature=0.0`) from a GSM8K
prompt. `rollout_tensor_parallelism=2` and `prefuse_moe_weights=true` mirror
production. One v5p `2x2x1` slice, no Raiden, no orchestrator, no trainer /
rollout mesh split — about 4 chips for ~15 minutes against 40 chips for ~25.

Useful knobs: `USE_CONVERTER=false` runs the legacy tunix
`transfer_state_directly` instead, for A/B; `DEBUG_CONVERTER=true` stops after
the key-coverage and weight-stat checks without generating; `ROLLOUT_TP` and
`PREFUSE` vary the two suspect parameters.

---

## 7. Host-side plumbing (no image change)

- `yaml_generator.py`: added `--priority_class`, emitting the
  `kueue.x-k8s.io/priority-class` label. Without it the workload sits at
  priority 0 and anything in `tpu-shared-cohort` evicts it — an earlier 100-step
  attempt died at step 43 that way. `medium` = 500, applied to all 11 workloads.
- `k8s_launcher.sh`: exports `KUEUE_PRIORITY_CLASS`, passes `--priority_class`
  at 3 sites, and passes `--prefuse_moe_weights` / `--max_seq_token_per_tpu` to
  the trainer block.
- `PATHWAYS_PROXY_MEMORY_LIMIT=250G`: the whole model is staged on the proxy
  host during Raiden D2H; the generator's 100G default OOM-kills it at 35B.

## 8. Metrics

`WANDB_API_KEY` is read out of `~/.bashrc` by the launcher
(`eval "$(grep -m1 '^export WANDB_API_KEY=' ~/.bashrc)"`) because `.bashrc`
returns early for non-interactive shells, so sourcing it is not enough. The key
value is deliberately never printed or written to any report or manifest.
Project `qwen35-35b-a3b-grpo`; `FLUSH_METRICS_EVERY_N_STEPS=1`.
