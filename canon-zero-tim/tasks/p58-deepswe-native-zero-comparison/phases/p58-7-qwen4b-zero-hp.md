# P58.7 — Optimized Qwen3-4B Zero-TIM full campaign

Status: implementation complete; host and pinned-image admission gates PASS;
DP8 x TP8 target NOT RUN because no 128-chip launch was authorized.

## Goal

Apply the reviewed high-performance Zero-TIM system to the frozen P58
Qwen3-4B-Instruct-2507 DeepSWE-derived recipe without changing its scientific
workload, then run one uninterrupted 1,000-commit DP8 x TP8 rollout plus DP8 x
TP8 trainer campaign after separate publication and launch approvals.

## Frozen scientific recipe

- exact 1,012-task P46 clean list, B8 x G16, concurrency 128 in one wave;
- prompt/response/turn limits 4,096 / 16,384 / 50;
- rollout DP8 x TP8 plus trainer DP8 x TP8, 128 TPU chips total;
- eight ordered 16-trajectory accumulation calls;
- RLOO `sequence-mean-token-scale`, fixed divisor 16,384;
- AdamW 1e-6, signed betas/decay/clip, device-resident optimizer;
- sampler IS and TIS off, rollout logps as old logps, compact filtering and
  existing no-commit/sandbox-outage semantics unchanged; and
- exactly 1,000 committed updates with updates 1–3 observed inside the same
  full JobSet, not a separate canary.

## Admitted performance bundle

- P47a automatic no-prompt-logprob allocation;
- continue-decode K=8 with async scheduling rejected;
- fixed-AR gather;
- DP-aware gathered selected logprobs;
- logprob step fusion;
- registered tied Qwen3-4B K2560/TP8 fixed output head;
- existing device-resident optimizer/accumulator placement;
- P59 rank-parallel DP-local backward followed by the original fixed reduction
  and unchanged trajectory-group order;
- `CANON_P28_BATCHED_REPORT=1` as common profile identity. P59 owns the report
  adjoint in this path, so no separate P28 batched-report speed claim is made;
- one warmed update XProf capture plus one semantic Perfetto trace.

APC, batched reverse, batched evidence, fused tree, norm/input matmul, sample
split, engine-logprob readback, anchor overlap, and optimizer offload remain
off. Native stock observers and warning-only alignment are forbidden.

## Target gate

The immutable target must satisfy all of the following:

1. exact pushed 40-character source and immutable image identity;
2. 128 durable trajectories per batch and existing Kueue/Pathways/sandbox
   capacity contracts;
3. exact live weights and the fixed-head endpoint receipts;
4. every real pre/post `CANON_ALIGN` record PASS with zero failures;
5. one P59 reducer receipt per effective update, finite gradients and monotonic
   one-commit transactions with TPU-resident optimizer state;
6. exactly 1,000 committed updates, complete checkpoints/evaluation/journals;
7. nonempty XPlane, trace.json.gz and exactly one semantic Perfetto artifact;
8. classifier `PASS`, with profiled update excluded from steady `[PERF]` means.

P59 is accepted under ordinary-JAX FP64 gradient correctness. The campaign
does not claim bitwise identity to historical serial-backward AdamW weights.
Any real strict-alignment failure kills the candidate without reinterpretation.

## Result log

- The default-off profile is implemented at
  `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env:1`; renderer selection
  is at `cluster/render_p58_deepswe_tim.py:1`; exact runtime admission is at
  `tunix/rl/deepswe_contract.py:1` and `cluster/steps/00_env.sh:1`.
- Fixed-head receipt and automatic postflight routing are implemented at
  `cluster/steps/90_run.sh:1`; the full decision/performance ledger is
  `scripts/classify_zero_hp_full.py:1`.
- Verified by host renderer 16/16, profile 4/4, full classifier 3/3, P57
  128/128, V1 12/12, P59 30/30, APC 31/31, flag audit 366/366, syntax and diff
  hygiene.
- Verified in pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  by terminal
  `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 zero_hp_full=1 p59_tp4_tp8=2 p57_wandb=1 regressions=1`.
- DP8 x TP8/128-chip Zero-HP is **not verified** because no image publication,
  YAML launch, Kubernetes apply, or TPU execution was authorized in this turn.
- P59, Qwen3-4B TP8 fixed head, DP-aware gathered logprobs, and the full
  Pathways split therefore remain **target not run**, not promoted from CPU or
  DP1 x TP4 evidence.

## Rollback

The profile, renderer selector, admission tuple, receipts, tests, and
classifier are additive and default off. Revert P58.7 in reverse order:
ledger/recipe, workload admission, trainer optimization, serving optimization.
The historical Native and strict unoptimized Zero profiles remain available.
