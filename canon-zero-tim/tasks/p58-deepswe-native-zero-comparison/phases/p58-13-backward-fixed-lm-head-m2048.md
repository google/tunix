# P58.13 — Qwen3-4B trainer-logprob M2048 and P59-only VMA repair

## Status

`LOCAL IMPLEMENTATION + PINNED-IMAGE CONSTRUCTION PASS / TARGET RETRY NOT RUN`

The repair is uncommitted and unpushed. Image publication, Kubernetes apply,
and a fresh target remain separately user-gated.

## Immutable trigger

Target `canon-p58-ds4b-zero-hp-full-p58z02` ran the signed Qwen3-4B-Instruct
DP8xTP8 recipe on 128 TPU chips. The P58.12 seed route worked:

```text
[P58.SEED] ... scope=engine-global
[VLLM.JAX_SEED] ... engine_seed=42 request_seed=none
[DEEPSWE.ROLLOUT_DEADLINE] batch_complete prompt_groups=8 elapsed_secs=1514.2
[P58.LOGPS_BATCH] configured_prompts=8 generations=16 execution_trajectories=128 observed_trajectories=128
```

All 128 collector rows returned in one wave inside the 3,600-second batch
deadline. This was not a timeout-free batch: one row was `MODEL_TIMEOUT` and
two rows were `MAX_CONTEXT_LIMIT_REACHED`. The signed compact-status policy
retained them; they did not cause the process failure.

After rollout, the first trainer canonical per-token-logprob forward emitted:

```text
[PATHTRACE] CANON_ADAPTER_DP_FIXED_M_CHUNKS data=8 static_width=20480 chunks=80 global_M=2048 local_M=256
ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256, 4096), got (2048, 2560)
```

The failure occurred in `_process_results -> get_actor_per_token_logps ->
compute_per_token_logps -> canonical adapter -> compute_logits -> tied embed
fixed lm_head`. It was before alignment completion, backward, gradient
accumulation, AdamW, or any optimizer commit. Therefore `p58z02` is immutable
failure evidence, not a resumable training checkpoint.

Preserved evidence:

- `evidence/p58z02_backward_fixed_lm_head_error/run.log`
- SHA-256 `7349c7965f31e2c84dfd98f8cb7fe175f9b2d4281759d0bb5c07bb336ef8784d`

## Root cause and exact shape ledger

Qwen3-4B has hidden width 2,560 and TP=8. The caller-global learner program
has semantic M=2,048, composed as eight data ranks times the DP-local sequence
bucket M=256. The canonical fixed kernel still executes eight deterministic
M=256 chunks. The fixed-head registry admitted M=2,048 only for the existing
Qwen3-8B `(hidden=4096, tp=8)` geometry, so Qwen3-4B `(2560, 8)` incorrectly
fell back to the generic learner admission `(4096,)`.

The repair is geometry-exact:

- `(2560, 8)` Qwen3-4B admits learner M `(2048, 4096)`;
- `(4096, 8)` Qwen3-8B keeps learner M `(2048, 4096)`;
- every other geometry keeps learner M `(4096,)`.

In particular Qwen3-32B `(5120, 8)` does not inherit M=2,048 without its own
signed target evidence. The implementation does not generalize admission to
all TP8 models.

## FrozenLake Wave-5 repair imported into DeepSWE

FrozenLake Wave 5 independently proved that the process-wide P66 checked-VMA
compatibility alias leaked VMA metadata into ordinary serving. Both the
`p66-off` and `serving-scope` arms recovered A-B=0 and B-C=0; the preferred
`serving-scope` arm retains checked-VMA inside the exact P59 backward while
preserving the historical serving graph.

P58 Zero-HP now imports that shared numerical repair with:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1       # derived compatibility alias
CANON_P67_P66_VMA_P59_ONLY=1   # serving graph unchanged; P59 backward only
```

Admission remains fail-closed to the exact Qwen3-4B P58 Zero/full,
DP8xTP8, 1,000-update, strict-alignment HP profile. Native raw, Native+IS,
ordinary three-update/non-HP Zero, Qwen3-32B, and unrelated profiles do not
receive P67. P67 is a graph-scoping repair, not a relaxed alignment threshold;
the target still requires A=B=C exactly.

## Validation

- Focused host tests: 50/50 PASS across the P58 profile/renderer, fixed-head
  geometry, and FrozenLake P67 red/recovery/B-C-negative controls.
- Adjacent host gates: P34 static 10 suites, P57 146/146, and the flag-registry
  regression PASS.
- Qwen3-4B installed overlay: all 37 manifest files match and the real endpoint
  reports `learner_M=2048,4096`.
- Qwen3-32B installed overlay: exact-image gate exits zero and continues to
  report `learner_M=4096`.
- Complete dependency-bearing P58 pinned-image gate exits zero with:

```text
P58_EXACT_IMAGE_CPU_PASS ... qwen4b_fixed_head=1 checked_vma=1 vma_p59_only=1 first_update=1 ... regressions=1
```

The pinned container has no `/dev/vfio`; this proves construction and installed
shim behavior only. It does not prove target TPU A=B=C, backward, optimizer,
or convergence.

## Next target gate

After explicit commit/push approval, exact remote readback, matching-image
publication, sandbox-capacity admission, and separate launch approval, render
a fresh `p58z03` with `--stage full --arm zero --high-performance`. Do not
resume or overwrite `p58z01` or `p58z02`.

The fresh target must first prove:

1. all 128 rows return inside the signed batch deadline;
2. installed Qwen3-4B fixed-head accepts global M=2,048/local M=256;
3. A-B and B-C are byte-exact with P67 enabled;
4. trainer forward and the 16-group backward are finite;
5. update 0 emits the checked-VMA, first-update, stable-clip, and exactly-one
   optimizer transaction receipts.

If those gates pass, the same job continues toward 1,000 commits. Any finite
alignment difference remains a hard Zero-HP failure; no warning-only fallback
or gate deletion is admitted.
