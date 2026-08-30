# GSM8K Native/mismatch versus Zero full handoff

## START HERE — prepare a matched DP16xTP4 full comparison

Status is `ATTEMPT 03 REPAIR IMPLEMENTED / PINNED-IMAGE PASS /
SOURCE CL READY / TARGET NOT RUN`. This task prepares one stock
Native/mismatch control; it does not launch it. Commit, push, and
Kubernetes/TPU launch each require their own explicit user approval. Render
only from a clean checkout of the exact approved, published 40-character SHA.

The latest immutable failure is Attempt 03 at source `0b62b6bb`: the learner
reached real Qwen3 Splash Attention, where an Explicit DPxTP mesh rejected a
replicated kernel-mask leaf against `shard_map.in_specs=P('model', ...)` before
executing model math. On repair base `2af1197f`, the current source CL
reshards the real Splash kernel pytree to its existing
`manual_sharding_spec`, only when the mesh has Explicit axes. Auto-mesh
programs are unchanged. The pinned production image passes the exact failing
negative, repaired positive, value-equality control, and adjacent Zero
renderer. A fresh target run is still required.

The comparison uses one W&B project and group for both arms:

```text
project=zero-tim-gsm8k-dp16-tp4
group=qwen3-1p7b-dp16-tp4
```

JobSet-derived W&B run names are intentionally different:

```text
Native/mismatch: canon-v1ctl-gsm-nat-<run-id>-<sha8>
Zero V1-HP:      canon-v1hp-gsm8k-<run-id>-<sha8>
```

### Render both arms without launching

From the physical `canon-zero-tim` repository root, after review,
publication, exact remote SHA read-back, and clean checkout:

```bash
bash canon-zero-tim/tests/v1_gsm8k_native_full/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a

bash canon-zero-tim/tasks/v1-gsm8k-native-full-control/prepare_gsm8k_native_full.sh \
  <approved-40-character-sha> \
  /tmp/v1-gsm8k-native-<fresh-wave-id> \
  <fresh-native-run-id>

bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_gsm8k_full_dp16tp4_p74.sh \
  <same-approved-40-character-sha> \
  /tmp/v1-gsm8k-zero-<fresh-wave-id> \
  <fresh-zero-run-id>
```

Both wrappers reject a dirty tree, a SHA/HEAD mismatch, a reused output
directory, and invalid identities. Each hashes exactly one manifest, prints
one unpiped `kubectl apply` command, and emits `launch=not-executed`; neither
executes a launch.

### Matched and intentionally different fields

| Contract | Native/mismatch control | V1-HP Zero |
|---|---|---|
| Source/model/data | same approved SHA, Qwen3-1.7B, GSM8K | same |
| Seed | driver constant `SEED=42` | same |
| Command | exact `_gsm8k_command(200)` | byte-identical |
| Geometry/horizon | DP16xTP4, 200 updates | same |
| Optimizer | resident | same |
| LM head | untreated stock head | fixed Zero head |
| W&B | same project/group | same project/group |
| Profile | stock `...gsm8k-native.env` | strict `...gsm8k-v1-hp.env` |
| Numerical runtime | P56 vanilla, `P32_WORKLOAD` absent, stock engine | canonical Zero-TIM |
| Alignment | no alignment observer | strict Zero-TIM |
| Backward | ordinary Tunix trainer | rank-parallel + checked-VMA |
| Latest system tuple | absent | checked-VMA/P70 receipts/P71 forward scan |

The Native arm reuses only the original P33 scientific command and full-run
restart policy. It deliberately does not reuse the P33 canonical GSM8K
profile. `CANON_GSM8K_VANILLA=1` selects the already exercised stock trainer
branch; `CANON_P32_WORKLOAD` is absent, no alignment observer runs, and the
entrypoint skips installation of the canonical engine overlay. “Mismatch” is
the untreated training arm where rollout/trainer log-prob drift is allowed to
affect ordinary training, not a warning-only Zero-TIM observer mode.

### Pre-launch manifest gate

For Native, require:

```text
CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env
CANON_P33_SHARED_MESH=16,4
CANON_P33_RUN_STAGE=full
CANON_P33_NO_COMMIT=0
CANON_OPT_STATE_RESIDENT=1
CANON_P30_OPT_STATE_OFFLOAD=0
CANON_P32_TRAIN_ADMITTED=0
CANON_P32_DP_REDUCTION_ADMITTED=0
CANON_P33_WORKLOAD_LAUNCH_ADMITTED=0
CANON_GSM8K_TRAIN=1
CANON_GSM8K_VANILLA=1
```

The following must all be absent from the raw Native manifest:

```text
CANON_P32_WORKLOAD
CANON_P59_RANK_PARALLEL_BACKWARD
CANON_P59_CHECKED_VMA
CANON_V1_HP_FULL
CANON_V1_HP_FIRST_UPDATE_GATE
CANON_DP_COMPARE_MODE
CANON_DP_DISTINCT_SCHEDULE
CANON_DP_FINITE_FETCH
CANON_P71_SCAN
CANON_DP_COLLECTIVE_REDUCE
CANON_P67_P66_VMA_P59_ONLY
CANON_P63_OVERFLOW_SAFE_CLIP
CANON_ALIGNMENT_GATE
CANON_ALIGNMENT_GATE_ONLY
CANON_ALIGNMENT_UPDATE_CANARY
CANON_ALIGNMENT_TRAIN
CANON_PRE_ALIGN_GATE
CANON_GSM8K_AB_REPORT_ONLY
CANON_GSM8K_ALIGNMENT_WARN_ONLY
CANON_P38_FIXED_LM_HEAD
CANON_PRE_ALIGN_REPORT
CANON_ALIGN_REPORT
CANON_UPDATE_REPORT
CANON_P38_MISMATCH_CAPSULE
CANON_P38_MISMATCH_CAPSULE_MAX_ROWS
```

The resolved profile must emit
`[GSM8K.NATIVE] ZERO_TIM_OFF_PASS p32=absent canonical_engine=off
alignment=off p59=off v1=off`. Its client `XLA_FLAGS` and Pathways proxy
environment must not contain `--xla_allow_excess_precision=false`. Before
training, entrypoint must emit both:

```text
[GSM8K.NATIVE] STOCK_PREFLIGHT_PASS files=6 driver_import=pass canonical_overlay=absent alignment=off
[entrypoint] GSM8K_NATIVE_STOCK_PATH ... canonical_overlay=skipped alignment=off
```

The training log must contain the two `[P56.VANILLA]` stock-arm receipts and
must not contain `[CANON_ADAPTER]`, `zt_tr_dp_parallel_bwd_`, or any
`CANON_ALIGN` verdict.

For Zero, require the strict V1-HP profile and:

```text
CANON_V1_HP_FULL=1
CANON_P59_RANK_PARALLEL_BACKWARD=1
CANON_P59_CHECKED_VMA=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
CANON_GSM8K_ALIGNMENT_WARN_ONLY=0
```

`CANON_DP_COLLECTIVE_REDUCE` remains absent from Zero too. P74 is source
behavior inside checked-VMA, not a flag to add by hand.

### Launch and result discipline

No apply is authorized by this handoff. Before any separately approved
launch, verify no conflicting JobSet/process is using the target and apply
the reviewed manifest without a pipe. Use fresh run IDs and never reuse a
failed output/run directory. Preserve raw logs, resolved `env.sh`, manifest
and index hashes, retry state, W&B identity, and XProf/Perfetto output. The
Native arm intentionally has no alignment/update-report files; the Zero arm
retains its alignment and update records.

Compare whole-step wall time and `p32_vag_reverse`; do not use
`grad_accumulate` as model-backward time. Because rollout execution can still
have operational nondeterminism despite the common seed, treat the full runs
as matched-configuration controls, not guaranteed bitwise paired rollouts.

## Offline verification status

The host task suite passed nine tests with one pinned-image-only skip. After
the Attempt 03 repair, the pinned image passed all ten Native contracts, nine
Qwen sharding tests (including the real Splash negative/positive), and one
Zero neighbor, emitting
`V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS native_contract=10 qwen_sharding=9
zero_neighbor=1`. Earlier FrozenLake/DeepSWE gates, the GSM8K XProf contract,
and the four-carrier aggregate remain construction evidence only. Commands
and rejected harness attempts are recorded in `validation.log`.

Not verified: clean-SHA wrapper success for this repair, post-fix Kubernetes
server dry-run/apply, TPU full training, a real optimizer commit, target
performance/XProf/convergence, live W&B comparison, or image publication.
The next approved target must use a fresh run ID and is accepted
only after the learner crosses the previously failing Splash call and at
least one optimizer commit; an admission pass alone is not a convergence or
performance result.
