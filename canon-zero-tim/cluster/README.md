# Running canon-zero-tim on Pathways/GKE

Operator guide. If you are running this on a cluster, read all four sections — especially
**§3 What counts as red**, because the characteristic failure here produces a *green* run.

---

## 0. What this is

`canon-zero-tim` makes three quantities bit-identical for the same weights and tokens:

```
A = the rollout engine's decode logprobs      (the behaviour policy the tokens were sampled from)
B = the same engine re-scoring them in prefill
C = the differentiable training forward
```

They diverge because bf16 addition is not associative and XLA compiles a *different program*
for a different shape, a different jit boundary, or a forward-plus-backward graph. The package
pins each of those degrees of freedom. See `../README.md` for the mechanism.

**Everything below assumes the numbers were only ever validated on a directly-attached 4-chip
v5p host, single slice, TP4.** Pathways is a different runtime and your topology is probably a
different shape. That is exactly why the first thing you run is a probe, not a training job.

---

## 1. How to run

```bash
kubectl apply -f jobset-64chip.yaml
kubectl logs -f jobs/canon-zero-tim-v5p-64-pathways-head-0
```

Before the first apply, edit `jobset-64chip.yaml`:

| Field | Why |
|---|---|
| `jax-tpu` container `image` | **Pin by digest.** A floating `:latest` means the same manifest runs on a different engine tomorrow — incompatible with a bitwise contract. |
| `CANON_P32_EXPECT_MODEL_MESH_IDS` | Leave empty for the first probe; fill from `probe_mesh_order.py` output on **this** topology. The profile maps it to the engine assertion without inheriting the four-chip default. |
| branch name in the sync block | If it is not `yuxzhang/canon-zero-tim`. |

### Promote one stage at a time

`CANON_MODE` selects how far the entry point goes. **Read the previous stage's output before
promoting.**

| Mode | Steps | Costs TPU? | Answers |
|---|---|---|---|
| `probe-only` | 00–25 | no | Is this image's `tpu_inference` the one the patches were cut against? Does this build need the RoPE fix? |
| `install-only` | 00–50 | no | Does the chain build, overlay and *load* here? |
| `gate-only` | 00–70 | yes, minutes | Does generic Pathways drift reproduce, and does the real canonical Qwen MLP operator chain remain bitwise across the third program? |
| `dp-gate-only` | 00–75 | yes, minutes | In the same Pathways client, does DP16×TP4 repeat exactly under a fixed sample→rank mapping, and are all replicas synchronized after gradient reduction? |
| `run` | 00–90 | yes | The workload in `CANON_RUN_CMD`; the P32 admission profile deliberately refuses this mode. |

### Secrets

`HF_TOKEN` and `WANDB_API_KEY` are inherited from the `yuxzhang-secrets` secret as
`INJECTED_*`. `00_env.sh` strips whitespace from them — Kubernetes secrets routinely carry a
trailing newline, and a token with `\n` fails authentication in a way that reads like a wrong
key. Never put a key in the manifest, a log, or a report.

---

## 2. What you should see

### `probe-only`

```
[probe] tpu_inference=<path>
[probe] SAME      layers/common/attention_interface.py     x6
[probe] SUMMARY same=6 drift=0 missing=0
[probe] image matches the patch anchor exactly
[rope] ROPE_FIX=not_needed        (or =applied on an older build)
```

### `install-only`

```
[verify] A. byte identity of overlay targets      -> 6x OK
[verify] B. live import of the promoted chain
[verify]   OK   tpu_inference.layers.jax.linear.P22XK_MATMUL_ACTIVE=True
[verify]   OK   tpu_inference.models.jax.qwen3.P22XK_RMSNORM_ACTIVE=True
[verify]   OK   tpu_inference.models.jax.qwen2.P22XK_SWIGLU_ACTIVE=True
[verify] OVERLAY VERIFIED
```

### `gate-only`

Reference numbers from the validated 4-chip v5p host — **your topology may legitimately
differ, which is the point of measuring**:

```
[T1.PATHWAYS] required=1 initialized=1 status=ok
[t1.devices] count=64 kind=TPU v5 platform=proxy
[waycount.mesh] width=4 shape=(16, 4) devices=64 unique=64 full_slice=1
[waycount] width= 4 replicas=16 depth=  8 arm=replicated ...
[waycount] width= 4 replicas=16 depth=  8 arm=stock-ar   ...
[waycount] width= 4 replicas=16 depth=  8 arm=f4-fixed   ...
[canonical-op] depth= 1 ... differing_bytes=0/... SAME
[canonical-op] depth= 2 ... differing_bytes=0/... SAME
[canonical-op] depth= 4 ... differing_bytes=0/... SAME
[canonical-op] depth= 8 ... differing_bytes=0/... SAME
[canonical-op] VERDICT: PASS
[mesh] slice_count=1  MULTI_SLICE=0
[mesh] create_device_mesh(shape=(4,)) -> ids=[0, 2, 1, 3]   REORDERED=1
[bucket] SET MIN_TOKEN_BUCKET=<value for your dp geometry>
```

The generic way-count table is diagnostic: its `replicated` arm determines whether a dirty
Pathways result requires TP reduction at all.  It does not call the promoted Qwen operators, so
even a dirty F4 row cannot reject the production path by itself.  The hard gate is P1b:
`[canonical-op] VERDICT: PASS` requires exact primals and live gradients at all four depths.
Byte counts are a binary gate and saturate; use rel-L2 / one-minus-cosine / max-abs only to compare
dirty generic arms.

### `dp-gate-only`

Use `cluster/profiles/qwen3-8b-dp16-tp4-admission.env`. The expected terminal markers are:

```
[P32.DP] CONFIG dp=16 tp=4 local_samples=16 global_samples=256
[P32.DP] MESH ids=(...64 ids...) shape=(16, 4) full_slice=1
[P32.DP] CHECKS {...}
[P32.DP] OBSERVATIONS {...}
[P32.DP] UPDATE {...four SHA-256s...}
[P32.DP] DECISION ...
[P32.DP] VERDICT PASS
[dp-gate] SAME_SESSION PASS artifact=...
```

`PASS` means fixed-placement repeatability and replica equality only. Read
`auto_regroup_exact`: if false, the same samples assigned to different DP ranks are not bitwise
invariant. Do not relabel that as a failed fixed-placement run or as arbitrary batch invariance.

The first run leaves `CANON_EXPECT_TRAIN_MESH_IDS` empty to measure placement. Pin the printed ids
and rerun. Only the pinned rerun is admission evidence.

### `run`

```
[run] PATHTRACE fixed_ar=<2 x layers>  embed=<>=1>  logprob_m=<>=1>
```

---

## 3. What counts as red

Ranked by how easily each is mistaken for success.

| Symptom | Meaning | Action |
|---|---|---|
| **`JOBSET_ATTEMPT` is not 0, or is unknown** | The JobSet restarts on failure and a red gate is a failure, so this log may be from a later attempt while `kubectl logs` shows only the current pod. A verdict from a retried run is not evidence of determinism -- it is evidence that one attempt out of several passed. | Report the attempt number with the verdict. For a strict single-shot gate set `failurePolicy.maxRestarts: 0` for that run. |
| **Missing successful `[T1.PATHWAYS]` marker** | The proxy backend was not proven initialized before JAX import. A local/direct-TPU fallback would test a different runtime. | **Void the topology run.** In proxy mode, import/initialization failure must exit nonzero. |
| **No `[PATHTRACE]` lines** | The intervention never executed. The chain is imported by path and by module name; a missing member does not raise — the engine silently uses its stock module while every switch still reads "on". | **Void the run.** Do not report its numbers. |
| **A gate printed no measurement line** | It did not run. Absence is never a pass. | Investigate before rerunning. |
| **`SKIP_TAINTED` appears** | The named earlier probe failed or raised; all later probes were intentionally suppressed because the Pathways session may be contaminated. | Fix the first failure and rerun from a fresh JobSet attempt. Never reuse earlier downstream rows from the failed session. |
| **P1 has no `shape=(16, 4) ... full_slice=1` row** | Production DP16×TP4 was not measured. A `devices[:4]` subset may also violate host boundaries. | Stop. Use the full-slice probe; do not disable Pathways subslice safety to force the subset through. |
| **P1b is missing or `[canonical-op] VERDICT: FAIL`** | The real promoted Qwen operator chain was absent, incomplete, had a dead gradient, or changed primal bits in the gradient program. | Stop before T2. Generic P1 numbers cannot override this hard gate. |
| **Step 75 starts another Pathways client** | The old split-process orchestration is active; the 64-chip attempt stalled after the second IFRT proxy connection. | Stop. T2 must run inside Step 70's unified Python session; Step 75 only validates its artifact. |
| **P32 `run` is refused** | Expected: the current shared mesh would be FSDP16×TP4 and the segmented VJP is not DP-local. | Do not bypass the refusal. Return the T1/T2 artifacts; implement the DP adapter in the next phase. |
| **T2-DP fixed repeat/replica check is false** | DP reduction or placement is not deterministic on this topology. | Stop before model initialization. |
| **T2-DP mesh lacks `shape=(16, 4) full_slice=1`** | The DP update probe did not attest the production topology. | Stop. Logical reshape fallback is forbidden; fix topology-aware construction. |
| **`THIRDPROG` red** | The forward-only and forward+backward programs are not the same family. Every downstream number in that run is meaningless. | Void the whole run, fix the config, rerun. |
| **`ROPE_FIX=unknown_version`** | Neither the old nor the new RoPE form was found. Somebody is about to patch a build nobody has looked at. | Stop. Inspect the file. Do not guess. |
| **`[probe] VERSION DRIFT`** | The image's `tpu_inference` differs from the patch anchor. Results can no longer be byte-identical to the signed evidence. | Decide deliberately; if you proceed with `CANON_ALLOW_IMAGE_DRIFT=1`, say so in the report. |
| **`MULTI_SLICE=1`** | Collectives cross slices, so XLA lowers a hierarchical reduction. This program family has **zero** coverage in this work. | Treat every bitwise claim on this topology as UNVERIFIED until re-measured here. |
| **Grepping a log and finding nothing** | Progress-bar control characters make `grep` treat the log as binary and drop **every** match silently — which reads exactly like "the intervention never fired". | Always `grep -a`. Cross-check a negative with `sha256sum`/`wc`. |

---

## 4. What to report back

Artifacts, not conclusions. A verdict without its raw log cannot be re-read later.

```
1. raw log path + sha256sum
2. the [entrypoint] JOBSET_ATTEMPT line verbatim -- without it a verdict cannot be shown
   to come from a first attempt rather than a retry
3. CANON_MODE, the resolved /tmp/canon-state/env.sh, and the image DIGEST (not the tag)
4. probe output verbatim:  [probe] SUMMARY / [rope] ROPE_FIX / [mesh] / [waycount] / [bucket] / [P32.DP]
5. any override used:      CANON_ALLOW_IMAGE_DRIFT, CANON_ALLOW_UNVERSIONED
6. exit codes of every step
```

Do not paste tokens or `.git/config`. Do not summarise a red gate as "mostly working".

---

## 5. Layout

```
cluster/
├── entrypoint.sh              the only thing the manifest calls
├── jobset-64chip.yaml         v5p 4x4x4, 16 worker pods x 4 chips
├── profiles/
│   ├── _canonical_engine.env  the switch set; shared by every profile
│   ├── qwen3-1p7b.env         model geometry (GSM8K release)
│   ├── qwen3-8b.env           model geometry (FrozenLake)
│   └── qwen3-8b-dp16-tp4-admission.env  P32 arithmetic + fail-closed contract
└── steps/
    ├── 00_env.sh              resolve config, strip secrets, refuse an incomplete set
    ├── 10_sync_repo.sh        verify the checkout is where it should be
    ├── 20_probe_image.sh      is this the image the patches were cut against?
    ├── 25_rope_fix.sh         apply the RoPE fix only if this build needs it
    ├── 30_install_canon.sh    build the chain in-container (--from-path, no docker)
    ├── 40_overlay_engine.sh   copy the six files over the engine's paths
    ├── 50_verify_overlay.sh   byte identity + live import of the chain
    ├── 60_wait_workers.sh     let TPU workers register
    ├── 70_run_t1.sh           topology + canonical-op + optional same-session DP probes
    ├── 75_run_dp.sh           validate same-session DP markers; no new Pathways client
    └── 90_run.sh              the workload, then the PATHTRACE tally
```

Change behaviour by committing to `steps/`, not by editing YAML.
