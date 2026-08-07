# Runbook

One section per executable task. Each gives prerequisites, the exact command, what you should
see, what counts as a pass, what counts as red, and what to send back.

Read `START_HERE.md` first if you have not. Do the tasks in order — each one's output decides
whether the next is worth running.

---

## Task A — CPU gates

Proves the differentiable attention contract is mathematically sound: the pure-JAX function
whose Jacobian becomes the backward computes the *same function* as a full-prefill oracle.

**Needs:** any machine with Python and JAX. No TPU, no model, no container, no network.
**Takes:** under a minute.

```bash
tests/t0_cpu/run.sh
tests/t0_cpu/negative_control.sh
```

**Expected**

```
[selftest] value |chain-oracle| = 0.000e+00
[selftest] grad rel = 5.039e-16
[selftest] FD best rel = 1.106e-08
[selftest] VERDICT: PASS
[msq] value: ... |Δ|=0.000e+00
[msq] grad rel: dq=4.492e-17 dk=6.828e-17 dv=5.082e-17
[msq] VERDICT: PASS
===== T0 PASS (2 gates, 7 measurements) =====

  REJECTED (exit 1)   N1 silent gate (prints nothing)
  REJECTED (exit 1)   N2 nonzero value residual
  REJECTED (exit 1)   N3 gradient error above threshold
  REJECTED (exit 1)   N4 good numbers but VERDICT: FAIL
===== NEGATIVE CONTROL PASS -- run.sh rejects all 4 arms =====
```

**Pass:** both scripts exit 0. Value residuals **exactly** `0.000e+00` — there is no tolerance
on those. Gradient and FD figures within the thresholds printed in `run.sh`.

**Red:**
- any exit non-zero;
- a value residual that is small but not exactly zero — that is a different function, not a
  rounding difference;
- the negative control **accepting** any arm. That means the gate is not fail-closed and its
  green results carry no information. Report this before anything else.

**Does not prove:** anything about the real attention kernel. That is Task B and beyond.

**Send back:** both outputs verbatim, exit codes, `python3 -c "import jax; print(jax.__version__)"`.

---

## Task B — Topology admission probes ← *the next thing that needs doing*

Answers whether the canonical switch set transfers to your hardware at all. The release probes
run in one fail-stop Pathways session. Historical subset-mesh reproducers are skipped on slices
larger than four devices because they are not valid full-slice experiments there.

**Needs:** ≥ 2 TPU chips. **No model, no checkpoint, no engine, no image build.**
**Takes:** seconds of compute, a few minutes of compilation.

```bash
tests/t1_tpu/run.sh                       # host with docker: re-enters the pinned image
CANON_IN_CONTAINER=1 tests/t1_tpu/run.sh  # already inside a TPU-capable container
```

`XLA_FLAGS` must contain `--xla_allow_excess_precision=false`. The runner sets it and refuses
to continue without it — without that flag you are measuring a different program family than
production, and the numbers would not be comparable to anything.

**Expected** — reference values from the 4-chip v5p host where this was developed. **Your
topology may legitimately differ; that is the point of measuring.**

```
[waycount.mesh] width=4 shape=(1, 4) devices=4 unique=4 full_slice=1
[waycount] width= 2 replicas= 2 depth=  8 arm=replicated ... SAME
[waycount] width= 4 replicas= 1 depth=  8 arm=stock-ar   ... DIFFERS
[waycount] width= 4 replicas= 1 depth=  8 arm=f4-fixed   ... <MEASURED>
[waycount] width= 8 replicas= 8 depth=  8 arm=f4-fixed   ... <MEASURED>
[mosaic.compat] VERSIONS jax=... jaxlib=... pathwaysutils=...
[mosaic.compat] VERDICT: PASS
[canonical-op] depth= 1 ... differing_bytes=0/... SAME
[canonical-op] depth= 2 ... differing_bytes=0/... SAME
[canonical-op] depth= 4 ... differing_bytes=0/... SAME
[canonical-op] depth= 8 ... differing_bytes=0/... SAME
[canonical-op] VERDICT: PASS
[mesh] slice_count=1   MULTI_SLICE=0
[mesh] create_device_mesh(shape=(4,)) -> ids=[0, 2, 1, 3]   REORDERED=1
[bucket] SET MIN_TOKEN_BUCKET=<derived for your dp geometry>
[f4cost] width  ring MiB  F4 MiB  ratio ...
===== T1 COMPLETE -- all probes produced measurements =====
```

**How to read it** — this is the part that matters more than the exit code:

On 64 devices, generic P1 scans TP widths `2,4,8`. The TP8 rows are future-facing platform
diagnostics. The installed Qwen8B production P1b and T2 contracts remain DP16xTP4.

| Observation | Meaning |
|---|---|
| `(16,4)` full-slice attestation is absent on 64 devices | TP4 was not measured. A partial prefix mesh is not a substitute. |
| `replicated SAME`, `stock-ar DIFFERS`, `f4-fixed SAME` | TP reduction order is a sufficient carrier and F4 removes it at this point. |
| `replicated DIFFERS` | A Pathways/compiler carrier exists without TP reduction; do not attribute stock/F4 byte-count differences solely to all-reduce. |
| `stock-ar` is **already** `SAME` | The fixed-order tree repairs nothing at that point. |
| `f4-fixed` `DIFFERS` in generic P1 | F4 alone does not close the handwritten diagnostic graph. This is platform evidence, not a production-Qwen verdict. Continue only to the fail-closed P1b gate. |
| P1a reports an unsupported stable-Mosaic version | The JAX client and Pathways service release are incompatible. Stop before P1b/T2 and align the `jax-<version>` release family. |
| P1b is missing or `[canonical-op] VERDICT: FAIL` | Stop before T2. The promoted Qwen operator chain was not admitted on this topology. |
| `MULTI_SLICE=1` | Collectives cross slices and XLA lowers a hierarchical reduction — a program family with **zero coverage** here. Every bitwise claim on this topology is UNVERIFIED. |
| `REORDERED=1` | Placement permuted your device order. Use the printed order for `CANON_EXPECT_MODEL_MESH_IDS`; never inherit one from a different mesh shape. |
| `[bucket] SET MIN_TOKEN_BUCKET=` ≠ 256 | Your dp geometry needs a different global value. Copying 256 would silently unpin the bucket while every switch still reads "on". |

`differing_bytes` answers only bitwise SAME/DIFFERS and saturates. Compare dirty-arm magnitudes
with `rel_l2`, `one_minus_cos`, and `max_abs`; never infer that one arm is worse because it has
slightly more differing bytes.

**Pass:** generic P1 must print `T1 COMPLETE` and a full-slice attestation, but its numeric rows
are diagnostic and may be dirty. Production numerical admission additionally requires exactly
one `[canonical-op] VERDICT: PASS` after all registered depths report zero differing bytes and a
finite, nonzero weight gradient. `T1 COMPLETE` alone is not an admission.

**Red:**
- `T1 FAIL` — a probe produced no measurement line. It did not run;
- `SKIP_TAINTED` — a prior probe failed; every named later probe was deliberately not run;
- `REFUSING: XLA_FLAGS lacks ...` — fix the flags, do not work around it;
- `Device or resource busy` on `/dev/vfio` — something else holds the TPU. **Find out what
  before killing anything**; it may be someone's multi-hour job.

**Send back:** the full log + `sha256sum`; chip count and kind; `slice_count`; the complete
`[waycount]` table; the `[mesh]` id list; the `[bucket]` derived value.

---

## Task C — Cluster ladder (Pathways / GKE)

Four staged modes, each answering one question. **Read the previous stage's output before
promoting** — that is the whole point of the staging.

**Needs:** a GKE cluster with Pathways, and the manifest edited per `cluster/README.md`.

```bash
kubectl apply -f cluster/jobset-64chip.yaml
kubectl logs -f jobs/canon-zero-tim-v5p-64-pathways-head-0
```

| `CANON_MODE` | Costs TPU? | Answers |
|---|---|---|
| `probe-only` | no | Is this image's `tpu_inference` the build the patches were cut against? Does it need the RoPE fix? |
| `install-only` | no | Does the chain build, overlay, and actually load here? |
| `gate-only` | seconds | Task B, on the cluster. |
| `dp-gate-only` | minutes | Task B plus DP16×TP4 gradient/update repeatability and placement sensitivity. |
| `run` | yes | The workload in `CANON_RUN_CMD`; refused by the P32 admission-only profile. |

Start at `probe-only`. Full expected output, the red table, and the reporting format are in
`cluster/README.md` — that file is the operator guide for this task and supersedes this summary.

For P32 use `cluster/profiles/qwen3-8b-dp16-tp4-admission.env` and promote only through
`dp-gate-only`. The first 64-chip process measured the train mesh and the standard manifest now
pins it with `CANON_REQUIRE_TRAIN_MESH_PIN=1`. Run a fresh Attempt 0 and require the same 64-id
sequence. The 256-cluster manifest remains discovery-only until independently measured. Return
`$CANON_STATE/t2_dp.log`; do not bypass the run refusal.

**Two things to set before the first apply**, both in the manifest:
- the `jax-tpu` image, **pinned by digest** — a floating `:latest` means the same manifest runs
  on a different engine tomorrow, which is incompatible with a bitwise contract;
- `CANON_P32_EXPECT_MODEL_MESH_IDS`, from Task B's output **on this topology**.

---

## Task D — Round-trip against the signed numbers

The only check that proves the package is *complete*. Code review cannot do it: the failure
mode of a missing chain member is a silent fallback to stock, which reviews as fine and runs
as green.

**Needs:** the pinned image, a 4-chip v5p host, and (for D2) a checkpoint and W&B key.

### D1 — cheap arm, ~20 min

Install from this package into a fresh directory, then run the probe gates against it and
compare to the recorded values.

```bash
./install.sh /tmp/canon --from-image tunix_frozenlake_image:vllm-tpu0.25.0 --model qwen1p7b
# then the G1/G2 probe gates, per recipes/README.md, pointed at /tmp/canon
```

**Pass:** byte-identical to the recorded numbers, not "close":

```
K2.abc      hidden 0/10240   logits 0/303872
THIRDPROG   primal 0/303872
四 boundaries 0 differing bytes; w = r = w*r = 1; clip_hits = tis_hits = 0
```

### D2 — full arm, ~1 h

One real GSM8K training step (`recipes/README.md`, G1a). Exercises the engine patches, the
shim chain, the training-side hooks, all four boundaries and the release classifier.

**Pass:**

```
verdict     P26_GSM8K_G1A_PASS
N_action    248
gradient    0.2502315640449524      exactly
boundaries  three, 0 differing bytes each
```

**Red for either arm:** any number that differs. That is not a tolerance question — the whole
contract is bitwise. Before concluding the package is broken, rule out environment drift by
running the same gate through the original (unpackaged) paths on the same host and comparing.

**Send back:** raw log + `sha256sum`, the classification JSON, and the install directory's
`sha256sum -c MANIFEST.sha256` output.
