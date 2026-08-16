# Recipe lifecycle

Load this reference when adding or modifying a GSM8K, FrozenLake, DeepSWE, or other real workload.

## Contents

- Current operator entry points
- Current P38/P39 routing
- Add one recipe
- Real-run preflight
- Promotion ladder
- Failure handling

## Current operator entry points

| Workload | Profile/runbook | Local gate | Target status source |
|---|---|---|---|
| GSM8K DP16xTP4 | `cluster/P33_QUEUE.md`, `profiles/qwen3-1p7b-dp16-tp4-gsm8k.env` | `tests/p33_workloads/run_cpu.sh` | P38 handoff/state and raw classifier; warning-only full is convergence-only |
| FrozenLake DP16xTP4 | `cluster/P33_QUEUE.md`, `profiles/qwen3-8b-dp16-tp4-frozenlake.env` | `tests/p33_workloads/run_cpu.sh` | P38 handoff/state and strict raw classifier |
| DeepSWE DP16xTP8 per role | `cluster/P34_DEEPSWE_RUNBOOK.md`, `profiles/qwen3-32b-dp16-tp8-deepswe.env` | `tests/p34_deepswe/run_static.sh` and `run_exact_image.sh` | P39 handoff/state and strict raw classifier |

Treat this table as routing, not as a PASS claim. Read status from the exact revision.

## Current P38/P39 routing

Use `tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md` as the source of truth when that task
exists.

1. Keep the P38 GSM8K/FrozenLake ledger separate from P39 DeepSWE. Never use one target's green
   result to promote another.
2. For strict P38 root-cause work, apply only the manifest named by the current handoff. The current
   serving-capture ladder is stock first, no backward, zero commits; a combined U or page-table
   counterfactual is forbidden until stock exactly reproduces the durable A/B record.
3. FrozenLake remains strict. Do not apply FrozenLake full until the serving carrier is repaired
   and a fresh backward-no-commit gate passes.
4. GSM8K full may be separately admitted with `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1`. Archive every
   alignment warning and classify only convergence behavior; retries restart from step zero when
   checkpointing is disabled.
5. DeepSWE must use the published `yuxzhang/canon-zero-tim` source SHA. Treat
   `yuxzhang/deepswe-quality-fix` as workload-reference provenance only. Start with
   `backward-no-commit`; no P33 result substitutes for its cross-role weight, A/B/C, gradient, or
   DP16xTP8 target gates.

## Add one recipe

Implement one concern at a time:

1. Add a model profile containing model id, geometry, dtype, and model-specific shim selection.
2. Add a workload profile containing topology, prompts, generations, lengths, scheduler limits,
   local/global M, VJP sequence count, update count, optimizer policy, evaluation policy, and
   default-off admission switches.
3. Add a pure contract module for arithmetic and topology. Keep secrets and cloud names out.
4. Extend the renderer with exact source SHA, image digest, input digest, fresh run id, stage,
   node pools, PVCs, and secret references. Refuse floating values and output overwrite.
5. Extend the runtime entrypoint. Initialize Pathways once before JAX and source exactly one
   reviewed profile.
6. Extend the learner/adapter only where the real workload needs a new semantic contract.
7. Add a classifier with exact counts, hard boundaries, fatal patterns, Attempt 0, W&B online,
   deterministic gradient/update checks, and fail-closed missing-line handling.
8. Add positive, one-fault negative, stale-evidence, retry, and adjacent-recipe tests.
9. Document rollback and preserve target status as `TARGET NOT RUN` until a real artifact passes.

## Real-run preflight

Verify before rendering:

- exact clean branch and 40-character source SHA;
- pinned client image digest and anchored engine package;
- model/input/whitelist digest and mounted path;
- single-slice topology and role split;
- DP/TP arithmetic, local/global M, and scheduler capacity;
- exact scheduler units: global `MIN_TOKEN_BUCKET`, per-rank token/request maxima, expected bucket
  list, expected global request capacity, and expected precompile count;
- parameters replicated over DP and sharded only over registered TP axes;
- fixed DP gradient reduction order and exact post-reduction replicas;
- optimizer location/commit transaction and checkpoint capacity;
- W&B online required, monotonic step metric, and no secret persistence;
- W&B/HF credential values, Kubernetes Secret references, and propagation wiring remain
  user-owned; never modify them without explicit credential-specific approval;
- evaluation/prefix cache/TIS/importance-correction policy explicitly signed;
- zero retries, no stale output path, and persistent raw artifact destination.
- reviewed `PriorityClass` on every Pathways head and worker Pod, verified read-only before apply;
- one Pathways client session, complete server/worker readiness, explicit proxy backend target,
  and no short-lived JAX probe that consumes the session before training;
- persistent-cache namespace includes source/image/runtime/topology/profile/shape/XLA flags;
  cache pull/upload failures are visible, periodic and EXIT sync are installed, and compile/cache
  miss logging is enabled for cache validation.

Run the pinned image's actual `get_token_paddings` implementation during exact-image admission.
Require equality with the registered list. A test that only proves the desired bucket is present
does not prevent accidental compilation of larger buckets.

At target runtime, require the exact prepared bucket list and backbone precompile count before
training. Then reject any unexpected larger-shape JAX compile or cache miss. Do not set
`SKIP_JAX_PRECOMPILE` to hide the cost; that only shifts compilation into the first training step.

## Promotion ladder

Use the cheapest valid ladder for the change:

```text
static -> CPU + negative controls -> exact image -> forced mesh -> target operator
       -> backward no-commit -> one update -> steady updates -> approved full run
```

A user may approve a direct full run to conserve scarce resources. That approval changes the
launch order, not the classifier: the full run must still emit every admission, forward,
backward, update, W&B, and postflight measurement in the same Attempt 0 artifact.

The registered GSM8K convergence exception is different: its classifier intentionally accepts
finite alignment warnings and labels the run `convergence-only`. Do not describe this as a strict
full-run shortcut or extend it to FrozenLake/DeepSWE.

## Failure handling

- Preserve the full raw log before editing.
- Identify the first hard failure; later green markers are tainted.
- Separate infrastructure, dependency, compiler, topology, numerical, and training failures.
- Add the failure as a regression fixture only after removing credentials and unstable metadata.
- Fix one variable and rerun with a fresh run id. Never overwrite prior evidence.
