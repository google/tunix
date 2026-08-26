# P58.12 — JAX engine seed route and bounded abort cleanup

## Status

`ACTIVE / SOURCE PUBLISHED / CONSTRUCTION PASS / TARGET RETRY NOT RUN`

## Trigger

Fresh Zero-HP run `canon-p58-ds4b-zero-hp-full-p58z01` admitted all 128 TPU
devices, loaded the exact 1,012-task clean list, launched 128 R2E sandboxes,
and initialized vLLM. Its first Step-0 generation then failed before producing
a trajectory:

```text
ValueError: JAX does not support per-request seed.
```

P58.10 had placed seed 42 in `RolloutConfig.seed`. Tunix forwarded that value
to `SamplingParams.seed`, but the TPU/JAX vLLM backend permits only a global
engine seed. Abort cleanup then exposed an independent kubernetes-client
defect while decoding an empty DELETE error body:

```text
AttributeError: 'NoneType' object has no attribute 'decode'
```

The immutable Attempt-0 evidence remains under
`evidence/p58z01_attempt0_seed_exception/`. It contains no trajectory,
backward, optimizer transaction, or resumable trainer checkpoint.

## Repair contract

1. All P58 arms still use signed seed 42 for both dataset shuffle and rollout.
2. On vLLM JAX, the rollout seed is passed only as
   `rollout_vllm_kwargs["seed"]`, which becomes `EngineArgs.seed`.
3. `RolloutConfig.seed` / `SamplingParams.seed` must remain `None` on JAX.
   Any caller that supplies a per-request JAX seed fails before engine use;
   the value is never silently discarded.
4. The runtime emits exactly one route receipt:

   ```text
   [VLLM.JAX_SEED] PASS engine_seed=42 request_seed=none scope=engine-global
   ```

5. W&B and the durable manifest record
   `seed_scope=engine-global; async completion order not claimed`. Global seed
   equality does not claim identical asynchronous R2E completion order.
6. Cleanup tolerates only the exact kubernetes-client `None.decode` defect.
   It confirms Pod deletion by a bounded read-until-404 loop and retries the
   same exactly-scoped DELETE if the first response was ambiguous and the Pod
   still exists. Other `AttributeError`, API errors, or an unconfirmed Pod
   remain fatal.

No sampling distribution, clean data, geometry, numerical flag, loss,
optimizer, timeout, update horizon, or strict A=B=C gate changes.

## Exit gates

1. Python/Bash syntax and diff hygiene pass.
2. P58 source tests prove engine-level seed placement, absence of the request
   seed, exact manifest scope, and both mandatory postflight markers.
3. The vLLM exact-image test proves the installed `EngineArgs` accepts `seed`,
   the JAX route returns `(request=None, engine=42)`, and per-request or
   non-integer engine seeds fail closed.
4. R2E cleanup regressions prove successful-delete, response-ambiguous retry,
   exact empty-body tolerance, and unrelated-attribute failure behavior.
5. The complete P58 pinned-image construction gate and adjacent P34/P57 gates
   pass.

## Target retry

After separate commit/push, exact remote readback, matching image publication,
sandbox-capacity admission, and launch approval, use a fresh run id such as
`p58z02`. Do not resume or overwrite `p58z01`. Require both seed markers before
accepting Step 0, then apply the unchanged P58.11 first-update and 1,000-commit
gates.

## Local validation checkpoint

Python/Bash syntax and diff hygiene pass. Focused P58 sampler/route tests pass
7/7, one-host artifact tests pass 5/5, the bounded R2E cleanup regression
passes, P34 emits `P34_STATIC_PASS suites=10`, P57 adjacent coverage passes
146/146, and the latest-tip flag registry is unchanged at 385/385 with
`FLAG_AUDIT_PASS`. The complete dependency-bearing pinned image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exits zero with:

```text
P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 checked_vma=1 first_update=1 stable_clip=1 ... regressions=1
```

The image reports no `/dev/vfio`; no real JAX TPU generation was executed.

## Publication checkpoint

Implementation commit
`c10fbe0487d1f6635975b84806f1efdce6bc95c1` was pushed only to
`yuxzhang/canon-zero-tim`. Immediate local, `FETCH_HEAD`, and remote-tracking
readback matched that full SHA with ahead/behind `0/0`. `main` was not
modified or pushed. This publication does not authorize image publication,
Kubernetes mutation, or the fresh `p58z02` target.

## Claim ceiling

Local and exact-image gates are construction evidence only. This repair is
not proven on JAX TPU until a fresh 128-chip run reaches real generation. A
successful Step-0 rollout does not by itself prove strict alignment, backward,
the first optimizer transaction, or 1,000 committed updates.

## Rollback

Remove the P58 engine-seed assignment, the JAX seed-route validator/receipt,
and the exact cleanup exception handling together. Do not restore the rejected
per-request JAX seed or weaken the fixed-seed comparison contract.
