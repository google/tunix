# P58.14 — DeepSWE Zero-HP disaggregated trainer-mesh binding

Status: SOURCE PUBLISHED + DEPENDENCY-IMAGE CPU GATE PASS; 128-TPU TARGET
RETRY NOT RUN.

## Trigger

Immutable attempt `p58z03` (`canon-p58-ds4b-zero-hp-full-p58z03`) ran source
`8eb65480d3705d96ab282799ad5a6c1901596248` on 128 TPU chips: rollout
DP8xTP8 on 64 devices and trainer DP8xTP8 on a disjoint 64 devices. All 128
Step-0 trajectories returned. The P58.13 fixed head admitted global/local
M=`2048/256` and emitted its tracing receipts. The first hard failure was the
first canonical trainer old-policy-logprob JIT:

```text
ValueError: Received incompatible devices for jitted computation.
state['embedder']['input_embedding'].value ... device ids
[2, 3, 18, 19, ... 126, 127]
sharding_constraint inside jit ... device ids
[0, 4, 8, 12, ... 121, 125]
at canonical_qwen3_adapter.py:483 (_safe_sharding_constraint)
```

The fixed-head/Pallas `PATHTRACE` lines preceding this error were emitted
during JAX tracing. They do not prove that a 36-layer forward, VJP, or
backward executed. The attempt stopped before trainer-logprob completion,
alignment completion, backward, gradient validation, AdamW, optimizer commit,
or checkpoint. The shutdown-time `Qwen3ForCausalLM.modules` exception is
secondary.

Evidence is immutable under
`evidence/p58z03_device_sharding_error/`; `sha256sum -c SHA256SUMS` passes.

## Root cause

`VllmRollout` loaded the actor/trainer state on the trainer-role mesh but
constructed `Qwen3EngineForwardAdapter` from the rollout runner alone. The
adapter therefore captured the rollout mesh in input/cache/output sharding,
sampling, and the mesh-bound canonical log-softmax scorer. The differentiable
canonical forward later consumed trainer-state arrays in the same JIT. JAX
correctly rejected the two disjoint device sets before execution.

This is a role-placement bug, not a numerical mismatch, an ordering-only
issue, or a reason to colocate the production workload.

## Repair

- Pass the live trainer state when the canonical adapter is registered.
- Derive exactly one trainer `dp,tp` mesh from that state and construct an
  engine-axis execution mesh on the same trainer devices. Require rollout and
  trainer DP/TP shapes to match. Admit only identical-device colocated layouts
  or fully disjoint role layouts; reject partial overlap.
- Bind differentiable input, KV cache, sample processing, outputs, and the
  trainer canonical scorer to that execution mesh.
- Keep serving on the rollout mesh. Because `shard_map` closes over physical
  devices, disaggregated serving and trainer use two mesh-bound scorer
  instances produced by the same factory and exact math.
- Do not catch the JAX error, copy values through host, change loss/sampling,
  weaken alignment, or touch the Native path.

Required fresh-target receipts include:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
```

## Validation

- Python compilation and `git diff --check`: PASS.
- Forced four-CPU-device disaggregated test: two rollout devices plus two
  trainer devices execute a real `jax.jit(jax.value_and_grad(...))`; primal
  and gradient are finite and the gradient is nonzero. A partial-overlap
  negative fails closed.
- Existing colocated adapter forward/VJP regressions: PASS.
- Complete P58 dependency-image CPU gate on local image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=3 ... regressions=1
```

The first full-gate run exposed an independent stale assertion: `FLAGS.md`
contained 386 names while a prefix-cache test expected 385. The assertion now
tracks 386; its 31-test suite and the rerun full gate pass.

The local image exposes no `/dev/vfio`. These are construction and CPU
placement tests only; they do not prove Pathways, 128-chip TPU execution,
full Qwen3-4B Pallas forward/backward, A=B=C, or an optimizer transaction.

## Next gate

Implementation commit
`dce0e93777548b7623e4f41702144f8d00f242f5` is published on
`yuxzhang/canon-zero-tim`. Image publication, Kubernetes
rendering/application, and TPU execution remain separately user-gated. Build
and pin a matching image, rerun the complete image gate, pass sandbox
admission, obtain separate launch approval, and use fresh Attempt-0 id
`p58z04`.
Never resume or overwrite `p58z01`, `p58z02`, or `p58z03`.

The fresh target must pass placement receipts, complete trainer old/current
logprobs, strict A=B=C, finite nonzero 16-group backward, and the coherent
update-0 transaction. A passing first update continues the same signed
1,000-commit job.
