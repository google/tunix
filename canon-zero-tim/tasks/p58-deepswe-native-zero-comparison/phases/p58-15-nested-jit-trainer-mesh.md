# P58.15 — DeepSWE Zero-HP nested-JIT trainer-mesh binding

Status: SOURCE IMPLEMENTATION `f60cdd569c2737df6cb2968125c8e42680938981`
PUBLISHED + DEPENDENCY-IMAGE CPU GATES PASS; 128-TPU TARGET RETRY NOT RUN.

## Trigger

Immutable attempt `p58z04` used source
`3f159250c4781b3faafde238f768457a0478446b` with Qwen3-4B-Instruct,
rollout DP8xTP8 on 64 devices, and trainer DP8xTP8 on a disjoint 64 devices.
The P58.14 placement repair emitted both of its expected receipts. All eight
prompt groups and all 128 Step-0 trajectories returned in 1,709 seconds,
inside the 3,600-second batch deadline. Eight `MODEL_TIMEOUT` rows and one
`MAX_CONTEXT_LIMIT_REACHED` row were compact-filter statuses, not the crash.

The first hard failure remained the first trainer old-policy-logprob call:

```text
ValueError: Received incompatible devices for jitted computation.
trainer state devices: [2, 3, 18, 19, ... 126, 127]
jit inside jit devices: [0, 4, 8, 12, ... 121, 125]
canonical_qwen3_adapter.py:9135 run_nonempty
```

No trainer logprob completed. Alignment, backward, AdamW, optimizer commit,
and checkpoint did not occur. Evidence is immutable under
`evidence/p58z04_disaggregated_mesh_error/`; `sha256sum -c SHA256SUMS` passes.

## Root cause

P58.14 correctly rebound the adapter's explicit input, cache, output, sample,
and log-softmax shardings to trainer devices. It did not rebind the two nested
JIT callables created earlier by `tpu_inference.get_flax_model`:

- `model_fn` closes over rollout-mesh output shardings;
- `compute_logits_fn` closes over rollout-mesh output shardings.

The P58.14 CPU mock used ordinary Python functions for those two callables, so
it could not reproduce the hidden rollout-device closure. JAX therefore saw
trainer-resident parameters entering an inner executable fixed to rollout
devices and rejected the program before execution.

## Repair

- For a fully disjoint layout only, reconstruct the same live NNX model class
  with `nnx.eval_shape` on the trainer execution mesh. This creates graph and
  static metadata without loading a second parameter copy.
- Fail closed unless the reconstructed and live model states have identical
  tree structure, leaf count, shapes, and dtypes.
- Rebuild `model_fn` and `compute_logits_fn` with the original vLLM static and
  donation contract, but trainer-mesh output shardings. Runtime leaves still
  come exclusively from the mapped trainer state.
- Bind the installed fixed-AR linear/embed mesh global only during the locked
  trainer JIT trace and restore the serving value immediately afterward.
- Use the same reconstructed graph for the segmented trainer forward and
  backward path. Diagnostic probes that intentionally inspect the live runner
  remain rollout-bound.
- Keep colocated and Native paths unchanged. Do not host-copy parameters,
  weaken strict A=B=C, or change sampling, loss, optimizer, or the signed
  B8xG16 workload.

Required target startup receipts are now exactly one each:

```text
[CANON_ADAPTER.PLACEMENT] PASS relation=disjoint rollout_devices=64 trainer_devices=64 execution_role=trainer
[CANON_ADAPTER.PLACEMENT] trainer logprob scorer rebound relation=disjoint implementation=factory-identical mesh_bound_instances=2
[CANON_ADAPTER.PLACEMENT] trainer model callables rebuilt relation=disjoint graph=abstract-clone mesh_bound_jits=2
```

The full-run classifier rejects a missing or duplicate receipt.

## Validation

- Pulled operator source first to
  `a36cbd1b156e013a75af4071e91a238be49bc95b`, then over
  `98a2dfd9e8ece301374fdfb55518b3bc9ebef4d4`, and finally rebased before
  publication onto exact base `be758e68faa9db5b06be153a0656c4c861e3119f`.
  The intervening FrozenLake and M15-evidence changes do not overlap P58.
- Python compilation and `git diff --check`: PASS.
- Forced four-CPU-device nested-JIT regression: rollout devices 0-1 and
  trainer devices 2-3 execute `jax.jit(value_and_grad)` with finite primal and
  finite nonzero gradient: PASS.
- Forced four-CPU-device segmented trainer graph executes forward and a real
  layer pullback on trainer devices with finite nonzero parameter gradient:
  PASS.
- Partial-overlap layout rejection: PASS.
- Complete dependency-image gate is required to end with:

```text
P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=4 ... regressions=1
```

The pinned image has no `/dev/vfio`. These tests prove the missing nested-JIT
placement mechanism on CPU only. They do not prove the full Qwen3-4B graph on
Pathways, 128-chip execution, strict A=B=C, backward completion, an optimizer
transaction, or a checkpoint.

## Next gate

Do not reuse or overwrite `p58z01` through `p58z04`; none has a resumable
trainer checkpoint. Fetch/read back the latest operator tip and prove it
contains implementation `f60cdd569c2737df6cb2968125c8e42680938981`, build
a matching pinned image, rerun the complete image gate, and pass
sandbox-capacity admission. Kubernetes launch requires separate approval. The
next target must use fresh Attempt-0 id `p58z05`.

`p58z05` must emit all three placement receipts, complete trainer old/current
logprobs, pass strict A=B=C, complete finite nonzero 16-group backward, and
commit one coherent update-0 transaction. If update 0 passes, continue the
same signed 1,000-update job.
