# P58.26 — K09 full-startup one-host selector scope

Status: `IMPLEMENTED / HOST PASS / PINNED-IMAGE PASS / STARTUP TARGET PASS`

## Incident

K09 (`canon-p58-ds4b-zero-hp-full-k09`) ran source
`0b62b6bbd3d9fa44268c7640047d4b60047cb4d5`. It admitted TiTO, filtered the
signed 4,578-row source to the 1,012-task clean set, connected all 128 devices,
and constructed rollout/trainer DP8xTP8 meshes. It then stopped before rollout
at `examples/deepswe/train_deepswe_nb.py:1804`:

```text
NameError: name 'P58_Q4_TP4_TRAJECTORY_REPLAY' is not defined
```

The immutable incident package is
`canon-zero-tim/evidence/p58_k09_deepswe_unbound_variable_incident/`; raw-log
SHA-256 is
`c50b9212f12a23b6f13e0eae41911e21c2032833b2e953daa1fd9f3f605c041d`.

## Root cause and repair

`P58_Q4_TP4_TRAJECTORY_REPLAY` is a one-host-only diagnostic selector. It was
assigned only inside `if ONEHOST_SMOKE:` but was read later by shared
`ClusterConfig` construction. Full training has `ONEHOST_SMOKE=False`, so the
name was unbound.

The repair binds the selector to `False` before the one-host block and also
requires `ONEHOST_SMOKE and P58_Q4_TP4_TRAJECTORY_REPLAY` before deriving
replay update geometry. Production therefore receives ordinary P34/P58
trajectory geometry and cannot invoke the one-host replay helper. One-host
replay retains its existing true path. No flag, model, data, sampler, loss,
precision, optimizer, mesh, timeout, TiTO, or numerical bundle changes.

## Gates

- Executable AST regression: full mode returns replay geometry `None` and
  never calls the one-host helper; admitted one-host replay returns its signed
  geometry.
- Static scope audit: every uppercase name assigned in the one-host admission
  block and loaded later must have a top-level binding.
- P34 static and focused P58 one-host/renderer/sampler/TP4 contracts.
- Deterministic flag audit and Python/diff hygiene.
- Complete digest-pinned P58 exact-image gate.

Final construction source is
`0d224e4a0e8c278f1bf9f699af235fdea83ef327` plus the local P58.26 diff. Host
results are P34 static 10 suites, focused P58 49/49, script contract 10/10,
and deterministic flag audit 409/409 with `changed_names=0`. Image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exits zero with `P58_EXACT_IMAGE_CPU_PASS ... regressions=1`.

## Claim ceiling and next target

The source and image gates can prove the K09 `NameError` is mechanically
closed. They cannot prove rollout, DP8xTP8 numerical identity, backward, an
optimizer commit, checkpointing, or 1,000-step completion. A fresh, separately
approved Attempt-0 target is still required. Its first required post-K09
receipts are one real `[DEEPSWE.TITO] CONTINUATION`, a complete 128-row Step-0
journal, and strict pre-backward A=B=C.

K10 at source `0e954153cdfd21ee79ebf57eaa6afb4bf273aff0` supplied those
receipts and crossed the former K09 failure boundary. It is therefore target
PASS for this startup-scope phase. K10's later shared-workload identity failure
is owned by P58.27 and does not prove backward or optimizer completion.
