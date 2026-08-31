# P58.29 — K15 disaggregated scan execution-mesh repair

Status: `LOCAL REPAIR COMPLETE / PINNED-IMAGE PASS / TARGET NOT RUN`

## K15 evidence

K15 ran on 128 v5p devices split into two disjoint 64-device roles. The raw
evidence is authoritative about geometry:

```text
[P34.TOPOLOGY] PASS rollout_devices=64 trainer_devices=64
Rollout Mesh: dp=8,tp=8
Train Mesh: dp=8,tp=8
```

The incident package's `DP32xTP4` prose is a stale label and is superseded by
the raw device inventory above. The immutable package is intentionally not
rewritten:
`canon-zero-tim/evidence/p58_k15_disaggregated_mesh_scan_incident/`.

The run completed all 128 multi-turn trajectories: 116 ended naturally, 12
reached max turns, and none timed out or failed in the environment. It solved
3 tasks, produced 31 nonzero-advantage samples (24.2%), and collected 407,262
action tokens. Rescore-B and strict Step-0 pre-alignment passed exactly:

```text
[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=407262 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)] diff_bytes=0 diff_elements=0 hash=1ef8b0406cb23d242698ebaf3c8a982e01dfdb8d7d91244cf5ef025fa25890d9
```

The first segmented backward then failed in `run_layers_fwd_tape_scan` while
tracing `zt_tr_fwd_scan`. Its arguments lived on trainer devices
`[2, 3, 18, 19, ...]`, but a nested `shard_map` was constructed from rollout
devices `[0, 4, 8, 12, ...]`, so JAX rejected the incompatible devices.

## Root cause and repair

`_P28SegmentedEngineForward` already entered
`_canonical_fixed_ar_execution_mesh` for eagerly built embed/layer/norm/head
callables. Four scan JITs are created lazily after initialization and bypassed
that binding:

- `_layer_scan_fn`;
- `_layer_tape_scan_fn`;
- `_p71_fwd_scan_fn`;
- `_layer_rev_scan_fn`.

During their first trace, the installed linear/RMSNorm path therefore read the
serving value of `linear._CANON_MESH` even though operands were trainer-sharded.

The repair promotes the existing binding closure to the instance method
`_bind_execution_mesh` and applies the same scoped source-to-trainer mesh
binding to all four lazy scan callables. It does not add a flag or alter model,
data, sampling, loss, precision, optimizer, token transport, DP/TP geometry,
or the alignment contract. The colocated path returns the original callable
by identity and creates no extra wrapper or program boundary.

The disaggregated path necessarily holds the existing fixed-AR mesh scope
while each scan callable is dispatched; this is the same bounded global-mesh
discipline already used by the other segmented callables.

## Local validation

Local changes are based on operator parent
`55553dfe0c3c895de81c66191e5082ed9ec41a32`; they are not published.

- A forced-four-device positive executes all four scan methods with rollout
  devices 0–1 and trainer devices 2–3. It observes four trainer-mesh scopes,
  finite outputs, and no result placed on the rollout devices.
- A colocated negative proves `_bind_execution_mesh` returns the original
  callable by identity and preserves `None`.
- Both focused tests pass 2/2 in the digest-pinned dependency image.
- `P34_STATIC_PASS suites=10`.
- Flag audit passes `declared=409 actual=409 unique=409`; no flag was added.
- The complete digest-pinned P58 image gate exits zero and includes
  `disaggregated_scan_mesh=2` plus `P58_EXACT_IMAGE_CPU_PASS`.

## Claim ceiling and K16 promotion

This closes the source/image exception only. No direct TPU, Pathways target,
segmented backward, gradient, optimizer commit, checkpoint, or 1,000-update
campaign has run from the repaired source. It is not a Zero-TIM target PASS.

After separate approvals for commit/push, matching-image publication, and a
fresh target launch, K16 must use the final clean remote readback SHA. It must:

1. preserve the K15 TiTO, 1,012-row clean-data, rollout, reward, Rescore-B, and
   exact A=B=C receipts;
2. cross the former first `zt_tr_fwd_scan` trace without a serving/trainer
   device mismatch;
3. complete segmented forward and reverse with finite nonzero gradients;
4. produce exactly the intended first optimizer transaction and checkpoint
   receipts before any broader training claim.

Rollback is limited to removing the four lazy-scan mesh bindings and their
two regressions. Historical K15 evidence must remain unchanged.
