# Phase 7 — P32 DP16×TP4 update admission

Status: **local implementation + CPU gate PASS; 64-chip TPU NOT RUN**
Date: 2026-08-06

## Contract

- 64 devices = replicated data parallel 16 × tensor parallel 4.
- Global prompt batch 32, 8 generations, 256 trajectories; each DP replica owns two prompts
  and one 16-trajectory local reverse batch.
- Parameters, optimizer state and gradient accumulator are replicated over DP and sharded only
  over TP.  FSDP is explicitly out of scope.
- Initial release contract is repeatability under a fixed sample→rank mapping.  Arbitrary
  redistribution of the same samples is a stronger, separately measured property.

## Finding before the expensive run

The current trainer cannot yet execute that contract. `FL_SHARED_MESH=16,4` names its first
axis `fsdp`, so it would shard weights. The segmented adapter also walks all global trajectories
on the host. Therefore the P32 profile is admission-only and `CANON_MODE=run` fails closed.

## Implemented gate

`tests/t2_dp/probe_dp_update.py` runs a small real SPMD gradient and AdamW arithmetic over a
`(dp,tp)` mesh. It records:

1. exact repeatability with a fixed mapping;
2. equality of every post-reduction replica;
3. sensitivity to physical mesh order;
4. sensitivity to redistributing the same global samples;
5. gradient/parameter/Adam-moment SHA-256s;
6. a rank-dependent negative that the classifier must reject.

The fixed-order arm gathers small probe gradients and adds logical ranks serially. It is a
reference only, not a proposal to all-gather model-sized gradients.

## Local evidence

```bash
CANON_DP_PROBE_CPU=1 CANON_DP_SIZE=4 CANON_TP_SIZE=1 \
  CANON_DP_PROBE_LOCAL_SAMPLES=16 tests/t2_dp/run.sh
tests/t2_dp/negative_control.sh
```

- fixed mapping: stock, reference and auto-GSPMD repeat exactly;
- replicas: exact;
- injected rank fault: rejected;
- regrouped global batch: all three arms differ.

The regroup result proves fixed all-reduce order is insufficient for arbitrary placement
invariance: the local sample grouping has already changed before the DP collective.

The exact logical DP16×TP4 CPU arm also passes with `global_samples=256`. Its update hashes are:

```
gradient  4ca008cf019c6ebb62c71ad53915cfe48aad64e9ebf9915d5d31bf70ca98afb9
parameter 06bc3e598959c407894c036101e613302654ed4b007b868efac43dba8077189c
moment    448f88c2c0637e1f59fe6459dd9ff833415c3e26f6c314021e27103b23bf7509
variance  44aebc5693c8f8f31c377c6dfecfb53ce8cce627a333f0bf4496c8c194a96152
```

These hashes belong to CPU XLA and are a regression anchor, not expected TPU values.

## Remote gate and stop rule

Run `probe-only → install-only → dp-gate-only`. Preserve `$CANON_STATE/t2_dp.log`, resolved
`env.sh`, image digest and raw JobSet log. The first run measures mesh ids; pin those ids for a
fresh second `dp-gate-only` run.

No model initialization, backward or training is admitted by this phase. A remote green result
only admits implementation of the real replicated-DP adapter, followed by the six-stage ladder
in the task phase. Any missing marker, replica mismatch or repeat mismatch is a hard stop.

Rollback: unset/remove the P32 admission profile and step 75; the earlier TP4 profiles and all
production defaults are unchanged.
