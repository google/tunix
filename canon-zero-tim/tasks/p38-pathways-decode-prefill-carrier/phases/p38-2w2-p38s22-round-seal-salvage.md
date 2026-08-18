# P38.2w2 — P38s22 independent-round seal salvage

Status: complete. The returned round-first audit is `PASS` for the scoped
three-round forward discriminator. Root run-level postflight remains
unadmitted.

## Entering evidence

P38.2w1/offsite-audit-v1 is a complete, self-sealed `rc=4` return. Its tool and
contract SHAs match the published source. It failed because root
`SHA256SUMS` was absent; `COLLECTED.json` and `COMPLETE.json` were also not
returned.

The same receipt preserves all three `sealed-and-verified` round markers and
their manifests. Each manifest has 10 sorted logical members, matches its
marker SHA/count, and names the preregistered capsule SHA. The v1 auditor did
not validate the actual tar bytes because its root-first ordering failed before
the round loop.

## Deliverable

One new append-only round-first audit:

- immutable contract registering all three archive, manifest, and capsule SHAs;
- read-only wrapper with a URI-free acquisition ledger;
- auditor that verifies all three deterministic tar objects and recomputes the
  endpoint decision from their sealed capsules;
- fake-GCS positive with root postflight absent;
- fail-closed missing archive, archive mutation, and cross-round-ledger
  controls; and
- operator-only runbook with an exact return inventory.

## Admission boundary

The three round seals are independent scientific durability boundaries. A
three-round PASS admits only the P38s22 forward causal discriminator:

`BF16_BF16_F32` did not remove A-B while B-C remained exact.

It does not admit:

- root run-level postflight;
- terminal/lm-head interval localization beyond P38s21;
- backward or optimizer completion; or
- a fixed-tile repair.

Root receipt presence is reported but never changes this phase's claim.

## Local gate

```bash
set -euo pipefail
python3 -m unittest \
  canon-zero-tim/tests/p38_serving/test_p38s22_round_salvage.py \
  canon-zero-tim/tests/p38_serving/test_p38s22_offsite_audit.py \
  canon-zero-tim/tests/p38_serving/test_p38_evidence_archive.py -v
python3 -m py_compile \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/audit_p38s22_round_salvage.py \
  canon-zero-tim/tests/p38_serving/test_p38s22_round_salvage.py
bash -n \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38s22_round_salvage.sh
python3 -m json.tool \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38s22_round_salvage_contract.json \
  >/dev/null
git diff --check
```

Result on the implementation worktree:

- focused round-salvage/offsite/archive suite: 16/16 PASS;
- broader P38 serving discovery: 90 runnable tests PASS;
- renderer collection: `TARGET NOT RUN` in this host interpreter because the
  optional dependency `metrax` is absent; this phase does not touch renderer
  code; and
- Python compilation, Bash syntax, contract JSON parsing, and
  `git diff --check`: PASS.

## Remote gate

Run only `P38S22_ROUND_SALVAGE_RUNBOOK.md` from the user-approved published
SHA. Expected current-source result:

- three actual archives verify against preregistered SHAs;
- all 30 logical round members verify;
- totals are 143,464 actions, 66 A-B elements, 111 A-B bytes, and 0 B-C;
- root postflight receipts remain absent and unadmitted; and
- verdict is
  `ROUND_SEALED_GENERIC_LM_HEAD_ALGORITHM_PRESET_REJECTED`.

Any missing/corrupt required round object returns a sealed rc=4 and blocks
promotion. It does not authorize a TPU relaunch automatically.

## Next phase

The returned bundle and SHA inventory passed review at source `c04013bd`:
three actual archives and all 30 logical members verify, and the sealed
capsules recompute 143,464 actions, 66 A-B elements, 111 A-B bytes, and exact
B-C. Begin P38.2x: dedicated fixed-tile/fixed-order Pallas lm-head, first with
real Qwen3-8B weights on one v5p, then one three-round P38s23 target only if
the one-host construction and negative gates pass.
