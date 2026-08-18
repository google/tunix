# P38.2w1 — P38s22 offsite evidence audit

Status: implementation and local/fake-GCS gates complete; remote audit not
run. No commit, push, TPU launch, backward, optimizer work, or GCS mutation
has occurred.

## Entering evidence

P38s22/source `ee0154b38ab81b2b4ee3eac35c65ed380aa744f6`
completed three frozen 64-TPU rounds with exact B-C and A-B red in every
round. The endpoint records and capsules establish the analysis-grade target
decision that `BF16_BF16_F32` does not eliminate the carrier.

The latest returned evidence commit does not close the signed durability
contract:

1. every returned archive SHA equals its capsule NPZ SHA, which is
   incompatible with `p38_evidence_archive.py` and the round-stage inventory;
2. the hand-written receipt carries stale P38s21 action counts for rounds 0
   and 1; and
3. the returned terminal classification has no raw terminal records or
   execution provenance, while the target run explicitly forbade that
   observer.

`sha256sum -c` over the returned Git files proves only that those returned
files have not changed; it cannot prove that their fields match the immutable
GCS objects.

## Deliverable

One contract-driven wrapper performs every remote read and calculation and
returns one compact, SHA-sealed directory. The remote agent supplies no
judgment and no manually transcribed field.

Files:

- `scripts/p38s22_offsite_audit_contract.json` — immutable source and expected
  target facts;
- `scripts/run_p38s22_offsite_audit.sh` — read-only GCS transport and single
  operator command;
- `scripts/audit_p38s22_offsite.py` — independent root/round/archive/capsule
  auditor and verdict writer;
- `tests/p38_serving/test_p38s22_offsite_audit.py` — fake-GCS positive and
  capsule-as-archive-SHA, mislabeled archive, missing completion, cross-round
  incident, staged/root record-identity, and orphan observer-pair negatives;
  and
- `P38S22_OFFSITE_AUDIT_RUNBOOK.md` — background-free operator card and exact
  return inventory.

## Local gate

```bash
set -euo pipefail
python3 -m unittest \
  canon-zero-tim/tests/p38_serving/test_p38s22_offsite_audit.py -v
python3 -m unittest \
  canon-zero-tim/tests/p38_serving/test_p38_evidence_archive.py -v
python3 -m py_compile \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/audit_p38s22_offsite.py \
  canon-zero-tim/tests/p38_serving/test_p38s22_offsite_audit.py
bash -n \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38s22_offsite_audit.sh
python3 -m json.tool \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38s22_offsite_audit_contract.json \
  >/dev/null
git diff --check
```

The negative must return rc 4 and a valid sealed `INCONCLUSIVE` directory; a
crash or missing receipt is a gate failure.

Current local result: all seven offsite-audit scenarios plus the four
deterministic-archive tests pass (`11/11`). The broader P38 discovery has 85
runnable tests passing; only the renderer module is `TARGET NOT RUN` in this
host interpreter because optional dependency `metrax` is absent. No renderer
code changed in this phase.

## Remote gate

Run only the exact command in `P38S22_OFFSITE_AUDIT_RUNBOOK.md` from the
user-approved published SHA. The generated `SHA256SUMS` must verify locally
after return.

| Remote result | Decision |
|---|---|
| PASS; generic preset rejected; all receipts/capsules close | Admit the P38s22 discriminator and start fixed-tile Pallas lm-head locally |
| rc 4 naming a wrong returned receipt while the immutable source is valid | Correct only the Git receipt from the returned mechanical output; no TPU rerun |
| rc 4 naming a missing/corrupt immutable source object | Keep P38s22 analysis-grade and preserve the failure; do not infer signed durability |
| Any terminal classification without raw terminal inputs | Unadmitted by construction; retain P38s21 as the interval-localization source |

## Claim ceiling

This phase may establish only that the P38s22 immutable source is complete and
that the generic lm-head algorithm preset failed its registered causal test.
It cannot turn the P38s22 no-observer arm into new terminal evidence and cannot
prove that a dedicated fixed-tile kernel will repair A-B.

## Rollback

The audit is read-only and default-off. Before publication, discard only this
phase's uncommitted files. After publication, revert its single documentation
and tooling CL. Never delete or overwrite the immutable P38s22 GCS source or
the previously returned evidence directory.
