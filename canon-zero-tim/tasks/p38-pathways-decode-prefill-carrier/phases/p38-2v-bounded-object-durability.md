# P38.2v — Bounded-object diagnostic durability

Status: host implementation and local fake-GCS gates complete; publication and
the replacement 64-TPU run require separate user approval.

## Entering evidence

P38s20/source `bea31f36655b137d7ab47ba94095cadda5b586ba` proved that the
4-GiB terminal-discriminator bound is sufficient. Round 0 produced 873 seam,
873 tail, and 873 terminal records and reached the numerical precheck. It then
timed out after 900 seconds waiting for the round durability ACK.

The failure was an object-count transport bug, not a numerical or Pathways
failure. The live worker entered snapshot sequence 15 before servicing the
round request. That snapshot copied and serially uploaded every accumulated
observer JSON/NPZ. The round path would have repeated the same pattern and
then downloaded every object again. Round 0 therefore represented about 5,246
logical files and more than ten thousand individual GCS operations across
upload and verification.

The admitted Round-0 numerical facts are:

```text
N_action=49,451
S_decode_vs_S_prefill=63 differing bytes / 41 differing elements
S_decode_vs_S_prefill max_abs=0.08359146118164062
S_prefill_vs_T_old=0 differing bytes / 0 differing elements
backward=0
optimizer_commits=0
```

P38s20 is `INCONCLUSIVE_DURABILITY_SEAL_TIMEOUT`. It has no rounds 1/2,
terminal classification, controlled exit, `COLLECTED.json`, or
`COMPLETE.json`.

## Deliverable

Replace file-per-object transport without changing any observer, model,
sampling, scoring, or alignment computation:

1. Build a sorted, self-verifying `SHA256SUMS` over every logical file.
2. Store those files and the manifest in one deterministic flat tar archive.
3. Upload one archive plus one manifest, read the archive back once, and verify
   both the archive SHA and every inner logical-file SHA.
4. Write `LIVE.json` or `ROUND_COMPLETE.json` last. Each remote snapshot or
   round therefore has exactly three objects.
5. Scope periodic observer snapshots to the current diagnostic round. A sealed
   round is not re-uploaded as observer payload in the next snapshot.
6. Service already-published round and terminal requests before starting a
   periodic snapshot.
7. Fail before archive construction when the target filesystem lacks the
   computed tar size plus a 16-MiB safety reserve.

The archive is uncompressed. NPZ payloads are already compressed, and the
failure was GCS object count rather than transfer bytes. Avoiding compression
also keeps archive construction deterministic and bounded in CPU cost.

## Local gate

Run from the package root:

```bash
set -euo pipefail
python3 -m unittest tests.p38_serving.test_p38_evidence_archive -v
bash tests/p38_serving/test_gcs_persistence.sh
bash tests/p38_serving/test_postflight.sh
bash -n \
  tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh
python3 -m py_compile \
  tasks/p38-pathways-decode-prefill-carrier/scripts/p38_evidence_archive.py
```

The archive unit test creates 5,246 logical files twice and requires identical
archive bytes/SHA. It also verifies extraction, rejects a bit flip, and rejects
a missing manifest member. The fake-GCS gate requires exactly three remote
objects per live snapshot and sealed round, verifies the read-back archive,
and proves independently sealed rounds survive an abrupt worker exit.

This phase is host-only. A model or TPU one-host replay is not a stronger test
of the transport and is not required; the terminal observer's real-v5p
exactness and off/on neutrality receipt remains unchanged.

## Target gate

One replacement stock run, registered as P38s21, keeps the P38s20 numerical
envelope unchanged:

```text
attempt=0
DP16xTP4
concurrency=256
prefix_cache=off
seam_mode=layer
terminal_tail=1
terminal_discriminator=1
terminal_max_bytes=4294967296
frozen_rounds=3
backward=0
optimizer_commits=0
```

Admission requires all of the following:

```text
PRECHECK_ROUND_COMPLETE=3
ROUND_COMPLETE=3
LIVE_ROUND_PASS=3
remote_objects_per_round=3
p38_terminal.classification.json=present and capsule-scoped
controlled_exit=42
COLLECTED.json=present
COMPLETE.json=present
all returned SHA checks=PASS
```

Any timeout, missing marker, ambiguous red join, archive mismatch, or partial
round remains `INCONCLUSIVE`. Do not increase the 900-second timeout as the
primary repair and do not reuse the P38s20 run id or attempt prefix.

## Decision table

Once the complete classifier exists, use the P38.2u table unchanged:

| First differing stage | Decision |
|---|---|
| `pre_lm_head_hidden` | Reopen the upstream seam |
| `lm_head_logits` | Localize lm_head/program envelope |
| `vocab_block_reduction` | Localize raw fixed block reduction |
| `logits_processing` | Localize raw-to-processed transform |
| `processed_vocab_block_reduction` | Localize processed reduction |
| `production_tail_only` | Localize production gather/subtract wiring |
| multiple or missing joins | Preserve all signatures or return `INCONCLUSIVE` |

## Claim ceiling and rollback

This phase repairs evidence transport only. It does not repair A-B and cannot
close zero-TIM. Roll back by reverting the archive-transport change; never
work around it by dropping SHA read-back, completion-last ordering, or the
per-round ACK.
