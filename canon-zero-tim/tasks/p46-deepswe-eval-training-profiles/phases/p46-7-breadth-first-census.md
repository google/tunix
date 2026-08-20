# P46.7 — breadth-first census before strict repair

- Status: implementation published as `365b46c1cd150839e3be1fd50adb33325fe3189f`; not target-run

## Trigger

The returned campaign has useful durable attempts but incomplete prompt
coverage. Strict resume retries invalid identities immediately and can spend a
long allocation on a small failing set before later prompts are sampled. The
operator chose one complete breadth-first pass first, with invalid repairs
deferred until every clean prompt/sample identity has durable evidence.

## Contract

- Qwen3-4B-Instruct-2507, signed 1,851-row clean data, N16, 16,384 response
  tokens, 50 steps, reward-only/no-logprobs, prefix cache off.
- One resident full campaign, 58 logical shards and 463 physical waves; one
  physical wave remains at most four prompts x N16 with a 3,600-second bound.
- Census schedules only identities with no durable attempt. A valid or invalid
  prior attempt suppresses another census attempt.
- Model/context/max-step/signed trajectory timeouts keep their existing valid
  unsolved classification. `FAILED`, environment/reward failures and malformed
  trajectories remain invalid and are listed for strict repair; they are never
  rewritten as reward zero.
- A bounded physical-wave timeout records the partial wave and traversal
  continues. The launch exits with `P46_EVAL_CENSUS_INCOMPLETE` if any identity
  still has no durable record. Relaunching the same census contract runs only
  those never-attempted identities.
- `P46_EVAL_CENSUS_PASS tasks=1851 scheduled_identities=29616 unattempted=0`
  proves breadth coverage only. Census snapshots live below `outputs/census/`
  and cannot create canonical washed lists.

## Resume migration gate

The current v6 campaign binds its old harness SHA and resume tag in every
fingerprint, so new census code must not append to that tag. After every old
producer is terminal and absent:

1. Copy, never move, the old `outputs/resume_contract.json` and complete
   `outputs/trajectories/*.jsonl` tree into a new tag's
   `imports/<import-id>/` staging directory.
2. Seal exactly `resume_contract.json` and every trajectory JSONL in
   `SHA256SUMS`; retain the source tree unchanged.
3. Start the new tag with `--frozen-v6-import-id <import-id>` exactly once.
4. Require `FROZEN_V6_IMPORT_PASS` before runtime preparation. Any digest,
   sampling/data/topology, per-logical fingerprint, attempt-sequence, reward,
   or provenance drift fails before TPU initialization.
5. Preserve the source `sampled_by` SHA and raw trajectory payload. Only the
   destination harness SHA and fresh resume tag change, and every copied row
   contains record-level migration provenance.

## Exit gate

- Local: all P46 CPU tests, adjacent P34/P44 gates and `git diff --check` pass.
- Publication: an explicitly approved commit is pushed only to
  `yuxzhang/canon-zero-tim`, then fetched/read back by exact 40-character SHA.
- Census target: the final census launch emits exact 1,851/29,616 coverage,
  `unattempted=0`, immutable snapshot digests and cleanup PASS.
- Strict target: relaunch the same new tag and exact source/harness contract
  with census/import controls omitted. Retry every identity without a valid
  record until 58 canonical summaries and
  `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16
  valid_trajectories=29616 logical_shards=58` exist.

CPU PASS does not prove TPU runtime, R2E sandbox throughput, census completion,
strict washing completion, or training-data readiness.
