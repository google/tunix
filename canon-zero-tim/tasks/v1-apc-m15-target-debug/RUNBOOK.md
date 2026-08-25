# M15 APC target carrier runbook

This runbook is for the remote execution agent. The agent runs checked-in
commands; it does not edit YAML, numerical code, or evidence. Large payloads
remain in GCS exactly like the earlier P38/lm-head investigation. Only small
machine-generated receipts are returned through Git or chat.

## What this run is for

One fresh target observation records enough information to replay a red
without running the FrozenLake environment or sampling new actions again:

1. all 256 final prompt/completion token streams and A/B/C logprobs;
2. every serving call's request dispatch order, DP rank/local slot, token
   prefix identity, logical position, and physical page table;
3. the exact first-red request/call/page-generation join.

The run still executes a real M15 rollout once because historical `m15i` did
not save these inputs. Later replay may skip environment generation, but must
still execute the real serving decode and independent full-reset B arm.

This carrier does not modify or repair RoPE, attention, KV values, lm-head,
loss, backward, or optimizer. Both JobSets stop before backward and commit.

## Attempt-0 through Attempt-2 incidents

Never relaunch source `eb58954f...`. That Attempt-0 command carried
`--p57_workload_candidate=m15 --p57_data_split=main`, but the rendered
environment omitted the matching signed `CANON_P57_*` fields. The workload
entrypoint therefore exited before learner construction, capture creation, or
any A/B/C numerical verdict. `INCONCLUSIVE` is the permanent classification.

The repaired source requires exact `m15/main` in both CLI and environment and
keeps the package-safe `python3 -u -m
examples.frozenlake.train_frozenlake_qwen3` entrypoint. This is a bootstrap
contract fix only; production APC remains off.

Never relaunch source `283cb67e...`. Attempt 1 proved that bootstrap repair
worked: all overlays and GCS preflight passed. It then stopped before learner
construction because `train_frozenlake_qwen3.py` reused the legacy P38 DP16
geometry for this different carrier. The old contract expected
`mini_batch_size=4`, token IS, workload `frozenlake`, and eight producer units;
the signed M15 target is `mini_batch_size=32`, no IS,
`frozenlake-dp8-tp8`, and one full producer unit. The current source keeps both
contracts separately fail-closed. This is still admission-only and is not an
APC numerical fix.

Never relaunch source `41a2043c...`. Attempt 2 was the first run to exercise
the real DP8xTP8 M15 serving envelope: it completed more than 1,800 serving
calls, all four standard tensor captures, and most of the 15-turn rollout.
It then exposed two observer defects, not a model mismatch:

- the incident ledger reached its 256 MiB ceiling at call 326 and emitted
  1,650 nonfatal capture errors;
- the drain tail entered the production `continue_decode` program, while the
  old observer asserted that every call in the process must be `standard`.

Do **not** remove `CANON_CONTINUE_DECODE=8`. The historical `m15i` production
red used that program; removing it changes the experiment. The current repair
keeps the four tensor records and generic request/incident ledgers
standard-only, admits continue-decode only after all four are complete, and
keeps writing the dedicated full replay chronology. A red
carrier must prove A used both `standard` and `continue_decode`; B must remain
the independent full-reset `standard` path. The M15-only signed ledger bound
is 2 GiB, based on Attempt 2's 268,192,266 bytes at call 326 and roughly 1,894
observed calls. Ordinary P38 renderer limits are unchanged.

## Approval boundaries

The following are four separate user decisions:

1. commit and push the prepared source;
2. run the exact-image gate;
3. launch the APC-off control;
4. after the control is green, launch the APC-on treatment.

Do not infer one approval from another. Do not launch from a dirty tree or an
abbreviated SHA.

## Render

After the source has been committed and pushed:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
SOURCE_SHA=<40-character-committed-sha>
RUN_ID=<new-unique-label>
OUT=/tmp/v1-apc-m15-${RUN_ID}
python3 canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  --source-commit "$SOURCE_SHA" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
sha256sum "$OUT"/*.yaml
```

Expected files:

```text
jobset-v1-apc-m15-off.yaml
jobset-v1-apc-m15-on.yaml
```

Never edit either file. The renderer fixes M15/main, seed 42, DP8xTP8,
32 prompts x 8 generations, concurrency 256, 15 turns, 4096 prompt tokens,
8192 response tokens, temperature 0.7, one diagnostic round, zero backward,
and zero optimizer commit. It also deliberately preserves
`CANON_CONTINUE_DECODE=8`, standard-only four-stratum tensor capture, and the
2 GiB M15 incident/replay byte bound.

The renderer and Step-00 resolver must reject any CLI/environment identity
split. A valid rendered arm carries:

```text
CANON_P57_WORKLOAD_CANDIDATE=m15
CANON_P57_DATA_SPLIT=main
--p57_workload_candidate=m15
--p57_data_split=main
```

## Optional exact-image admission

This is not a target numerical result and needs its own approval:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Expected post-fix terminal marker includes `apc_m15_carrier=44`.

## Launch order

Launch commands must be standalone. Do not append `tee`, a pipe, `&&`, or a
monitor.

First, with separate approval:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-off.yaml"
```

Wait for termination, then read the stored raw log. Continue only if the M15
classification is `CONTROL_GREEN`, B-C is zero, and the GCS attempt has all
three terminal markers.

Second, with a new approval:

```bash
kubectl apply -f "$OUT/jobset-v1-apc-m15-on.yaml"
```

Accepted treatment outcomes are:

- `FRESH_TARGET_RED_FROZEN`: a complete replay carrier must also be frozen;
- `TARGET_NOT_REPRODUCED`: one target observation was exact, no fix claim.

Any other classification is a hard stop.

## GCS layout

Each JobSet writes to:

```text
gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0/
```

Root completeness requires:

```text
PREFLIGHT.json
COLLECTED.json
COMPLETE.json
SHA256SUMS
run.log
pre-alignment.jsonl
serving-classification.json
serving-capture.tar
```

The large `serving-capture.tar` contains, for both clean and red runs:

```text
m15_producer_unit.npz
m15_replay_envelope.jsonl
m15_apc_target.classification.json
```

The complete `m15_replay_envelope.jsonl` must contain both `standard` and
`continue_decode` records for serving arm A, and only `standard` records for
serving arm B. `m15_full_replay_carrier/replay_contract.json` records the
mechanically checked `program_paths_by_arm` map. Missing the continue tail,
placing B on it, or observing any unknown path makes packaging inconclusive.

For a red treatment it must additionally contain:

```text
m15_first_red_replay/first_red_capsule.npz
m15_first_red_replay/first_red_contract.json
m15_first_red_replay/SHA256SUMS
m15_full_replay_carrier/replay_contract.json
m15_full_replay_carrier/request_row_joins.jsonl
m15_full_replay_carrier/SHA256SUMS
```

The periodic live snapshot also includes the growing replay envelope. A pod
loss therefore leaves a bounded chronology snapshot in GCS even before final
collection, but only the terminal `COMPLETE.json` admits a finished attempt.
The M15 carrier intentionally prints only SHA/size/`encoding=gcs-only` receipts
to the pod log; it does not base64-duplicate the large NPZ/tar into that log.

## Run the GCS-side audit

Run this on the machine that can read the bucket. It downloads and verifies
the immutable root, checks the nested producer/envelope/first-red joins, and
uploads a small derived audit beside the large payload:

```bash
cd /home/yuxuan/code_rl_repro/sequence_packing/tunix
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_replay_gcs_audit.sh \
  gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/<jobset>/attempt-0
```

Expected terminal marker:

```text
[M15.APC.GCS] COMPLETE status=<status> ... destination=.../derived/m15-replay-audit-v1
```

The derived prefix contains only small receipts under `files/`; its
`SHA256SUMS` is uploaded last. Raw logs, the complete producer NPZ, and the
serving envelope remain in the original GCS attempt and are not committed.

## What to return

Return only:

1. the exact source SHA, JobSet, attempt number, Kubernetes terminal state,
   and source GCS URI;
2. the one-line `[M15.APC.GCS] COMPLETE ...` output;
3. the derived GCS URI;
4. the small derived `RETURN_RECEIPT.json`, `SHA256SUMS`,
   `m15-classification.json`, and, on red, `replay-contract.json`;
5. any nonzero command return code and its complete stderr.

Do not manually summarize the large NPZ/JSONL and do not add them to Git.

## Claim ceiling

Even a successful red capture means only:

```text
FULL_REPLAY_CARRIER_FROZEN_REPLAY_NOT_RUN
```

The next phase must execute the carrier through serving. It may then say
`ONEHOST_NOT_REPRODUCED` or advance to first-red localization. It may not say
that APC, RoPE, pages, or topology is the root cause merely because the
carrier was captured.
