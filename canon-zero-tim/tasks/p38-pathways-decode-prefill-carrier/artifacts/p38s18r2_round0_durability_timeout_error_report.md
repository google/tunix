# P38s18r2 Round 0 Durability Seal Timeout Error Report

## 1. Executive Summary & Root Cause

- **Workload**: `canon-p38-fl-stock-p38s18r2-10fe951f` (64 TPU `DP16xTP4`, Concurrency 256, 3 Diagnostic Rounds)
- **Failure Point**: Main Python thread timed out at line 194 of `tunix/rl/alignment.py` (`_seal_p38_diagnostic_round`) after 900 seconds waiting for `round-000000.ack`.
- **Root Cause**:
  1. Round 0 with concurrency 256 generated a massive volume of fine-grained files:
     - **971 Tail JSON records + 971 Tail NPZ arrays**
     - **915 Seam JSON records + 915 Seam NPZ arrays**
     - **Incident Ledger (21.9 MB), Request Journal (1.97 MB), Pre-alignment (26 KB), Mismatch Capsule (33.7 KB)**
     - **Total: 3,776 individual files to upload to GCS per round.**
  2. `persist_p38_gcs.sh` / `stage_p38_round.py` uploads these 3,776 files sequentially to GCS (`gs://yuxzhang-tunix-models/...`).
  3. Sequential GCS upload of 3,776 files took **~57 minutes** (started at `05:38:09 UTC`, completed at `06:35:xx UTC`).
  4. The Python learner in `tunix/rl/alignment.py:166` has a hardcoded timeout `deadline = time.monotonic() + 900.0` (15 minutes).
  5. At `05:53:09 UTC` (15 minutes), Python timed out and raised:
     `tunix.rl.alignment.AlignmentGateError: timed out waiting for P38 round 0 durability acknowledgement`
  6. The background worker `p38_live_snapshot_worker.sh` continued running in the background and successfully finished uploading all 3,776 files, generated `rounds/000000/ROUND_COMPLETE.json`, verified SHA256 checksums, and wrote `round-000000.ack` at `06:35:xx UTC`.
  7. But because Python had already exited, Round 1 and Round 2 never started.

---

## 2. Full Error Traceback

```text
[CANON_ALIGN_PRE_EVIDENCE] path=/tmp/canon-state/canon-p38-fl-stock-p38s18r2-10fe951f/pre_alignment.jsonl sha256=379beed1e45b38e2adf44235abe78d0a8083634fdb70e5bcb5a4597bb2993d43
[CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=45559 bounds=[('S_decode_vs_S_prefill', 45), ('S_prefill_vs_T_old', 0)]
[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/3 step=0 N_action=45559 verdict=FAIL a_b_differing_bytes=45 backward=0 optimizer_commits=0
[CANON_P38] ROUND_SEAL_REQUESTED round=0 request=/tmp/canon-state/canon-p38-fl-stock-p38s18r2-10fe951f/p38_round_seal_requests/round-000000.request
INFO 08-17 05:38:09 [loggers.py:273] Engine 000: Avg prompt throughput: 2092.3 tokens/s, Avg generation throughput: 1.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
INFO 08-17 05:38:19 [loggers.py:273] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/examples/frozenlake/train_frozenlake_qwen3.py", line 1302, in <module>
[rank0]:     grpo_trainer.train(
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2202, in train
[rank0]:     train_examples = self._batch_to_train_example(
[rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1871, in _batch_to_train_example
[rank0]:     return self._process_results(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_grpo_learner.py", line 1447, in _process_results
[rank0]:     alignment.stop_after_diagnostic_precheck(precheck_record)
[rank0]:   File "/app/tunix/rl/alignment.py", line 334, in stop_after_diagnostic_precheck
[rank0]:     _seal_p38_diagnostic_round(round_index)
[rank0]:   File "/app/tunix/rl/alignment.py", line 194, in _seal_p38_diagnostic_round
[rank0]:     raise AlignmentGateError(
[rank0]: tunix.rl.alignment.AlignmentGateError: timed out waiting for P38 round 0 durability acknowledgement
```

---

## 3. Worker Sealing Confirmation Log

```text
[P38.GCS] ROUND_UPLOADED round=0 name=pre-alignment.jsonl bytes=26074
[P38.GCS] ROUND_UPLOADED round=0 name=request-journal.jsonl bytes=1971975
[P38.GCS] ROUND_UPLOADED round=0 name=run.log bytes=970127
[P38.GCS] ROUND_UPLOADED round=0 name=SHA256SUMS bytes=336822
[P38.GCS] ROUND_COMPLETE round=0 prefix=gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000 manifest_sha256=ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e
[P38.GCS] LIVE_ROUND_PASS round=0 ack=/tmp/canon-state/canon-p38-fl-stock-p38s18r2-10fe951f/p38_round_seal_acks/round-000000.ack
```

---

## 4. Suggested Fixes for the Other Agent

To allow all 3 rounds to complete and reach controlled exit code 42 without seal timeout:

1. **Fix A (Tar Archive Batch Upload - Recommended)**:
   In `persist_p38_gcs.sh` / `stage_p38_round.py`:
   Instead of uploading 3,776 small files individually (`gsutil cp file1; gsutil cp file2; ...`), bundle them into a single `round-00000X.tar.gz` and upload the archive in 3 seconds, followed by `ROUND_COMPLETE.json` and `SHA256SUMS`.
2. **Fix B (Parallel GCS Upload)**:
   Use `gcloud storage cp -m -r` or `gsutil -m cp -r` to upload in parallel threads rather than sequential single-file operations.
3. **Fix C (Increase Python Timeout)**:
   In `tunix/rl/alignment.py:166`:
   Increase `deadline = time.monotonic() + 900.0` or allow configuration via environment variable (e.g. `P38_SEAL_TIMEOUT_SECONDS=3600`).
