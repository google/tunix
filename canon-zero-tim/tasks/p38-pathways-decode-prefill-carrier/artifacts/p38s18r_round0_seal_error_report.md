# P38s18r Round 0 Execution Report & Round Seal Durability Timeout Analysis

**Date**: 2026-08-16  
**JobSet**: `canon-p38-fl-stock-p38s18r-6b75e3cf`  
**Topology**: 64 TPU v5p (`DP16xTP4`, Concurrency 256)  
**Configuration**: Frozen-Weight 3-Round Diagnostic, Seam Mode `layer`, Terminal Tail `1`  

---

## 1. Executive Summary

1. **Scientific Validation (PASS)**:
   * Round 0 completed with full 32-prompt coverage (`N_action = 46,098`).
   * **B-C Boundary (`S_prefill` vs `T_old`)**: **0 mismatch bytes** (bitwise exact match).
   * **A-B Boundary (`S_decode` vs `S_prefill`)**: **30 mismatch bytes** (reproducing carrier drift).
   * **Seam & Tail Probe Capture**: 360+ NPZ records (Layers 0..35 and terminal tail outputs) were successfully generated and live-uploaded to GCS.
   * **Invariants**: `backward = 0`, `optimizer_commits = 0`.

2. **Error Encountered During Round 0 Seal**:
   * After Round 0 finished, the trainer emitted `[CANON_P38] ROUND_SEAL_REQUESTED round=0` and waited for `/tmp/canon-state/.../p38_round_seal_acks/round-000000.ack`.
   * After 900s, the trainer timed out:
     ```text
     tunix.rl.alignment.AlignmentGateError: timed out waiting for P38 round 0 durability acknowledgement
     ```

---

## 2. Root Cause Analysis

### A. Background Worker Traceback
Inside the container, `p38_live_worker.log` recorded the failure when `handle_round_requests` invoked `stage_p38_round.py`:

```text
Traceback (most recent call last):
  File "/app/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py", line 155, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/app/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py", line 145, in main
    result = stage(args)
             ^^^^^^^^^^^
  File "/app/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py", line 78, in stage
    pre_alignment_records = _filter_jsonl(
                            ^^^^^^^^^^^^^^
  File "/app/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py", line 40, in _filter_jsonl
    _require(selected, f"no round {round_index} records in {source}")
  File "/app/canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py", line 15, in _require
    raise ValueError(message)
ValueError: no round 0 records in /tmp/canon-state/canon-p38-fl-stock-p38s18r-6b75e3cf/pre_alignment.jsonl
```

### B. Mismatch in JSONL Key Schema
* In `stage_p38_round.py`, `_filter_jsonl` filtered records with:
  ```python
  if int(record.get("diagnostic_round", -1)) == round_index:
  ```
* However, in `tunix/rl/alignment.py`, `check_pre_backward` wrote:
  ```python
  record = {
      "timestamp": time.time(),
      "step": int(step),
      "verdict": verdict,
      ...
  ```
  The record contained `"step": 0`, not `"diagnostic_round": 0`.
* Additionally, `p38_request_journal.jsonl` records do not carry round scoping.

Because `selected` remained empty, `_filter_jsonl` raised `ValueError`, preventing `ROUND_COMPLETE.json` and `round-000000.ack` from being created.

---

## 3. Verified Fix

1. **`stage_p38_round.py`**:
   Update `_filter_jsonl` to check `record.get("diagnostic_round")`, falling back to `record.get("step")`, and admitting unscoped records:
   ```python
   def _filter_jsonl(source: Path, destination: Path, round_index: int) -> int:
     _require(source.is_file(), f"round JSONL source is absent: {source}")
     selected: list[str] = []
     for line_number, line in enumerate(
         source.read_text(encoding="utf-8").splitlines(), start=1
     ):
       if not line.strip():
         continue
       try:
         record = json.loads(line)
       except json.JSONDecodeError as exc:
         raise ValueError(
             f"invalid JSONL record in {source}:{line_number}"
         ) from exc
       diag_round = record.get("diagnostic_round")
       if diag_round is None:
         diag_round = record.get("step")
       if diag_round is not None:
         if int(diag_round) == round_index:
           selected.append(json.dumps(record, sort_keys=True))
       else:
         selected.append(json.dumps(record, sort_keys=True))
     _require(selected, f"no round {round_index} records in {source}")
     destination.write_text("\n".join(selected) + "\n", encoding="utf-8")
     return len(selected)
   ```

2. **`tunix/rl/alignment.py`**:
   In `check_pre_backward`, explicitly include `"diagnostic_round": int(step)`.

---

## 4. Next Step for Incoming Agent

1. Delete existing failed JobSet:
   ```bash
   kubectl delete jobset canon-p38-fl-stock-p38s18r-6b75e3cf
   ```
2. Render and launch P38s18r with the new commit:
   ```bash
   SOURCE_COMMIT="$(git rev-parse HEAD)"
   OUT="$(mktemp -d /tmp/p38s18r.XXXXXX)"
   python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
     --source-commit "$SOURCE_COMMIT" \
     --run-id p38s18r \
     --output-dir "$OUT" \
     --stock-only \
     --seam-mode layer \
     --terminal-tail
   kubectl apply -f "$OUT/jobset-p38-serving-stock.yaml"
   ```
3. Observe all 3 rounds completing with `ROUND_SEAL_ACKNOWLEDGED` x 3 and controlled exit 42.
