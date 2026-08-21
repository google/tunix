# 🚨 DeepSWE 128 Snapshot Resume Incident & Resolution Guide

**Incident Timestamp**: 2026-08-21T00:45:51Z  
**Context / Run ID**: `canon-p46-eval-census-128-p46c128a0`  
**Target Campaign Tag**: `p46q4census01` (128-TPU v5p, Topology `4x4x8`)  
**Associated Pod**: `canon-p46-eval-census-128-p46c128a0-pathways-head-0-0-rrqpt`  

---

## 1. 🔍 Incident Summary & Error Stack Trace

When starting the DeepSWE 128-chip census campaign (`p46c128a0`), all 32 TPU Worker Pods (128 chips) were successfully auto-provisioned by GKE NAP (`nap-ct5p-hightp-4t-1roqnuqt`) and transitioned to `Running (1/1)`.

However, the Head Pod exited with `rc=1` during the startup phase in `eval_deepswe.py`:

```text
2026-08-21 00:45:51,203 INFO P46 clean-data gate PASS dataset=4578 whitelist=1851 sha256=2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7
Traceback (most recent call last):
  File "/app/examples/deepswe/eval_deepswe.py", line 1029, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/app/examples/deepswe/eval_deepswe.py", line 900, in main
    receipt = import_legacy_v5_snapshot(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/examples/deepswe/deepswe_eval_artifacts.py", line 812, in import_legacy_v5_snapshot
    raise ValueError(
ValueError: legacy trajectory contract mismatch in /mnt/disks/linchai_data/deepswe_eval/p46q4census01/imports/p46e12806-v6-final/trajectories/q4i16k-n16-128-01b3047f8a076bc3.p0.20260816T001929Z.jsonl: {'config_fingerprint': '01b3047f8a076bc33209e9d340ad85a443928c26a346a1f4811612ca72b4e0af', 'run_tag': 'q4i16k-n16-128-01b3047f8a076bc3'}
[run] exit=1
[run] transport_rc=0
[P46.EVAL.POSTFLIGHT] rc=1 transport_rc=0 subshard=0 report=0 campaign=0 campaign_logical=0 census=0 census_logical=0 census_incomplete=0 timeout=0 log=/mnt/disks/linchai_data/deepswe_eval/p46q4census01/logs/campaign.attempt-20260821T004530Z.2QgL7l.log
[entrypoint] FATAL: 90_run.sh exited 1
```

---

## 2. 🧩 Root Cause Breakdown

### A. Fingerprint / Sampling Contract Mismatch
1. The snapshot under `/mnt/disks/linchai_data/deepswe_eval/p46q4census01/imports/p46e12806-v6-final` contains 510 trajectories recorded with fingerprint `01b3047f8a076bc3...` and `sampled_by: stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`.
2. When launching the JobSet, `--source-commit 5f2d016147a55c032ea7b89b156a583d3b4ca7e8` was passed without specifying `--sampling-source-commit ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`.
3. `legacy_v5_fingerprint(logical_config)` dynamically recalculated the expected hash using `5f2d0161...` instead of `ac2c31bc...`, yielding `86f28c74...` $\neq$ `01b3047f...`.
4. As designed by `import_legacy_v5_snapshot`'s strict defensive integrity checks, the process failed closed to prevent silent configuration pollution.

### B. Legacy Import vs. Frozen v6 Import API
As documented in [`canon-zero-tim/cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`](file:///usr/local/google/home/yuxzhang/yuxuan_dev/tunix_code_rl/canon-zero-tim/tunix/canon-zero-tim/cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md#L10-L56) (Section *P46.7 breadth-first census and v6 handoff*):
- Modern campaigns should use `--frozen-v6-import-id <snapshot_id>` instead of `--legacy-import-id <snapshot_id>`.
- The snapshot directory must include both `resume_contract.json` and all `trajectories/*.jsonl`, indexed in `SHA256SUMS`.
- `--sampling-source-commit` must be explicitly passed to preserve historical sampler lineage while `--source-commit` reflects the live harness commit.

---

## 3. 🛠️ Resolution Options for the Next Agent

### Option 1: Frozen v6 Import (Recommended per P46 Runbook)
1. Ensure the snapshot directory has both `resume_contract.json` and `trajectories/*.jsonl`:
   ```bash
   cd /mnt/disks/linchai_data/deepswe_eval/p46q4census01/imports/p46e12806-v6-final
   sha256sum resume_contract.json trajectories/*.jsonl > SHA256SUMS
   chmod -R a-w .
   ```
2. Render the JobSet using `--frozen-v6-import-id` and `--sampling-source-commit`:
   ```bash
   python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
     --base cluster/jobsets/qwen3-4b-dp-parity-deepswe-eval.yaml \
     --output /tmp/canon-p46-eval-census-128-p46c128a1.yaml \
     --workload eval \
     --topology 128 \
     --source-commit 5f2d016147a55c032ea7b89b156a583d3b4ca7e8 \
     --sampling-source-commit ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e \
     --frozen-v6-import-id p46e12806-v6-final \
     --client-image us-docker.pkg.dev/cloud-tpu-v2-images/base-images/vllm:jax-v0.10.2@sha256:d8ba39a67425f381f8f307bb5c7a0d4218eb84a3b7d15923985ec2840742f9b2 \
     --run-id p46c128a1 \
     --resume-tag p46q4census01 \
     --cpu-nodepool cpu-np \
     --worker-nodepool auto \
     --model-pvc haoyugao-cpu-np-pvc \
     --full-campaign \
     --first-pass-census
   ```
3. Apply the rendered JobSet:
   ```bash
   kubectl apply -f /tmp/canon-p46-eval-census-128-p46c128a1.yaml
   ```

---

### Option 2: Clean First-Pass Census (Fresh Start without Snapshot Dependency)
If starting a clean breadth-first census run across all 1,851 whitelist tasks:
```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base cluster/jobsets/qwen3-4b-dp-parity-deepswe-eval.yaml \
  --output /tmp/canon-p46-eval-census-128-p46c128clean.yaml \
  --workload eval \
  --topology 128 \
  --source-commit 5f2d016147a55c032ea7b89b156a583d3b4ca7e8 \
  --client-image us-docker.pkg.dev/cloud-tpu-v2-images/base-images/vllm:jax-v0.10.2@sha256:d8ba39a67425f381f8f307bb5c7a0d4218eb84a3b7d15923985ec2840742f9b2 \
  --run-id p46c128clean \
  --resume-tag p46q4census02 \
  --cpu-nodepool cpu-np \
  --worker-nodepool auto \
  --model-pvc haoyugao-cpu-np-pvc \
  --full-campaign \
  --first-pass-census
```

---

## 4. 📊 Existing Historical Data Status
All 22,918 trajectories (with 2,280 golden positive SFT trajectories and 4,374 DPO pairs) from previous campaigns remain fully intact and safely archived under:
[`clean_data/p46_128chip_deepswe_campaign/`](file:///usr/local/google/home/yuxzhang/yuxuan_dev/tunix_code_rl/canon-zero-tim/tunix/clean_data/p46_128chip_deepswe_campaign/REPORT.md)
