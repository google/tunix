# Tunix Utility & CI Monitoring Scripts

This directory contains setup utilities, verification scripts, and automated TPU
CI queue & runtime monitoring tools for Tunix.

## Environment & Dependency Setup

- `install_tunix_vllm_requirement.sh`: Installs pinned `requirements.txt` and
  `special_requirements.txt` dependencies for local TPU VM or developer
  environments integrating `vllm` and `tpu-inference`.
- `setup_cli_tpu_single_host.sh` / `setup_cli_gpu_single_host.sh`: Sets up a
  single-host TPU or GPU environment for running the Tunix CLI.
- `setup_notebook_tpu_single_host.sh`: Sets up a single-host TPU environment for
  interactive notebooks.

## TPU CI Queue & Runtime Monitoring (`tpu_ci_monitor.py` & `tpu_ci_alert.py`)

The scheduled GitHub Actions workflow `.github/workflows/tpu-ci-utilization.yml`
runs every 6 hours (`0 */6 * * *`) to monitor self-hosted TPU runner queue wait
times and job durations across `Tunix Package Tests` and
`Tunix Nightly Regression Tests`. Both scripts use only the Python standard
library and the `gh` CLI (pre-installed on GitHub-hosted runners).

### 1. `tpu_ci_monitor.py` — Telemetry Collector & SLA Evaluator

Queries recent workflow runs via `gh api` (with a `urllib.request` fallback),
extracts timing metrics for TPU jobs (`run_core`, `run_dev`, `run_prod`,
`run_latest` on `linux-x86-ct6e-180-8tpu`), and evaluates two queue wait SLAs:

| Metric | Window | SLA Threshold | Behavior |
| :--- | :--- | :--- | :--- |
| **Queue Wait `p90`** (`started_at - created_at`) | Rolling `24h` | `<= 600s` (10 min) | Detects sustained runner pool queuing across the day. |
| **Single-Job Queue Wait `Max`** | Latest `6h` or active (`queued` / `in_progress`) | `<= 1200s` (20 min) | Detects acute stalls while allowing transient single-job spikes to auto-recover on the next 6h run. |

**Local Usage:**

```bash
python3 scripts/tpu_ci_monitor.py \
  --repo google/tunix \
  --window-hours 24 \
  --markdown-output tpu_ci_report.md \
  --json-output tpu_ci_alert.json
```

### 2. `tpu_ci_alert.py` — P1 GitHub Issue Alert Synchronizer

Synchronizes at most one open P1 GitHub issue per alert category using `gh`:

- **Queue SLA Breach (`labels: ["tpu-ci-alert", "bug", "p1"]`)**: Opened or
  updated when `tpu_ci_alert.json` reports `"breached": true`, and auto-closed
  with a recovery summary when `"breached": false`.
- **Monitor Execution Error (`labels: ["tpu-ci-monitor-error", "bug", "p1"]`)**:
  Opened or updated if `tpu_ci_monitor.py` fails
  (`--monitor-outcome != "success"`) or produces no valid JSON payload, and
  auto-closed on the next successful monitor run without affecting any open
  `tpu-ci-alert` issue.
- **Notification Hygiene**: New P1 issues are assigned to `@dingying-ai` and CC
  `@fineguy`, `@shz-google`, `@s-noghabi`, and `@tianshub` on initial creation;
  follow-up and recovery comments omit `@`-mentions to avoid repeated pings.

**Local Usage:**

```bash
python3 scripts/tpu_ci_alert.py \
  --repo google/tunix \
  --run-id 12345 \
  --run-url "https://github.com/google/tunix/actions/runs/12345" \
  --monitor-outcome success \
  --alert-json tpu_ci_alert.json
```
