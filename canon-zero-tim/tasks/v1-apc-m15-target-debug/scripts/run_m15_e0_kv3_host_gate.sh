#!/usr/bin/env bash
# Local-only aggregate gate for the three-round M15 E0 targeted-KV carrier.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
canon="$(cd "$script_dir/../../.." && pwd)"
repo="$(cd "$canon/.." && pwd)"
cd "$repo"

python3 -m unittest discover \
  -s canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts \
  -p 'test_*.py'
bash canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_m15_attempt19_e0_kv3_return.sh
python3 canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_review_m15_attempt20_on_round0.py
bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
python3 -m unittest discover \
  -s canon-zero-tim/tests/p3_prefix_cache \
  -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
python3 canon-zero-tim/tests/manage_canon_flags/test_audit_flag_registry.py

python3 -m py_compile \
  canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_kv_observer.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/aggregate_m15_e0_kv_rounds.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/review_m15_attempt20_on_round0.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/stage_m15_e0_kv_round.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_m15_e0_kv_three_round.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_review_m15_attempt20_on_round0.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_resolved_env.py \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py
bash -n \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/install.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt19_e0_kv3_pair.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/prepare_m15_attempt20_e0_kv3_pair.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt19_e0_kv3_gcs_return.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt19_e0_kv3_return_recovery.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt20_e0_kv3_return_recovery.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/run_m15_attempt20_on_round0_offline_recovery.sh \
  canon-zero-tim/tasks/v1-apc-m15-target-debug/scripts/test_m15_attempt19_e0_kv3_return.sh \
  canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh \
  canon-zero-tim/tests/v1_phase4/run_exact_image.sh

grep -Fqx \
  'dae6dfa8a45bfd0a34b41baa9ec7c258229e8824c427a2fb863b620add074f98  tpu_runner_p21_l30.py' \
  canon-zero-tim/MANIFEST.sha256
git diff --check

echo "M15_E0U_HOST_PASS task_discovery=199 return=1 round0_recovery=6 v1_cpu=91 p3_prefix_cache=31 persistence=1 flags=409 manifest=dae6dfa8 syntax=1 diff_check=1 exact_image=0 target_rerun=0 gcs=0 kubernetes=0 tpu=0"
