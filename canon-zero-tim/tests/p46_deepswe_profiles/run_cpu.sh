#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

bash -n \
  canon-zero-tim/cluster/entrypoint.sh \
  canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/90_run.sh
python3 -m py_compile \
  examples/deepswe/eval_deepswe.py \
  examples/deepswe/deepswe_eval_artifacts.py \
  examples/deepswe/deepswe_reward_only_parity.py \
  examples/deepswe/probe_reward_only_v5p.py \
  tunix/generate/vllm_sampler.py \
  canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  canon-zero-tim/tests/p34_deepswe/classify_run.py
python3 -m unittest discover \
  -s canon-zero-tim/tests/p46_deepswe_profiles \
  -p 'test_*.py' \
  -v

echo "P46_DEEPSWE_PROFILES_CPU_PASS cases=33"
