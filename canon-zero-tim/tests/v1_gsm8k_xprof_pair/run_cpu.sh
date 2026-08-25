#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python3 -m unittest discover \
  -s "$repo/canon-zero-tim/tests/v1_gsm8k_xprof_pair" \
  -p 'test_*.py'
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_common.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_inner.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_native.sh"
bash -n "$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh"
bash -n "$repo/canon-zero-tim/tests/v1_gsm8k_xprof_pair/run_exact_image.sh"
docs=(
  EXECUTOR_PROMPT_P60_2.md
  HANDOFF_P60_2.md
  goal.md
  state.md
  plan.md
  log.md
  HANDOFF.md
  RUNBOOK.md
  phases/p60-2a-readability-baseline.md
  phases/p60-2b-hierarchy-instrumentation.md
  phases/p60-2c-onehost-visual-certification.md
  phases/p60-2d-attribution-and-next-decision.md
  phases/p60-2e-microstep-readability.md
)
task="$repo/canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair"
for doc in "${docs[@]}"; do
  test -s "$task/$doc"
done
grep -Fq 'StepTraceAnnotation("train"' "$task/phases/p60-2b-hierarchy-instrumentation.md"
grep -Eq 'non-empty.*`Steps`.*8/8' "$task/HANDOFF_P60_2.md"
grep -Fq 'P60-2B' "$task/state.md"
grep -Fq 'P60-2B' "$task/plan.md"
grep -Fq 'micro_step=0..15' "$task/phases/p60-2e-microstep-readability.md"
grep -Fq 'same `/host:CPU` `python3` track' "$task/HANDOFF_P60_2.md"
grep -Fq "matches Native's annotation API only" "$task/phases/p60-2e-microstep-readability.md"
grep -Fq 'jit__precomputed_gradient_scaled_step' "$task/phases/p60-2e-microstep-readability.md"
grep -Fq 'jit__precomputed_gradient_commit' "$task/phases/p60-2e-microstep-readability.md"
echo "P60_2_DOCSET_PASS files=${#docs[@]} phase=p60-2e"
echo "V1_GSM8K_XPROF_CPU_PASS"
