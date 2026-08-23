#!/usr/bin/env python3
"""Shell-profile contracts for the P58 native and zero numerical arms."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
PROFILE = PKG / "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
CANON = PKG / "cluster/profiles/_canonical_engine.env"
HP_PROFILE = PKG / "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env"


def _source(arm: str) -> str:
  script = f"""
set -euo pipefail
source {CANON}
export CANON_P58_TIM_ARM={arm}
export CANON_P34_RUN_STAGE=three-update
export CANON_P58_EXPECTED_UPDATES=3
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P32_DP_REDUCTION_ADMITTED=1
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
source {PROFILE}
printf 'arm=%s dp=%s gen=%s local=%s global=%s warn=%s engine=%s vjp=%s prompt=%s stock_observer=%s p32=%s launch=%s\\n' \
  "$CANON_P58_TIM_ARM" "$CANON_DP_SIZE" "$CANON_NUM_GENERATIONS" \
  "$CANON_LOCAL_TRAJECTORIES" "$CANON_GLOBAL_TRAJECTORIES" \
  "$CANON_DEEPSWE_ALIGNMENT_WARN_ONLY" "$CANON_ENGINE_MODULE_C" \
  "$CANON_RPA_VJP2" "$CANON_PROMPT_PROCESSED_LOGPROBS" \
  "$CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER" \
  "$CANON_P32_TRAIN_ADMITTED" \
  "$CANON_P33_WORKLOAD_LAUNCH_ADMITTED"
printf 'reduction=%s l3=%s p27=%s flwarn=%s\\n' \
  "$CANON_P32_DP_REDUCTION_ADMITTED" "$CANON_FROZENLAKE_L3" \
  "$CANON_FROZENLAKE_P27" "$CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"
if [[ -v CANON_FIXED_AR ]]; then printf 'fixed=present\\n'; else printf 'fixed=absent\\n'; fi
if [[ -v CANON_LOGPROB_M ]]; then printf 'logm=present\\n'; else printf 'logm=absent\\n'; fi
printf 'xla=%s\\n' "$XLA_FLAGS"
"""
  return subprocess.run(
      ["bash", "-c", script],
      check=True,
      text=True,
      capture_output=True,
  ).stdout


class P58ProfileTest(unittest.TestCase):

  def test_native_removes_complete_numerical_bundle(self):
    output = _source("native")
    self.assertIn(
        "arm=native dp=8 gen=16 local=16 global=128 warn=1 "
        "engine=0 vjp=0 prompt=0 stock_observer=1 p32=1 launch=1",
        output,
    )
    self.assertIn("fixed=absent", output)
    self.assertIn("logm=absent", output)
    self.assertIn("xla=--xla_cpu_max_isa=AVX2", output)
    self.assertIn("reduction=0 l3=0 p27=0 flwarn=0", output)

  def test_zero_retains_complete_numerical_bundle(self):
    output = _source("zero")
    self.assertIn(
        "arm=zero dp=8 gen=16 local=16 global=128 warn=0 "
        "engine=1 vjp=1 prompt=1 stock_observer=0 p32=1 launch=1",
        output,
    )
    self.assertIn("fixed=present", output)
    self.assertIn("logm=present", output)
    self.assertIn("--xla_allow_excess_precision=false", output)
    self.assertIn("reduction=1 l3=0 p27=0 flwarn=0", output)

  def test_zero_hp_profile_resolves_exact_bundle(self):
    script = f"""
set -euo pipefail
export CANON_STATE=/tmp/p58-hp-test
export CANON_V1_HP_FULL=1
export CANON_P58_TIM_ARM=zero
export CANON_P58_DEEPSWE_TIM=1
export CANON_P58_TIM_ADMITTED=1
export CANON_P34_RUN_STAGE=full
export CANON_P34_NO_COMMIT=0
export CANON_P58_EXPECTED_UPDATES=1000
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P32_DP_REDUCTION_ADMITTED=1
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
source {CANON}
source {HP_PROFILE}
printf '%s\n' "$CANON_PROFILE|$CANON_CONTINUE_DECODE|$CANON_FIXED_AR_GATHER|$CANON_PALLAS_GATHERED_LOGPROBS|$CANON_LOGPROB_STEP_FUSION|$CANON_P59_RANK_PARALLEL_BACKWARD|$CANON_P38_FIXED_LM_HEAD|$CANON_VLLM_ENABLE_PREFIX_CACHING|$CANON_BATCHED_EVIDENCE"
"""
    output = subprocess.run(
        ["bash", "-c", script], check=True, text=True, capture_output=True
    ).stdout
    self.assertIn(
        "qwen3-4b-dp8-tp8-deepswe-v1-hp|8|1|1|1|1|1|0|0",
        output,
    )

  def test_zero_hp_profile_rejects_unsigned_or_native_entry(self):
    for v1, arm in (("0", "zero"), ("1", "native")):
      with self.subTest(v1=v1, arm=arm):
        script = f"""
set -euo pipefail
export CANON_STATE=/tmp/p58-hp-negative
export CANON_V1_HP_FULL={v1}
export CANON_P58_TIM_ARM={arm}
export CANON_P58_DEEPSWE_TIM=1
export CANON_P58_TIM_ADMITTED=1
export CANON_P34_RUN_STAGE=full
export CANON_P34_NO_COMMIT=0
export CANON_P58_EXPECTED_UPDATES=1000
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P32_DP_REDUCTION_ADMITTED=1
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
source {CANON}
source {HP_PROFILE}
"""
        result = subprocess.run(
            ["bash", "-c", script], check=False, text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
  unittest.main()
