# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the knobs of the MLPerf submission launcher.

Each test renders mlperf_gbs1024_submission.sh with DRY_RUN=true and checks the
rendered JobSets, so the defaults, the DROPLESS switch and the learning-rate
rule are tested through the same path a launch takes.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from absl.testing import absltest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = (
    _REPO
    / "tunix"
    / "experimental"
    / "examples"
    / "recipes"
    / "mlperf_gbs1024_submission.sh"
)
_PREFIX = "launcher-test"
_HBM_CAP = "--xla_tpu_max_hbm_size_mib=84992"


def _render(**overrides: str) -> str:
  """Returns the DRY_RUN render of the submission launcher.

  The environment is scrubbed (an inherited DROPLESS or LEARNING_RATE would
  change the result), HOME is a temporary directory and KUBECONFIG an empty
  file, so the recipe's `kubectl config use-context` cannot touch a real
  kubeconfig and the render needs no cluster.
  """
  with tempfile.TemporaryDirectory() as home:
    kubeconfig = os.path.join(home, "kubeconfig")
    open(kubeconfig, "w").close()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "USER": "launcher-test-user",
        "KUBECONFIG": kubeconfig,
        "DRY_RUN": "true",
        "JOB_PREFIX": _PREFIX,
        "TUNIX_IMAGE": "example.com/tunix:test",
        "K8S_NAMESPACE": "test-ns",
        "UBENCH_REPORTING": "false",
    }
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(_RECIPE), "start"],
        env=env,
        cwd=home,
        capture_output=True,
        text=True,
        check=False,
    )
  if result.returncode != 0:
    raise AssertionError(
        f"launcher exited {result.returncode}:\n{result.stderr[-2000:]}"
    )
  return result.stdout


def _jobset(render: str, suffix: str) -> str:
  """Returns the rendered document of the JobSet `<prefix>-<suffix>`."""
  for doc in render.split("\n---\n"):
    if re.search(rf"^  name: {_PREFIX}-{suffix}$", doc, re.M):
      return doc
  raise AssertionError(f"no JobSet {_PREFIX}-{suffix} in the render")


def _maxtext_flags(trainer: str) -> dict[str, str]:
  flags = re.search(r'MAXTEXT_EXTRA_FLAGS="([^"]*)"', trainer).group(1)
  return dict(token.split("=", 1) for token in flags.split())


def _vllm_config(rollout: str) -> dict:
  return json.loads(re.search(r"--vllm_config_json='([^']*)'", rollout).group(1))


@absltest.skipIf(shutil.which("python3") is None, "the recipes need python3")
class SubmissionLauncherTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.default = _render()

  def test_dropless_is_the_default(self):
    trainer = _jobset(self.default, "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "-1")
    self.assertEqual(flags["context"], "offload")
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    # The HBM budget reaches the trainer env, the Pathways workers and the
    # Pathways proxy (which compiles the trainer program).
    self.assertEqual(trainer.count(_HBM_CAP), 3)
    self.assertIn("--pipeline_train_microbatches", _jobset(self.default, "orch"))

  def test_dropless_false_is_the_capacity_factor_trainer(self):
    trainer = _jobset(_render(DROPLESS="false"), "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "2.0")
    self.assertEqual(flags["context"], "remat")
    self.assertEqual(flags["num_moe_token_chunks"], "2")
    self.assertNotIn("xla_tpu_max_hbm_size_mib", trainer)

  def test_dropless_values_can_be_overridden_one_at_a_time(self):
    trainer = _jobset(_render(NUM_MOE_TOKEN_CHUNKS="8"), "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["num_moe_token_chunks"], "8")
    self.assertEqual(flags["ragged_buffer_factor"], "-1")

  def test_finite_buffer_factor_needs_the_retry_to_stay_dropless(self):
    with self.assertRaisesRegex(AssertionError, "RETRY_WHEN_TOKENS_DROPPED"):
      _render(RAGGED_BUFFER_FACTOR="2.75")

  def test_finite_buffer_factor_with_retry_keeps_the_dropless_layout(self):
    trainer = _jobset(
        _render(RAGGED_BUFFER_FACTOR="2.75", RETRY_WHEN_TOKENS_DROPPED="true"),
        "train",
    )
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")
    self.assertEqual(flags["retry_when_tokens_dropped"], "true")
    self.assertEqual(flags["retry_num_moe_token_chunks"], "8")
    self.assertEqual(flags["context"], "offload")
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    self.assertEqual(trainer.count(_HBM_CAP), 3)

  def test_moe_audit_and_padding_knobs_are_off_by_default(self):
    flags = _maxtext_flags(_jobset(self.default, "train"))
    for key in (
        "moe_exclude_padding_from_routing",
        "log_moe_drops",
        "retry_dropless_final_micro_batch",
    ):
      self.assertNotIn(key, flags)

  def test_moe_audit_and_padding_knobs_append_their_flags(self):
    trainer = _jobset(
        _render(
            RAGGED_BUFFER_FACTOR="2.75",
            RETRY_WHEN_TOKENS_DROPPED="true",
            RETRY_DROPLESS_FINAL_MICRO_BATCH="true",
            MOE_EXCLUDE_PADDING="true",
            LOG_MOE_DROPS="true",
        ),
        "train",
    )
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["moe_exclude_padding_from_routing"], "true")
    self.assertEqual(flags["log_moe_drops"], "true")
    self.assertEqual(flags["retry_dropless_final_micro_batch"], "true")
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")

  def test_final_micro_batch_knob_needs_the_retry(self):
    flags = _maxtext_flags(
        _jobset(_render(RETRY_DROPLESS_FINAL_MICRO_BATCH="true"), "train")
    )
    self.assertNotIn("retry_dropless_final_micro_batch", flags)

  def test_dropless_rejects_other_values(self):
    with self.assertRaisesRegex(AssertionError, "DROPLESS must be true or false"):
      _render(DROPLESS="yes")

  def test_default_batch_gives_the_gbs_1024_hyperparameters(self):
    orch = _jobset(self.default, "orch")
    self.assertIn("--batch_size=64 ", orch)
    self.assertIn("--learning_rate=2e-06 ", orch)
    self.assertIn("--max_grad_norm=0.0625 ", orch)

  def test_batch_size_72_derives_learning_rate_and_clip(self):
    render = _render(BATCH_SIZE="72")
    orch = _jobset(render, "orch")
    self.assertIn("--learning_rate=2.121320343559642e-06 ", orch)
    self.assertIn("--max_grad_norm=0.05892556509887896 ", orch)
    trainer = _jobset(render, "train")
    self.assertIn("--optimizer_learning_rate=2.121320343559642e-06 ", trainer)
    self.assertIn("{'max_norm': 0.05892556509887896}", trainer)

  def test_explicit_learning_rate_and_clip_win(self):
    orch = _jobset(
        _render(BATCH_SIZE="72", LEARNING_RATE="3e-06", MAX_GRAD_NORM="0.1"),
        "orch",
    )
    self.assertIn("--learning_rate=3e-06 ", orch)
    self.assertIn("--max_grad_norm=0.1 ", orch)

  def test_start_up_flags_are_emitted(self):
    orch = _jobset(self.default, "orch")
    self.assertIn("--initial_dispatch_stagger_s=75", orch)
    self.assertIn("--initial_dispatch_first_stagger_s=175", orch)
    self.assertIn("--trainer_warmup_compile", orch)
    self.assertIn("--trainer_warmup_routed_experts", orch)
    self.assertIn("--no-sandbox_wait_initial", orch)

  def test_start_up_flags_can_be_turned_off(self):
    orch = _jobset(
        _render(
            INITIAL_DISPATCH_STAGGER_S="0",
            INITIAL_DISPATCH_FIRST_STAGGER_S="0",
            TRAINER_WARMUP_COMPILE="false",
        ),
        "orch",
    )
    self.assertNotIn("--initial_dispatch_stagger_s", orch)
    self.assertNotIn("--initial_dispatch_first_stagger_s", orch)
    self.assertNotIn("--trainer_warmup_compile", orch)

  def test_in_flight_cap_is_off_by_default(self):
    self.assertNotIn("--max_in_flight_batches", _jobset(self.default, "orch"))

  def test_in_flight_cap_reaches_the_orchestrator(self):
    orch = _jobset(_render(MAX_IN_FLIGHT_BATCHES="1.5"), "orch")
    self.assertIn("--max_in_flight_batches=1.5", orch)

  def test_rollout_attention_without_data_parallelism(self):
    cfg = _vllm_config(_jobset(self.default, "roll-0"))
    strategy = cfg["additional_config"]["sharding"]["sharding_strategy"]
    self.assertFalse(strategy["enable_dp_attention"])
    self.assertEqual(strategy["tensor_parallelism"], 1)
    self.assertEqual(strategy["expert_parallelism"], 8)
    self.assertEqual(cfg["max_num_seqs"], 64)
    # The trainer replicates K/V heads to the rollout's EP x TP.
    self.assertEqual(_maxtext_flags(_jobset(self.default, "train"))["kv_tp_size"], "8")

  def test_rollout_attention_data_parallelism_can_be_restored(self):
    render = _render(ROLLOUT_ATTENTION_DP="true")
    cfg = _vllm_config(_jobset(render, "roll-0"))
    strategy = cfg["additional_config"]["sharding"]["sharding_strategy"]
    self.assertTrue(strategy["enable_dp_attention"])
    self.assertEqual(cfg["max_num_seqs"], 16)
    self.assertNotIn("kv_tp_size", _maxtext_flags(_jobset(render, "train")))


if __name__ == "__main__":
  absltest.main()
