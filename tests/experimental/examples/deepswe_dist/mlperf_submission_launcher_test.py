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
rule are tested through the same path a launch takes. Two tests render the other
397B v7x recipes, which leave TRAINER_TUNED_FLAGS, the sharded log-prob head and
the staged step boundary off.
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
_SC_FLAG = "--xla_tpu_enable_offloading_copy_to_sparsecore=false"
# What TRAINER_TUNED_FLAGS changes in the trainer's MaxText flags at the bench
# cell: key -> (value with it off, value with it on). A repeated key takes its
# last value, as MaxText's command line parsing does, and an absent key its
# MaxText default (_MAXTEXT_DEFAULTS).
_MAXTEXT_DEFAULTS = {
    "cast_grads_after_all_reduce": "false",
    "gdn_cp_matmul_precision": "highest",
    "gdn_cp_state": "remat",
    "moe_route": "remat",
}
_TUNED_KEYS = {
    "cast_grads_after_all_reduce": ("false", "true"),
    "context_parallel_strategy": ("ring", "all_gather"),
    "gdn_cp_matmul_precision": ("highest", "high"),
    "gdn_cp_mode": ("head", "seq"),
    "gdn_cp_state": ("remat", "device"),
    "moe_route": ("remat", "device"),
    "num_moe_token_chunks": ("4", "2"),
    "sa_fuse_reciprocal": ("false", "true"),
    "sa_use_base2_exp": ("true", "false"),
}
# The launch whose trainer the RL trainer bench (AI-Hypercomputer/maxtext#5701)
# measured TRAINER_TUNED_FLAGS on: the replay layout (buffer factor 2.75 with
# the dropless replay), with context on device.
_BENCH_CELL = {
    "BATCH_SIZE": "72",
    "RAGGED_BUFFER_FACTOR": "2.75",
    "RETRY_WHEN_TOKENS_DROPPED": "true",
    "REMAT_CONTEXT": "device",
    "MOE_EXCLUDE_PADDING": "true",
    "LOG_MOE_DROPS": "true",
    "COMPUTE_LOGPS_CHUNK_SIZE": "1024",
}


def _render(recipe: Path = _RECIPE, **overrides: str) -> str:
  """Returns the DRY_RUN render of a launcher (default: the submission one).

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
        ["bash", str(recipe), "start"],
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


def _value(flags: dict[str, str], key: str) -> str | None:
  """Returns the value MaxText resolves for `key` from these flags."""
  return flags.get(key, _MAXTEXT_DEFAULTS.get(key))


def _flag_changes(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, tuple[str | None, str | None]]:
  """Returns key -> (before, after) for each MaxText flag that differs."""
  return {
      key: (_value(before, key), _value(after, key))
      for key in before.keys() | after.keys()
      if _value(before, key) != _value(after, key)
  }


def _without_tuned_flags(trainer: str) -> str:
  """Returns the trainer JobSet without the libtpu flag and MaxText flags."""
  trainer = re.sub(rf'\n *- "{re.escape(_SC_FLAG)}"', "", trainer)  # proxy arg
  trainer = trainer.replace(f" {_SC_FLAG}", "")  # trainer and worker env
  return re.sub(r'MAXTEXT_EXTRA_FLAGS="[^"]*"', "MAXTEXT_EXTRA_FLAGS=", trainer)


@absltest.skipIf(shutil.which("python3") is None, "the recipes need python3")
class SubmissionLauncherTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.default = _render()
    cls.default_off = _render(TRAINER_TUNED_FLAGS="false")
    cls.bench_on = _render(**_BENCH_CELL)
    cls.bench_off = _render(TRAINER_TUNED_FLAGS="false", **_BENCH_CELL)

  def test_the_default_trainer_is_the_measured_bench_cell(self):
    trainer = _jobset(self.default, "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")
    self.assertEqual(flags["retry_when_tokens_dropped"], "true")
    self.assertEqual(flags["retry_num_moe_token_chunks"], "8")
    self.assertEqual(flags["context"], "device")
    self.assertEqual(flags["num_moe_token_chunks"], "2")
    self.assertEqual(flags["moe_exclude_padding_from_routing"], "true")
    self.assertEqual(flags["log_moe_drops"], "true")
    self.assertIn("--compute_logps_chunk_size=1024 ", trainer)
    # The HBM budget reaches the trainer env, the Pathways workers and the
    # Pathways proxy (which compiles the trainer program).
    self.assertEqual(trainer.count(_HBM_CAP), 3)
    self.assertIn("--pipeline_train_microbatches", _jobset(self.default, "orch"))
    # The bench cell differs only in its batch size (GBS 1,152), which sets no
    # trainer MaxText or libtpu flag.
    bench = _jobset(self.bench_on, "train")
    self.assertEqual(flags, _maxtext_flags(bench))
    libtpu = r"LIBTPU_INIT_ARGS='([^']*)'"
    self.assertEqual(
        re.search(libtpu, trainer).group(1), re.search(libtpu, bench).group(1)
    )

  def test_buffer_factor_minus_one_keeps_the_fully_dropless_layout(self):
    trainer = _jobset(_render(RAGGED_BUFFER_FACTOR="-1"), "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "-1")
    self.assertEqual(flags["context"], "offload")
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    for key in (
        "retry_when_tokens_dropped",
        "moe_exclude_padding_from_routing",
        "log_moe_drops",
    ):
      self.assertNotIn(key, flags)
    self.assertEqual(trainer.count(_HBM_CAP), 3)

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
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")
    self.assertEqual(flags["context"], "device")

  def test_finite_buffer_factor_needs_the_retry_to_stay_dropless(self):
    for overrides in (
        {"RETRY_WHEN_TOKENS_DROPPED": "false"},
        {"TRAINER_TUNED_FLAGS": "false"},
    ):
      with self.assertRaisesRegex(AssertionError, "RETRY_WHEN_TOKENS_DROPPED"):
        _render(RAGGED_BUFFER_FACTOR="2.75", **overrides)

  def test_finite_buffer_factor_with_retry_keeps_the_dropless_layout(self):
    # With TRAINER_TUNED_FLAGS on, this layout runs two chunks instead (see the
    # tests of that switch below).
    trainer = _jobset(
        _render(
            RAGGED_BUFFER_FACTOR="2.75",
            RETRY_WHEN_TOKENS_DROPPED="true",
            TRAINER_TUNED_FLAGS="false",
        ),
        "train",
    )
    flags = _maxtext_flags(trainer)
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")
    self.assertEqual(flags["retry_when_tokens_dropped"], "true")
    self.assertEqual(flags["retry_num_moe_token_chunks"], "8")
    self.assertEqual(flags["context"], "offload")
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    self.assertEqual(trainer.count(_HBM_CAP), 3)

  def test_moe_audit_and_padding_knobs_follow_the_replay_layout(self):
    flags = _maxtext_flags(_jobset(self.default, "train"))
    self.assertEqual(flags["moe_exclude_padding_from_routing"], "true")
    self.assertEqual(flags["log_moe_drops"], "true")
    self.assertNotIn("retry_dropless_final_micro_batch", flags)
    for render in (_render(RAGGED_BUFFER_FACTOR="-1"), self.default_off):
      flags = _maxtext_flags(_jobset(render, "train"))
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
    # The fully dropless final program keeps the layout's four chunks and
    # context offload; two chunks were measured without it.
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    self.assertEqual(flags["context"], "offload")

  def test_first_phase_buffer_factor_keeps_four_chunks(self):
    flags = _maxtext_flags(
        _jobset(_render(FIRST_PHASE_RAGGED_BUFFER_FACTOR="4.0"), "train")
    )
    self.assertEqual(flags["first_phase_ragged_buffer_factor"], "4.0")
    self.assertEqual(flags["ragged_buffer_factor"], "2.75")
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    self.assertEqual(flags["context"], "offload")

  def test_final_micro_batch_knob_needs_the_retry(self):
    flags = _maxtext_flags(
        _jobset(
            _render(
                RAGGED_BUFFER_FACTOR="-1", RETRY_DROPLESS_FINAL_MICRO_BATCH="true"
            ),
            "train",
        )
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

  def test_the_staged_step_boundary_is_on_by_default(self):
    orch = _jobset(self.default, "orch")
    self.assertIn("--pipeline_step_boundary", orch)
    self.assertIn("--stage_next_microbatch", orch)
    self.assertNotIn("--no-pipeline_step_boundary", orch)
    self.assertNotIn("--no-stage_next_microbatch", orch)

  def test_the_pipelined_step_boundary_off_turns_staging_off(self):
    orch = _jobset(_render(PIPELINE_STEP_BOUNDARY="false"), "orch")
    self.assertIn("--no-pipeline_step_boundary", orch)
    self.assertIn("--no-stage_next_microbatch", orch)

  def test_the_pipelined_step_boundary_flags_reach_the_orchestrator(self):
    orch = _jobset(
        _render(PIPELINE_STEP_BOUNDARY="true", STAGE_NEXT_MICROBATCH="1"),
        "orch",
    )
    self.assertIn("--pipeline_step_boundary", orch)
    self.assertIn("--stage_next_microbatch", orch)
    self.assertNotIn("--no-pipeline_step_boundary", orch)
    self.assertNotIn("--no-stage_next_microbatch", orch)

  def test_staging_without_the_pipelined_boundary_is_refused(self):
    with self.assertRaisesRegex(
        AssertionError, "STAGE_NEXT_MICROBATCH=true requires"
    ):
      _render(PIPELINE_STEP_BOUNDARY="false", STAGE_NEXT_MICROBATCH="true")

  def test_the_sharded_log_prob_head_is_on_in_the_trainer_only(self):
    self.assertRegex(
        _jobset(self.default, "train"), r"\bTUNIX_LOGPS_SHARDED_HEAD=local\b"
    )
    for suffix in ("orch", "roll-0"):
      self.assertNotIn(
          "TUNIX_LOGPS_SHARDED_HEAD", _jobset(self.default, suffix), suffix
      )

  def test_the_sharded_log_prob_head_can_be_turned_off(self):
    trainer = _jobset(_render(TUNIX_LOGPS_SHARDED_HEAD="0"), "train")
    self.assertRegex(trainer, r"\bTUNIX_LOGPS_SHARDED_HEAD=0\b")
    self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD=local", trainer)

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

  def test_tuned_flags_set_exactly_the_nine_keys_at_the_bench_cell(self):
    before = _maxtext_flags(_jobset(self.bench_off, "train"))
    after = _maxtext_flags(_jobset(self.bench_on, "train"))
    self.assertEqual(_flag_changes(before, after), _TUNED_KEYS)

  def test_tuned_flags_change_only_the_trainer_flags(self):
    off_docs = self.bench_off.split("\n---\n")
    on_docs = self.bench_on.split("\n---\n")
    self.assertLen(on_docs, len(off_docs))
    changed = [on for on, off in zip(on_docs, off_docs) if on != off]
    self.assertEqual(changed, [_jobset(self.bench_on, "train")])
    off = _jobset(self.bench_off, "train")
    on = _jobset(self.bench_on, "train")
    self.assertNotIn("offloading_copy_to_sparsecore", off)
    # The trainer env, the Pathways worker env and the Pathways proxy args.
    self.assertEqual(on.count(_SC_FLAG), 3)
    self.assertEqual(_without_tuned_flags(on), _without_tuned_flags(off))

  def test_tuned_flags_false_turns_the_block_off(self):
    trainer = _jobset(self.default_off, "train")
    flags = _maxtext_flags(trainer)
    self.assertEqual(
        {key: _value(flags, key) for key in _TUNED_KEYS},
        {key: off for key, (off, _) in _TUNED_KEYS.items()},
    )
    self.assertNotIn("offloading_copy_to_sparsecore", trainer)
    # The fully dropless buffers and chunks of 512, as before the switch.
    self.assertEqual(flags["ragged_buffer_factor"], "-1")
    self.assertEqual(flags["context"], "offload")
    self.assertNotIn("retry_when_tokens_dropped", flags)
    self.assertIn("--compute_logps_chunk_size=512 ", trainer)

  def test_tuned_flags_keep_four_chunks_without_a_buffer_factor(self):
    # The fully dropless layout was not measured with two chunks.
    trainer = _jobset(_render(RAGGED_BUFFER_FACTOR="-1"), "train")
    before = _maxtext_flags(_jobset(self.default_off, "train"))
    expected = dict(_TUNED_KEYS)
    del expected["num_moe_token_chunks"]
    self.assertEqual(_flag_changes(before, _maxtext_flags(trainer)), expected)
    self.assertEqual(_maxtext_flags(trainer)["num_moe_token_chunks"], "4")
    self.assertEqual(trainer.count(_SC_FLAG), 3)

  def test_tuned_flags_with_the_capacity_factor_trainer(self):
    # DROPLESS=false runs two chunks with or without the switch.
    before = _maxtext_flags(
        _jobset(_render(DROPLESS="false", TRAINER_TUNED_FLAGS="false"), "train")
    )
    trainer = _jobset(_render(DROPLESS="false"), "train")
    expected = dict(_TUNED_KEYS)
    del expected["num_moe_token_chunks"]
    self.assertEqual(_flag_changes(before, _maxtext_flags(trainer)), expected)
    self.assertEqual(_maxtext_flags(trainer)["num_moe_token_chunks"], "2")
    self.assertEqual(trainer.count(_SC_FLAG), 3)

  def test_tuned_flags_keep_an_explicit_chunk_count(self):
    flags = _maxtext_flags(
        _jobset(_render(NUM_MOE_TOKEN_CHUNKS="4", **_BENCH_CELL), "train")
    )
    self.assertEqual(flags["num_moe_token_chunks"], "4")
    self.assertEqual(flags["gdn_cp_mode"], "seq")

  def test_tuned_flags_keep_an_explicit_sparsecore_copy_flag(self):
    explicit = "--xla_tpu_enable_offloading_copy_to_sparsecore=true"
    libtpu_args = f"{explicit} --xla_tpu_scoped_vmem_limit_kib=65472"
    trainer = _jobset(_render(TRAINER_LIBTPU_INIT_ARGS=libtpu_args), "train")
    self.assertEqual(trainer.count(explicit), 3)
    self.assertNotIn(_SC_FLAG, trainer)

  def test_tuned_flags_reject_other_values(self):
    with self.assertRaisesRegex(
        AssertionError, "TRAINER_TUNED_FLAGS must be true or false"
    ):
      _render(TRAINER_TUNED_FLAGS="yes")

  def test_tuned_flags_are_off_in_the_other_397b_v7x_recipes(self):
    for name in (
        "mlperf_397b_256_v7x.sh",
        "mlperf_397b_1024_v7x.sh",
        "mlperf_gbs512.sh",
    ):
      render = _render(_RECIPE.with_name(name))
      self.assertNotIn("offloading_copy_to_sparsecore", render, name)
      flags = _maxtext_flags(_jobset(render, "train"))
      for key, (off, _) in _TUNED_KEYS.items():
        if key != "num_moe_token_chunks":  # two in these recipes either way
          self.assertEqual(_value(flags, key), off, f"{name}: {key}")

  def test_head_and_staging_are_off_in_the_other_397b_v7x_recipes(self):
    for name in (
        "mlperf_397b_256_v7x.sh",
        "mlperf_397b_1024_v7x.sh",
        "mlperf_gbs512.sh",
    ):
      render = _render(_RECIPE.with_name(name))
      self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD", render, name)
      orch = _jobset(render, "orch")
      self.assertIn("--no-pipeline_step_boundary", orch, name)
      self.assertIn("--no-stage_next_microbatch", orch, name)


if __name__ == "__main__":
  absltest.main()
