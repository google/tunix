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
    # Defaults in the MaxText of AI-Hypercomputer/maxtext#5698 and #5758; see
    # _BENCH_MAXTEXT_KEYS.
    "gdn_conv1d_dense_token_stack": "false",
    "gdn_conv1d_input_in_qkv_dtype": "false",
    "moe_gmm_v2_dlhs_transpose_rhs": "false",
    "moe_roe_packed_routing_gather": "false",
    "moe_router_grad_from_wo_input": "false",
    "moe_routing_index_math": "sort",
    "moe_routing_maps": "remat",
    "moe_unsplit_prefused_wi": "false",
    "moe_wi_gmm_fused_swiglu": "false",
    # remat_policy=custom's per-tensor keys (MaxText base.yml).
    "query_proj": "remat",
    "key_proj": "remat",
    "value_proj": "remat",
    "out_proj": "remat",
    "qkv_proj": "remat",
}
_TUNED_KEYS = {
    "cast_grads_after_all_reduce": ("false", "true"),
    "context_parallel_strategy": ("ring", "all_gather"),
    "custom_mesh_and_rule": ("cp-as-ep", "cp-as-ep-dense-no-cp"),
    "decoder_layer_input": ("offload", "device"),
    "gdn_conv1d_dense_token_stack": ("false", "true"),
    "gdn_conv1d_input_in_qkv_dtype": ("false", "true"),
    "gdn_cp_matmul_precision": ("highest", "high"),
    "gdn_cp_mode": ("head", "seq"),
    "gdn_cp_state": ("remat", "device"),
    "moe_gmm_v2_dlhs_transpose_rhs": ("false", "true"),
    "moe_roe_packed_routing_gather": ("false", "true"),
    "moe_router_grad_from_wo_input": ("false", "true"),
    "moe_routing_index_math": ("sort", "count"),
    "moe_routing_maps": ("remat", "device"),
    "moe_unsplit_prefused_wi": ("false", "true"),
    "moe_wi_gmm_fused_swiglu": ("false", "true"),
    "num_moe_token_chunks": ("4", "2"),
    "sa_fuse_reciprocal": ("false", "true"),
    "wi_tile_dlhs_mlp_dim": ("1024", "2048"),
    "wi_tile_drhs_batch_seq": ("256", "512"),
    "wi_tile_fwd_batch_seq": ("256", "512"),
    "wo_tile_dlhs_batch_seq": ("256", "512"),
    "wo_tile_drhs_batch_seq": ("256", "512"),
    "wo_tile_fwd_batch_seq": ("256", "512"),
}
# Tuned keys that MaxText before AI-Hypercomputer/maxtext#5721 (e.g. e526ea02)
# does not define: two from AI-Hypercomputer/maxtext#5698 and seven from
# AI-Hypercomputer/maxtext#5758, which is on top of it. MaxText refuses a key it
# does not define when the trainer starts, so only TRAINER_TUNED_FLAGS=true may
# pass them.
_BENCH_MAXTEXT_KEYS = (
    "gdn_conv1d_dense_token_stack",
    "gdn_conv1d_input_in_qkv_dtype",
    "moe_gmm_v2_dlhs_transpose_rhs",
    "moe_roe_packed_routing_gather",
    "moe_router_grad_from_wo_input",
    "moe_routing_index_math",
    "moe_routing_maps",
    "moe_unsplit_prefused_wi",
    "moe_wi_gmm_fused_swiglu",
)
# The custom mesh rule TRAINER_TUNED_FLAGS=true selects, which
# AI-Hypercomputer/maxtext#5783 (in #5721) defines and MaxText before #5721
# does not: only TRAINER_TUNED_FLAGS=true may pass it.
_BENCH_MESH_RULE = "cp-as-ep-dense-no-cp"
# The launch whose trainer the RL trainer bench (AI-Hypercomputer/maxtext#5701)
# measured TRAINER_TUNED_FLAGS on: the replay layout (buffer factor 2.75 with
# the dropless replay), with context on device, log-prob chunks of 4096 with
# the fused log-prob head, and packed rows padded first.
_BENCH_CELL = {
    "BATCH_SIZE": "72",
    "RAGGED_BUFFER_FACTOR": "2.75",
    "RETRY_WHEN_TOKENS_DROPPED": "true",
    "REMAT_CONTEXT": "device",
    "MOE_EXCLUDE_PADDING": "true",
    "LOG_MOE_DROPS": "true",
    "COMPUTE_LOGPS_CHUNK_SIZE": "4096",
    "TUNIX_LOGPS_SHARDED_HEAD": "fused",
    "PACK_PAD_POSITION": "start",
}
# The 256-chip trainer meshes and the keys TRAINER_TUNED_FLAGS adds at them.
_MESH_32X2X8 = {
    "TRAINER_TPU_SLICE": "tpu7x:4x8x8",
    "TRAINER_MESH_FSDP": "32",
    "TRAINER_MESH_EXPERT": "2",
    "TRAINER_MESH_CONTEXT": "8",
}
_MESH_64X1X8 = {
    "TRAINER_TPU_SLICE": "tpu7x:4x8x8",
    "TRAINER_MESH_FSDP": "64",
    "TRAINER_MESH_EXPERT": "1",
    "TRAINER_MESH_CONTEXT": "8",
}
_PROJ_KEYS = ("query_proj", "key_proj", "value_proj", "out_proj", "qkv_proj")


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
    # Base-2 exponentials stay on: the block no longer turns them off.
    self.assertEqual(flags["sa_use_base2_exp"], "true")
    # The block's mesh rule replaces the recipe's cp-as-ep.
    self.assertEqual(flags["custom_mesh_and_rule"], _BENCH_MESH_RULE)
    self.assertIn("--compute_logps_chunk_size=4096 ", trainer)
    self.assertRegex(trainer, r"\bTUNIX_LOGPS_SHARDED_HEAD=fused\b")
    self.assertRegex(
        _jobset(self.default, "orch"), r"--pack_pad_position=start\b"
    )
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

  def test_packed_rows_pad_first_with_the_tuned_flags(self):
    orch = _jobset(self.default, "orch")
    self.assertRegex(orch, r"--pack_pad_position=start\b")
    self.assertNotIn("--pack_pad_align", orch)
    # An explicit value wins.
    orch = _jobset(_render(PACK_PAD_POSITION="end"), "orch")
    self.assertRegex(orch, r"--pack_pad_position=end\b")
    self.assertNotIn("--pack_pad_position=start", orch)

  def test_packed_rows_pad_at_the_end_without_the_tuned_flags(self):
    orch = _jobset(self.default_off, "orch")
    self.assertNotIn("--pack_pad_position", orch)
    self.assertNotIn("--pack_pad_align", orch)

  def test_the_packed_row_pad_position_reaches_only_the_orchestrator(self):
    render = _render(
        TRAINER_TUNED_FLAGS="false", PACK_PAD_POSITION="start",
        PACK_PAD_ALIGN="8192",
    )
    orch = _jobset(render, "orch")
    self.assertRegex(orch, r"--pack_pad_position=start\b")
    self.assertRegex(orch, r"--pack_pad_align=8192\b")
    docs = render.split("\n---\n")
    default_docs = self.default_off.split("\n---\n")
    self.assertLen(docs, len(default_docs))
    changed = [doc for doc, before in zip(docs, default_docs) if doc != before]
    self.assertEqual(changed, [orch])

  def test_the_sharded_log_prob_head_is_on_in_the_trainer_only(self):
    self.assertRegex(
        _jobset(self.default, "train"), r"\bTUNIX_LOGPS_SHARDED_HEAD=fused\b"
    )
    for suffix in ("orch", "roll-0"):
      self.assertNotIn(
          "TUNIX_LOGPS_SHARDED_HEAD", _jobset(self.default, suffix), suffix
      )
    # TRAINER_TUNED_FLAGS=false keeps the "local" head; an explicit value wins.
    off = _jobset(self.default_off, "train")
    self.assertRegex(off, r"\bTUNIX_LOGPS_SHARDED_HEAD=local\b")
    self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD=fused", off)
    local = _jobset(_render(TUNIX_LOGPS_SHARDED_HEAD="local"), "train")
    self.assertRegex(local, r"\bTUNIX_LOGPS_SHARDED_HEAD=local\b")
    self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD=fused", local)
    self.assertIn("--compute_logps_chunk_size=4096 ", local)

  def test_the_sharded_log_prob_head_can_be_turned_off(self):
    # Without the head, chunks of 4096 failed to compile on TPU: 1024 instead.
    for value in ("0", "false", "OFF", "No"):
      trainer = _jobset(_render(TUNIX_LOGPS_SHARDED_HEAD=value), "train")
      self.assertRegex(trainer, rf"\bTUNIX_LOGPS_SHARDED_HEAD={value}\b")
      self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD=local", trainer)
      self.assertNotIn("TUNIX_LOGPS_SHARDED_HEAD=fused", trainer)
      self.assertIn("--compute_logps_chunk_size=1024 ", trainer, value)
    # An explicit chunk size still wins.
    trainer = _jobset(
        _render(TUNIX_LOGPS_SHARDED_HEAD="0", COMPUTE_LOGPS_CHUNK_SIZE="512"),
        "train",
    )
    self.assertIn("--compute_logps_chunk_size=512 ", trainer)

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

  def test_tuned_flags_set_exactly_the_tuned_keys_at_the_bench_cell(self):
    before = _maxtext_flags(_jobset(self.bench_off, "train"))
    after = _maxtext_flags(_jobset(self.bench_on, "train"))
    self.assertEqual(_flag_changes(before, after), _TUNED_KEYS)

  def test_tuned_flags_leave_moe_route_at_its_default(self):
    # moe_route=device was inert in the bench's MaxText and is live at the pin.
    self.assertNotIn("moe_route", _maxtext_flags(_jobset(self.default, "train")))

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
    for key in _BENCH_MAXTEXT_KEYS:
      self.assertNotIn(key, flags)
    self.assertNotIn(_BENCH_MESH_RULE, trainer)
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

  def test_tuned_flags_at_the_256_chip_mesh_32x2x8(self):
    before = _maxtext_flags(_jobset(self.bench_on, "train"))
    flags = _maxtext_flags(
        _jobset(_render(**_BENCH_CELL, **_MESH_32X2X8), "train")
    )
    self.assertEqual(
        _flag_changes(before, flags),
        {key: ("remat", "device") for key in _PROJ_KEYS},
    )
    self.assertEqual(flags["num_moe_token_chunks"], "2")
    self.assertEqual(flags["moe_chunk_pipeline"], "true")
    self.assertEqual(flags["moe_roe_packed_routing_gather"], "true")

  def test_tuned_flags_at_the_256_chip_mesh_64x1x8(self):
    before = _maxtext_flags(_jobset(self.bench_on, "train"))
    flags = _maxtext_flags(
        _jobset(_render(**_BENCH_CELL, **_MESH_64X1X8), "train")
    )
    expected = {key: ("remat", "device") for key in _PROJ_KEYS}
    # One token chunk, so no chunk pipeline, and so no packed routing gather,
    # which MaxText refuses without the pipeline.
    expected["num_moe_token_chunks"] = ("2", "1")
    expected["moe_chunk_pipeline"] = ("true", "false")
    expected["moe_roe_packed_routing_gather"] = ("true", "false")
    self.assertEqual(_flag_changes(before, flags), expected)

  def test_tuned_flags_keep_an_explicit_chunk_count_at_64x1x8(self):
    flags = _maxtext_flags(
        _jobset(
            _render(NUM_MOE_TOKEN_CHUNKS="2", **_BENCH_CELL, **_MESH_64X1X8),
            "train",
        )
    )
    self.assertEqual(flags["num_moe_token_chunks"], "2")
    self.assertEqual(flags["moe_chunk_pipeline"], "true")
    self.assertEqual(flags["moe_roe_packed_routing_gather"], "true")
    for key in _PROJ_KEYS:
      self.assertEqual(flags[key], "device", key)

  def test_tuned_flags_false_leaves_the_256_chip_meshes_alone(self):
    for mesh in (_MESH_32X2X8, _MESH_64X1X8):
      flags = _maxtext_flags(
          _jobset(_render(TRAINER_TUNED_FLAGS="false", **mesh), "train")
      )
      for key in _PROJ_KEYS:
        self.assertNotIn(key, flags, mesh)
      self.assertEqual(flags["num_moe_token_chunks"], "4", mesh)
      self.assertEqual(flags["moe_chunk_pipeline"], "true", mesh)

  def test_one_token_chunk_turns_the_chunk_pipeline_off(self):
    # MaxText refuses moe_chunk_pipeline=true with one token chunk, and
    # moe_roe_packed_routing_gather=true without the chunk pipeline.
    flags = _maxtext_flags(
        _jobset(_render(NUM_MOE_TOKEN_CHUNKS="1", **_BENCH_CELL), "train")
    )
    self.assertEqual(flags["num_moe_token_chunks"], "1")
    self.assertEqual(flags["moe_chunk_pipeline"], "false")
    self.assertEqual(flags["moe_roe_packed_routing_gather"], "false")
    for key in _PROJ_KEYS:
      self.assertNotIn(key, flags)

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
      for key in _BENCH_MAXTEXT_KEYS:
        self.assertNotIn(key, flags, name)
      self.assertNotIn(_BENCH_MESH_RULE, render, name)

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
      self.assertNotIn("--pack_pad_position", orch, name)

  def test_trajectory_store_is_enabled_by_default(self):
    orch = _jobset(self.default, "orch")
    self.assertIn("--no-disable_trajectory_store", orch)
    self.assertIn("--trajectory_store_root_dir=", orch)

  def test_disable_trajectory_store_omits_store_args_and_passes_flag(self):
    render = _render(DISABLE_TRAJECTORY_STORE="true")
    orch = _jobset(render, "orch")
    self.assertIn("--disable_trajectory_store", orch)
    self.assertNotIn("--no-disable_trajectory_store", orch)
    self.assertNotIn("--trajectory_store_root_dir", orch)
    self.assertNotIn("--trajectory_store_db_url", orch)

  def test_enable_trajectory_store_false_disables_store(self):
    render = _render(ENABLE_TRAJECTORY_STORE="false")
    orch = _jobset(render, "orch")
    self.assertIn("--disable_trajectory_store", orch)
    self.assertNotIn("--no-disable_trajectory_store", orch)
    self.assertNotIn("--trajectory_store_root_dir", orch)
    self.assertNotIn("--trajectory_store_db_url", orch)

  def test_256_chip_slice_uses_64x1x8_tuned_mesh(self):
    render = _render(TRAINER_TPU_SLICE="tpu7x:4x8x8")
    trainer = _jobset(render, "train")
    flags = _maxtext_flags(trainer)
    self.assertIn("--mesh_fsdp=64 ", trainer)
    self.assertIn("--mesh_expert=1 ", trainer)
    self.assertIn("--mesh_context=8 ", trainer)
    self.assertEqual(flags["num_moe_token_chunks"], "1")
    self.assertEqual(flags["moe_chunk_pipeline"], "false")
    self.assertEqual(flags["query_proj"], "device")

  def test_pad_start_with_finite_moe_capacity_requires_exclude_padding(self):
    with self.assertRaisesRegex(
        AssertionError,
        "PACK_PAD_POSITION=start with finite MoE capacity requires",
    ):
      _render(
          RAGGED_BUFFER_FACTOR="2.75",
          RETRY_WHEN_TOKENS_DROPPED="true",
          MOE_EXCLUDE_PADDING="false",
          PACK_PAD_POSITION="start",
      )


if __name__ == "__main__":
  absltest.main()
