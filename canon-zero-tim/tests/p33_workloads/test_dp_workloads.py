"""Tests for the frozen DP16 workload contracts."""

from __future__ import annotations

import os
from pathlib import Path
import types
import unittest
from unittest import mock

import datasets
import grain
import pandas as pd
from examples.frozenlake import data as frozenlake_data
from tunix.models.qwen3 import model as qwen3_model
from tunix.cli.utils import data as data_lib
from tunix.rl import dp_workloads


def _environment(name: str) -> dict[str, str]:
  environ = {
      "CANON_P32_WORKLOAD": name,
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P32_DP_REDUCTION_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_WANDB_ONLINE_REQUIRED": "1",
      "CANON_P31_MONOTONIC_METRICS": "1",
      "CANON_WANDB_PROJECT": f"zero-tim-{name}-dp16-tp4",
      "CANON_WANDB_GROUP": f"qwen3-{name}-dp16-tp4",
      "CANON_WANDB_RUN_NAME": f"p33-{name}-test",
      "WANDB_MODE": "online",
      "WANDB_API_KEY": "test-key-not-a-credential",
      "CANON_DP_SIZE": "16",
      "CANON_TP_SIZE": "4",
      "CANON_TOTAL_DEVICES": "64",
      "CANON_GLOBAL_PROMPTS": "32",
      "CANON_LOCAL_PROMPTS": "2",
      "CANON_NUM_GENERATIONS": "8",
      "CANON_LOCAL_TRAJECTORIES": "16",
      "CANON_GLOBAL_TRAJECTORIES": "256",
      "CANON_LOGPROB_M": "256",
      "MIN_TOKEN_BUCKET": "4096",
      "CANON_FIXED_AR": "1",
      "CANON_FIXED_AR_EMBED": "1",
      "CANON_RPA_VJP2": "1",
      "CANON_VJP2_MAX_SEQS": "1",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
      "CANON_PALLAS_LOGSOFTMAX": "1",
      "CANON_P32_DP16_SEGMENTED": "1",
      "CANON_P28_SEGMENTED_FORWARD": "1",
      "CANON_P28_SEGMENTED_TRAIN": "1",
      "CANON_P28_G6_UPDATE": "1",
      "CANON_P29_FULL_TRAIN": "1",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "1",
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
      "CANON_P30_RELEASE_CAPTURED_STATE": "1",
      "CANON_P30_RESHARD_ACCUMULATOR": "1",
      "FL_SHARED_MESH": "1,4",
      "XLA_FLAGS": (
          "--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false"
      ),
  }
  if name == "frozenlake":
    environ["CANON_P33_DISABLE_EVAL"] = "1"
  if name == "gsm8k":
    environ["CANON_GSM8K_GRAD_PROBE"] = "0"
  return environ


class DPWorkloadsTest(unittest.TestCase):

  def test_prompt_logprobs_chunk_each_dp_rank_at_canonical_local_m(self):
    patch = (
        Path(__file__).parents[2]
        / "patches"
        / "tpu_inference"
        / "06-tpu-runner.patch"
    ).read_text(encoding="utf-8")
    self.assertIn("def _canon_compute_prompt_logprobs(", patch)
    self.assertIn("for start in range(0, rows_per_dp, target_rows):", patch)
    self.assertNotIn(
        "processed prompt global rows must equal",
        patch,
    )
    self.assertIn("f\"chunks={prompt_logprob_chunks}\"", patch)

  def test_decode_logprobs_chunk_rows_above_canonical_m(self):
    patch = (
        Path(__file__).parents[2]
        / "patches"
        / "tpu_inference"
        / "06-tpu-runner.patch"
    ).read_text(encoding="utf-8")
    self.assertIn("def _canon_compute_decode_logprobs(", patch)
    self.assertIn("for start in range(0, rows, target_rows):", patch)
    self.assertIn("chunks={canon_logprob_chunks}", patch)
    self.assertIn(
        "_canon_compute_decode_logprobs(\n+                            logprobs_logits,",
        patch,
    )

  def test_registered_workloads_share_production_topology(self):
    gsm8k = dp_workloads.get_workload("gsm8k")
    frozenlake = dp_workloads.get_workload("frozenlake")
    for workload in (gsm8k, frozenlake):
      self.assertEqual((workload.dp_size, workload.tp_size), (16, 4))
      self.assertEqual(workload.global_trajectories, 256)
      self.assertEqual(workload.local_trajectories, 16)
      self.assertEqual(workload.global_m, 4096)
      self.assertIn("--mesh_dp=16", workload.command())
      self.assertIn("--mesh_tp=4", workload.command())
      self.assertNotIn("--mesh_fsdp=16", workload.command())
      self.assertIn(
          "--train_trajectory_micro_batch_size=16", workload.command()
      )

  def test_gsm8k_preserves_signed_local_recipe_lengths(self):
    workload = dp_workloads.get_workload("gsm8k")
    self.assertEqual(workload.max_steps, 200)
    self.assertEqual(
        (workload.max_prompt_length, workload.max_response_length),
        (1024, 1024),
    )

  def test_gsm8k_requires_gradient_probe_explicitly_disabled(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    del environ["CANON_GSM8K_GRAD_PROBE"]
    with self.assertRaisesRegex(ValueError, "environment mismatch"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_frozenlake_preserves_signed_convergence_lengths(self):
    workload = dp_workloads.get_workload("frozenlake")
    self.assertEqual(workload.max_steps, 450)
    self.assertFalse(workload.periodic_evaluation)
    self.assertEqual(
        (workload.max_prompt_length, workload.max_response_length),
        (4096, 2048),
    )

  def test_frozenlake_rollout_limits_are_per_dp_rank(self):
    workload = dp_workloads.get_workload("frozenlake")
    command = workload.command()
    self.assertIn("--vllm_max_num_seqs=16", command)
    self.assertIn("--vllm_max_num_batched_tokens=256", command)
    self.assertEqual(workload.dp_size * 16, workload.global_trajectories)
    self.assertEqual(workload.dp_size * 256, workload.global_m)
    self.assertNotIn("--vllm_max_num_seqs=256", command)
    self.assertNotIn("--vllm_max_num_batched_tokens=4096", command)

  def test_frozenlake_command_disables_periodic_evaluation(self):
    command = dp_workloads.get_workload("frozenlake").command()
    self.assertFalse(
        any(arg.startswith("--num_test_batches=") for arg in command)
    )
    self.assertFalse(
        any(arg.startswith("--eval_every_n_steps=") for arg in command)
    )

  def test_frozenlake_requires_evaluation_disabled_contract(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    del environ["CANON_P33_DISABLE_EVAL"]
    with self.assertRaisesRegex(ValueError, "environment mismatch"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_bounded_run_stage_commands(self):
    workload = dp_workloads.get_workload("gsm8k")
    self.assertIn("--max_steps=1", workload.command(run_stage="one-update"))
    self.assertIn("--max_steps=3", workload.command(run_stage="three-update"))
    self.assertIn(
        "--max_steps=200", workload.command(run_stage="full")
    )

  def test_contract_only_allows_unadmitted_reduction(self):
    workload = dp_workloads.get_workload("gsm8k")
    dp_workloads.validate_environment(
        workload,
        _environment("gsm8k"),
        require_reduction_admission=False,
    )

  def test_launch_rejects_unadmitted_reduction(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P32_TRAIN_ADMITTED"] = "1"
    environ["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"] = "1"
    environ["FL_SHARED_MESH"] = "16,4"
    with self.assertRaisesRegex(ValueError, "rank-local DP16 reduction"):
      dp_workloads.validate_environment(
          workload,
          environ,
          require_reduction_admission=True,
      )

  def test_launch_accepts_admitted_reduction(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ["CANON_P32_TRAIN_ADMITTED"] = "1"
    environ["CANON_P32_DP_REDUCTION_ADMITTED"] = "1"
    environ["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"] = "1"
    environ["FL_SHARED_MESH"] = "16,4"
    dp_workloads.validate_environment(
      workload, environ, require_reduction_admission=True
    )

  def test_launch_rejects_offline_wandb(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "FL_SHARED_MESH": "16,4",
        "WANDB_MODE": "offline",
    })
    with self.assertRaisesRegex(ValueError, "online W&B telemetry"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=True
      )

  def test_online_wandb_run_attestation(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    run = types.SimpleNamespace(
        project=environ["CANON_WANDB_PROJECT"],
        group=environ["CANON_WANDB_GROUP"],
        name=environ["CANON_WANDB_RUN_NAME"],
        settings=types.SimpleNamespace(mode="online"),
    )
    with mock.patch.object(
        dp_workloads,
        "_wandb_module",
        return_value=types.SimpleNamespace(run=run),
    ):
      self.assertEqual(
          dp_workloads.require_online_wandb_run(workload, environ),
          {
              "status": "online",
              "project": environ["CANON_WANDB_PROJECT"],
              "group": environ["CANON_WANDB_GROUP"],
              "name": environ["CANON_WANDB_RUN_NAME"],
          },
      )

  def test_online_wandb_run_rejects_missing_live_run(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    with mock.patch.object(
        dp_workloads,
        "_wandb_module",
        return_value=types.SimpleNamespace(run=None),
    ):
      with self.assertRaisesRegex(RuntimeError, "did not initialize"):
        dp_workloads.require_online_wandb_run(workload, environ)

  def test_launch_accepts_backward_no_commit(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P32_TRAIN_ADMITTED"] = "1"
    environ["CANON_P32_DP_REDUCTION_ADMITTED"] = "1"
    environ["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"] = "1"
    environ["CANON_P33_NO_COMMIT"] = "1"
    environ["CANON_P33_RUN_STAGE"] = "backward-no-commit"
    environ["FL_SHARED_MESH"] = "16,4"
    dp_workloads.validate_environment(
        workload, environ, require_reduction_admission=True
    )

  def test_contract_only_rejects_backward_no_commit(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P33_NO_COMMIT"] = "1"
    environ["CANON_P33_RUN_STAGE"] = "backward-no-commit"
    with self.assertRaisesRegex(ValueError, "cannot request backward"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_contract_only_rejects_training_admission(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P32_TRAIN_ADMITTED"] = "1"
    with self.assertRaisesRegex(ValueError, "training admission mismatch"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_launch_requires_training_admission(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ["CANON_P32_DP_REDUCTION_ADMITTED"] = "1"
    environ["FL_SHARED_MESH"] = "16,4"
    with self.assertRaisesRegex(ValueError, "training admission mismatch"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=True
      )

  def test_launch_requires_workload_admission(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P32_TRAIN_ADMITTED"] = "1"
    environ["CANON_P32_DP_REDUCTION_ADMITTED"] = "1"
    environ["FL_SHARED_MESH"] = "16,4"
    with self.assertRaisesRegex(ValueError, "workload launch admission"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=True
      )

  def test_environment_rejects_global_bucket_256(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ["MIN_TOKEN_BUCKET"] = "256"
    with self.assertRaisesRegex(ValueError, "environment mismatch"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_environment_rejects_missing_precision_flag(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX2"
    with self.assertRaisesRegex(ValueError, "excess_precision"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_run_stage_rejects_no_commit_mismatch(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P33_RUN_STAGE"] = "one-update"
    environ["CANON_P33_NO_COMMIT"] = "1"
    with self.assertRaisesRegex(ValueError, "stage/no-commit mismatch"):
      dp_workloads.requested_max_steps(workload, environ)

  def test_active_workload_is_default_off(self):
    self.assertIsNone(dp_workloads.active_workload({}))
    self.assertEqual(
        dp_workloads.active_workload({"CANON_P32_WORKLOAD": "gsm8k"}).name,
        "gsm8k",
    )

  def test_alignment_train_mode_includes_p31_and_p33(self):
    self.assertFalse(dp_workloads.requires_alignment_train_mode({}))
    self.assertTrue(
        dp_workloads.requires_alignment_train_mode(
            {"CANON_P31_CONVERGENCE": "1"}
        )
    )
    self.assertTrue(
        dp_workloads.requires_alignment_train_mode(
            {
                "CANON_P31_CONVERGENCE": "0",
                "CANON_P32_WORKLOAD": "frozenlake",
            }
        )
    )

  def test_alignment_train_mode_rejects_unknown_workload(self):
    with self.assertRaisesRegex(ValueError, "unknown canonical workload"):
      dp_workloads.requires_alignment_train_mode(
          {"CANON_P32_WORKLOAD": "unknown"}
      )

  def test_frozenlake_dataset_adds_required_prompt_column(self):
    frame = pd.DataFrame({"seed": [1, 2], "size": [4, 5], "p": [0.7, 0.8]})
    dataset = frozenlake_data.add_empty_prompt_column(
        datasets.Dataset.from_pandas(frame)
    )
    self.assertIn("prompts", dataset.column_names)
    self.assertEqual(dataset["prompts"], ["", ""])

    tokenizer = types.SimpleNamespace(encode=lambda value: [value])
    training, validation = data_lib.post_init_dataset(
        grain.MapDataset.source(dataset),
        tokenizer,
        batch_size=1,
        num_batches=2,
        max_prompt_length=4,
    )
    self.assertIsNone(validation)
    self.assertEqual(len(list(training)), 2)

  def test_qwen3_replicated_parameter_sharding_drops_fsdp_axis(self):
    config = qwen3_model.ModelConfig.qwen3_1p7b()
    self.assertIn("fsdp", repr(config.shd_config))
    dp_workloads.configure_replicated_parameter_sharding(config)
    self.assertNotIn("fsdp", repr(config.shd_config))
    self.assertIn("dp", repr(config.shd_config.act_btd))

  def test_unknown_workload_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "unknown canonical workload"):
      dp_workloads.get_workload("unknown")

  def test_mesh_rejects_incomplete_slice_before_topology_call(self):
    workload = dp_workloads.get_workload("gsm8k")
    with mock.patch.object(
        dp_workloads.mesh_utils, "create_device_mesh"
    ) as create:
      with self.assertRaisesRegex(ValueError, "exactly 64"):
        dp_workloads.create_mesh((), workload)
      create.assert_not_called()

  def test_module_environment_does_not_enable_workload(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertIsNone(dp_workloads.active_workload())


if __name__ == "__main__":
  unittest.main()
