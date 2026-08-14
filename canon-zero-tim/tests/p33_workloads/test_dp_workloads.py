"""Tests for the frozen 64-device DP/TP workload contracts."""

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
from tunix.rl.agentic import agentic_rl_learner


def _environment(name: str) -> dict[str, str]:
  workload = dp_workloads.get_workload(name)
  environ = {
      "CANON_P32_WORKLOAD": name,
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P32_DP_REDUCTION_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_WANDB_ONLINE_REQUIRED": "1",
      "CANON_P31_MONOTONIC_METRICS": "1",
      "CANON_WANDB_PROJECT": workload.wandb_project,
      "CANON_WANDB_GROUP": f"qwen3-{name}-test",
      "CANON_WANDB_RUN_NAME": f"p33-{name}-test",
      "WANDB_MODE": "online",
      "WANDB_API_KEY": "test-key-not-a-credential",
      "CANON_DP_SIZE": str(workload.dp_size),
      "CANON_TP_SIZE": str(workload.tp_size),
      "CANON_TOTAL_DEVICES": str(workload.total_devices),
      "CANON_ENGINE_DP_SIZE": str(workload.dp_size),
      "CANON_QWEN3_TP_SIZE": str(workload.tp_size),
      "CANON_GLOBAL_PROMPTS": str(workload.global_prompts),
      "CANON_LOCAL_PROMPTS": str(workload.local_prompts),
      "CANON_NUM_GENERATIONS": str(workload.num_generations),
      "CANON_LOCAL_TRAJECTORIES": str(workload.local_trajectories),
      "CANON_GLOBAL_TRAJECTORIES": str(workload.global_trajectories),
      "CANON_LOGPROB_M": str(workload.local_m),
      "MIN_TOKEN_BUCKET": str(workload.global_m),
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
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P33_SHORT_ALIGNMENT": "0",
      "CANON_OPT_STATE_RESIDENT": "0",
      "CANON_P30_OPT_STATE_OFFLOAD": "1",
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
      "CANON_P30_RELEASE_CAPTURED_STATE": "1",
      "CANON_P30_RESHARD_ACCUMULATOR": "1",
      "FL_SHARED_MESH": f"1,{workload.tp_size}",
      "XLA_FLAGS": (
          "--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false"
      ),
  }
  if name.startswith("frozenlake"):
    environ["CANON_P33_ENABLE_EVAL"] = "0"
    environ["CANON_P33_DISABLE_EVAL"] = "1"
    environ["CANON_P31_ENABLE_EVAL"] = "0"
    environ["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"] = "0"
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

  def test_frozenlake_dp8_tp8_resident_candidate_geometry(self):
    workload = dp_workloads.get_workload("frozenlake-dp8-tp8")
    self.assertEqual((workload.dp_size, workload.tp_size), (8, 8))
    self.assertEqual(workload.total_devices, 64)
    self.assertEqual(workload.global_trajectories, 256)
    self.assertEqual(workload.local_trajectories, 32)
    self.assertEqual(workload.global_m, 2048)
    self.assertIn("--mesh_dp=8", workload.command())
    self.assertIn("--mesh_tp=8", workload.command())
    self.assertIn("--vllm_max_num_seqs=32", workload.command())
    self.assertIn("--vllm_max_num_batched_tokens=256", workload.command())

    environ = _environment("frozenlake-dp8-tp8")
    environ.update({
        "CANON_OPT_STATE_RESIDENT": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "0",
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "FL_SHARED_MESH": "8,8",
    })
    dp_workloads.validate_environment(
        workload, environ, require_reduction_admission=True
    )

    segmented_geometry = agentic_rl_learner._segmented_update_geometry({
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    })
    self.assertEqual(
        segmented_geometry,
        (256, 8, "[CANON_P33_DP8]", True),
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

  def test_gsm8k_rollout_limits_are_per_dp_rank(self):
    workload = dp_workloads.get_workload("gsm8k")
    command = workload.command()
    self.assertIn("--rollout_vllm_max_num_seqs=16", command)
    self.assertIn("--rollout_vllm_max_num_batched_tokens=256", command)
    self.assertEqual(
        workload.dp_size * workload.local_trajectories,
        workload.global_trajectories,
    )
    self.assertEqual(workload.dp_size * workload.local_m, workload.global_m)
    self.assertNotIn("--rollout_vllm_max_num_seqs=256", command)
    self.assertNotIn("--rollout_vllm_max_num_batched_tokens=4096", command)

  def test_gsm8k_recipe_validates_topology_local_scheduler_limits(self):
    source = (
        Path(__file__).parents[3]
        / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        "expected_num_seqs = P32_WORKLOAD.local_trajectories", source
    )
    self.assertIn("P32_WORKLOAD.local_m if CANON_P32_WORKLOAD", source)
    self.assertNotIn(
        "expected_batched_tokens = 4096 if CANON_P32_WORKLOAD", source
    )

  def test_frozenlake_preserves_signed_convergence_lengths(self):
    workload = dp_workloads.get_workload("frozenlake")
    self.assertEqual(workload.max_steps, 450)
    self.assertFalse(workload.periodic_evaluation)
    self.assertEqual(
        (workload.max_prompt_length, workload.max_response_length),
        (4096, 2048),
    )

  def test_frozenlake_short_alignment_preserves_shape_contract(self):
    workload = dp_workloads.get_workload("frozenlake")
    command = workload.command(run_stage="alignment-short")
    self.assertIn("--batch_size=32", command)
    self.assertIn("--num_generations=8", command)
    self.assertIn("--max_prompt_length=4096", command)
    self.assertIn("--max_response_length=512", command)
    self.assertIn("--env_max_steps=2", command)
    self.assertIn("--vllm_max_num_batched_tokens=256", command)

  def test_gsm8k_short_alignment_preserves_full_response_shape(self):
    command = dp_workloads.get_workload("gsm8k").command(
        run_stage="alignment-short"
    )
    self.assertIn("--batch_size=32", command)
    self.assertIn("--num_generations=8", command)
    self.assertIn("--max_prompt_length=1024", command)
    self.assertIn("--max_response_length=1024", command)
    self.assertIn("--max_steps=1", command)
    self.assertIn("--rollout_vllm_max_num_seqs=16", command)
    self.assertIn("--rollout_vllm_max_num_batched_tokens=256", command)

  def test_gsm8k_envelope_short_preserves_shape_contract(self):
    command = dp_workloads.get_workload("gsm8k").command(
        run_stage="envelope-short"
    )
    self.assertIn("--batch_size=32", command)
    self.assertIn("--num_generations=8", command)
    self.assertIn("--max_prompt_length=1024", command)
    self.assertIn("--max_response_length=256", command)
    self.assertNotIn("--max_response_length=64", command)
    self.assertIn("--rollout_vllm_max_num_seqs=16", command)
    self.assertIn("--rollout_vllm_max_num_batched_tokens=256", command)
    self.assertIn("--max_steps=1", command)

  def test_frozenlake_rejects_gsm8k_envelope_short_stage(self):
    with self.assertRaisesRegex(ValueError, "only defined for GSM8K"):
      dp_workloads.get_workload("frozenlake").command(
          run_stage="envelope-short"
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

  def test_frozenlake_max_concurrency_defaults_to_256(self):
    workload = dp_workloads.get_workload("frozenlake")
    dp_workloads.validate_frozenlake_max_concurrency(
        workload, 256, _environment("frozenlake")
    )
    with self.assertRaisesRegex(ValueError, "must be 256"):
      dp_workloads.validate_frozenlake_max_concurrency(
          workload, 32, _environment("frozenlake")
      )

  def test_frozenlake_p38_stock_capture_admits_concurrency_32(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ.update({
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
        "CANON_P38_SERVING_CAPTURE_DIR": "/tmp/p38-capture",
        "CANON_P38_REQUEST_JOURNAL": "/tmp/p38-capture/journal.jsonl",
        "CANON_P38_INCIDENT_LEDGER": "/tmp/p38-capture/incident.jsonl",
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": "/tmp/p38-round",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
        "CANON_KV_UNIFIED": "0",
    })
    dp_workloads.validate_frozenlake_max_concurrency(
        workload, 32, environ
    )
    dp_workloads.validate_frozenlake_max_concurrency(
        workload, 256, environ
    )

  def test_frozenlake_p38_concurrency_32_scope_fails_closed(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ.update({
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
        "CANON_P38_SERVING_CAPTURE_DIR": "/tmp/p38-capture",
        "CANON_P38_REQUEST_JOURNAL": "/tmp/p38-capture/journal.jsonl",
        "CANON_P38_INCIDENT_LEDGER": "/tmp/p38-capture/incident.jsonl",
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": "/tmp/p38-round",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
        "CANON_KV_UNIFIED": "0",
    })
    guarded_names = (
        "CANON_P33_RUN_STAGE",
        "CANON_P33_NO_COMMIT",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED",
        "CANON_P38_PRECHECK_ONLY",
        "CANON_P38_CONTROLLED_EXIT",
        "CANON_P38_DIAGNOSTIC_ROUNDS",
        "CANON_P38_SERVING_CAPTURE_DIR",
        "CANON_P38_REQUEST_JOURNAL",
        "CANON_P38_INCIDENT_LEDGER",
        "CANON_P38_DIAGNOSTIC_ROUND_FILE",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH",
        "CANON_KV_UNIFIED",
    )
    for name in guarded_names:
      with self.subTest(missing=name):
        candidate = dict(environ)
        del candidate[name]
        with self.assertRaisesRegex(ValueError, "bounded stock P38"):
          dp_workloads.validate_frozenlake_max_concurrency(
              workload, 32, candidate
          )
    with self.assertRaisesRegex(ValueError, "bounded stock P38"):
      dp_workloads.validate_frozenlake_max_concurrency(
          workload, 64, environ
      )
    with self.assertRaisesRegex(ValueError, "bounded stock P38"):
      dp_workloads.validate_frozenlake_max_concurrency(
          dp_workloads.get_workload("frozenlake-dp8-tp8"), 32, environ
      )

  def test_frozenlake_recipe_uses_scoped_concurrency_contract(self):
    source = (
        Path(__file__).parents[3]
        / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text(encoding="utf-8")
    self.assertIn("validate_frozenlake_max_concurrency(", source)
    self.assertNotIn(
        '"max_concurrency": (args.max_concurrency, 256)', source
    )

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
    with self.assertRaisesRegex(ValueError, "CANON_P33_DISABLE_EVAL"):
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=False
      )

  def test_frozenlake_full_admits_explicit_evaluation(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "FL_SHARED_MESH": "16,4",
        "CANON_P33_ENABLE_EVAL": "1",
        "CANON_P33_DISABLE_EVAL": "0",
        "CANON_P31_ENABLE_EVAL": "1",
    })
    dp_workloads.validate_environment(
        workload, environ, require_reduction_admission=True
    )

  def test_frozenlake_evaluation_rejects_diagnostic_stage(self):
    environ = _environment("frozenlake")
    environ.update({
        "CANON_P33_ENABLE_EVAL": "1",
        "CANON_P33_DISABLE_EVAL": "0",
        "CANON_P31_ENABLE_EVAL": "1",
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
    })
    with self.assertRaisesRegex(ValueError, "committed full training"):
      dp_workloads.frozenlake_evaluation_enabled(environ)

  def test_frozenlake_evaluation_rejects_mismatched_learner_flag(self):
    environ = _environment("frozenlake")
    environ["CANON_P33_ENABLE_EVAL"] = "1"
    environ["CANON_P33_DISABLE_EVAL"] = "0"
    with self.assertRaisesRegex(ValueError, "must match"):
      dp_workloads.frozenlake_evaluation_enabled(environ)

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

  def test_contract_accepts_explicit_device_resident_optimizer(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_OPT_STATE_RESIDENT"] = "1"
    environ["CANON_P30_OPT_STATE_OFFLOAD"] = "0"
    dp_workloads.validate_environment(
        workload, environ, require_reduction_admission=False
    )
    self.assertEqual(
        dp_workloads.canonical_optimizer_placement(
            environ, require_explicit=True
        ),
        "device-resident",
    )

  def test_contract_rejects_ambiguous_optimizer_placement(self):
    workload = dp_workloads.get_workload("frozenlake")
    for resident, offload in (("1", "1"), ("0", "0")):
      with self.subTest(resident=resident, offload=offload):
        environ = _environment("frozenlake")
        environ["CANON_OPT_STATE_RESIDENT"] = resident
        environ["CANON_P30_OPT_STATE_OFFLOAD"] = offload
        with self.assertRaisesRegex(ValueError, "optimizer"):
          dp_workloads.validate_environment(
              workload, environ, require_reduction_admission=False
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

  def test_launch_accepts_frozenlake_short_alignment(self):
    workload = dp_workloads.get_workload("frozenlake")
    environ = _environment("frozenlake")
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P33_RUN_STAGE": "alignment-short",
        "CANON_P33_SHORT_ALIGNMENT": "1",
        "FL_SHARED_MESH": "16,4",
    })
    self.assertEqual(
        dp_workloads.requested_max_steps(workload, environ), 1
    )
    dp_workloads.validate_environment(
        workload, environ, require_reduction_admission=True
    )

  def test_launch_accepts_gsm8k_short_alignment(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ.update({
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
        "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P33_RUN_STAGE": "alignment-short",
        "CANON_P33_SHORT_ALIGNMENT": "1",
        "FL_SHARED_MESH": "16,4",
    })
    self.assertEqual(
        dp_workloads.requested_max_steps(workload, environ), 1
    )
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

  def test_envelope_short_requires_no_commit(self):
    workload = dp_workloads.get_workload("gsm8k")
    environ = _environment("gsm8k")
    environ["CANON_P33_RUN_STAGE"] = "envelope-short"
    environ["CANON_P33_NO_COMMIT"] = "1"
    self.assertEqual(
        dp_workloads.requested_max_steps(workload, environ), 1
    )
    environ["CANON_P33_NO_COMMIT"] = "0"
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
