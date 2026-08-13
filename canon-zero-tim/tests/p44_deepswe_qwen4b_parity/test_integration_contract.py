"""Static guards for the P44 training and evidence integration path."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


class P44IntegrationContractTest(unittest.TestCase):

  def test_both_dataset_entrypoints_use_the_modern_datasets_api(self):
    notebook = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    dataset_call = notebook.split("datasets_lib.load_dataset(", 1)[1].split(
        ")", 1
    )[0]
    self.assertNotIn("trust_remote_code", dataset_call)
    cli_data = (ROOT / "examples/deepswe/deepswe_data.py").read_text()
    self.assertNotIn("trust_remote_code", cli_data)

  def test_training_selects_physical_split_from_role_size(self):
    text = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn("32: deepswe_contract.split_4x4x4_role_devices", text)
    self.assertIn("64: deepswe_contract.split_4x4x8_role_devices", text)
    self.assertIn("128: deepswe_contract.split_4x8x8_role_devices", text)
    self.assertIn("deepswe_contract.split_4x4x4_role_devices", text)
    self.assertIn("deepswe_contract.split_4x4x8_role_devices", text)
    self.assertIn("deepswe_contract.split_4x8x8_role_devices", text)
    self.assertIn("[P34.DEVICE_INVENTORY] PASS", text)

  def test_reviewed_agentic_batch_semantics_are_present(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    grpo = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    self.assertIn("prompts = [chat_lists]", learner)
    self.assertIn("configured_compute_logps", grpo)
    self.assertIn(
        "configured_compute_logps * self.algo_config.num_generations", grpo
    )
    self.assertIn("[{marker}.LOGPS_BATCH]", grpo)

  def test_artifacts_precede_alignment_and_use_active_directory(self):
    text = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    artifact = text.index("deepswe_debug.persist_batch(")
    alignment = text.index("if alignment.enabled():", artifact)
    self.assertLess(artifact, alignment)
    self.assertIn("output_dir=deepswe_debug.artifact_directory()", text)
    self.assertIn("deepswe_debug.rollout_only()", text)
    self.assertIn('"p44-qwen4b-parity-64",', text)
    self.assertIn('"p44-qwen4b-parity-128",', text)

  def test_rollout_only_marker_and_alignment_are_shared(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    alignment = (ROOT / "tunix/rl/alignment.py").read_text()
    self.assertIn("marker = deepswe_debug.marker_prefix()", learner)
    self.assertIn("deepswe_debug.rollout_only()", learner)
    self.assertIn("p44_parity", alignment)
    self.assertIn("sum((p39_pilot, p43_debug, p44_parity)) == 1", alignment)

  def test_onehost_is_shared_dp1_tp4_docker_and_default_off(self):
    script = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    runner = (
        ROOT
        / "canon-zero-tim/tests/p44_deepswe_qwen4b_parity/"
        "run_onehost_deepswe_v5p.sh"
    ).read_text()
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text()
    grpo = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text()
    trainer = (ROOT / "tunix/sft/peft_trainer.py").read_text()
    self.assertIn(
        'ONEHOST_SMOKE = _ONEHOST_RAW == "1"', script
    )
    self.assertIn('rollout_dims = [("dp", 1), ("tp", 4)]', script)
    self.assertIn('train_dims = [("dp", 1), ("tp", 4)]', script)
    self.assertIn("rollout_devices = shared_devices", script)
    self.assertIn("train_devices = shared_devices", script)
    self.assertIn('{"backend": "docker"} if ONEHOST_SMOKE', script)
    self.assertIn('"enable_prefix_caching": not P34_DEEPSWE', script)
    self.assertIn(
        'vllm_rollout_dict["rollout_vllm_kwargs"]'
        '["enable_prefix_caching"] = False',
        script,
    )
    self.assertIn("onehost_before", learner)
    self.assertIn('"commits": 0', learner)
    self.assertIn("INCONCLUSIVE_NO_SIGNAL", learner)
    self.assertIn(
        "have_actor_mesh and not deepswe_debug.rollout_only()", grpo
    )
    self.assertIn("_deepswe_onehost_no_commit", trainer)
    self.assertIn("optimizer_boundary_skipped commits=0", trainer)
    self.assertIn("export JAX_PLATFORMS=tpu,cpu", runner)
    self.assertIn("/mnt/disks/tunix-data/deepswe-onehost-evidence/", runner)
    self.assertIn("git status --porcelain --untracked-files=no", runner)
    self.assertIn("DEEPSWE_ONEHOST_ALLOW_DIRTY", runner)
    self.assertIn("--max_prompt_length 3584", runner)
    self.assertIn("--max_response_length 512", runner)
    self.assertIn("--max_turns 2", runner)
    self.assertIn("--max_num_batched_tokens 512", runner)
    self.assertIn(
        "training_data_sharding_axis = (train_axis_names[0],)",
        script,
    )
    self.assertIn(
        "data_sharding_axis=training_data_sharding_axis",
        script,
    )

  def test_onehost_does_not_change_production_role_split(self):
    script = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    p34 = script.index("if P34_DEEPSWE:", script.index("# 1. Resolve"))
    local = script.index("elif ONEHOST_SMOKE:", p34)
    legacy = script.index("elif rollout_fsdp", local)
    self.assertLess(p34, local)
    self.assertLess(local, legacy)
    production = script[p34:local]
    self.assertIn("p34.dp_size", production)
    self.assertIn("p34.tp_size", production)

  def test_postflight_routes_p44_before_existing_deepswe_lanes(self):
    text = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text()
    reservation = text.index("report_keys+=(CANON_P44_DEBUG_DIR)")
    p44 = text.index(
        "tests/p44_deepswe_qwen4b_parity/classify_run.py"
    )
    p43 = text.index("tests/p43_deepswe_debug/classify_run.py", p44)
    p39 = text.index("tests/p39_deepswe_pilot/classify_run.py", p43)
    p34 = text.index("tests/p34_deepswe/classify_run.py", p39)
    self.assertLess(reservation, p44)
    self.assertLess(p44, p43)
    self.assertLess(p43, p39)
    self.assertLess(p39, p34)
    self.assertIn('--topology "$CANON_P44_TOPOLOGY"', text[p44 - 400:p43])
    self.assertIn('--debug-dir "$CANON_P44_DEBUG_DIR"', text[p44:p43])


if __name__ == "__main__":
  unittest.main()
