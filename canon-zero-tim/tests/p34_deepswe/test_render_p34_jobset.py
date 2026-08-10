"""Fail-closed tests for the P34 4x8x8 JobSet renderer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p34_jobset.py"
SPEC = importlib.util.spec_from_file_location("p34_jobset_renderer", RENDERER)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P34 JobSet renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _base():
  return yaml.safe_load(
      (ROOT / "canon-zero-tim/cluster/jobset-256cluster-64chip.yaml").read_text()
  )


def _render(base=None, **changes):
  arguments = {
      "source_commit": "1" * 40,
      "source_branch": renderer.DEFAULT_SOURCE_BRANCH,
      "client_image": "registry.example/tunix@sha256:" + "2" * 64,
      "run_id": "gate-001",
      "stage": "three-update",
      "cpu_nodepool": "deepswe-cpu-pool",
      "worker_nodepool": "v5p-256-pool",
      "model_pvc": "models-pvc",
      "whitelist": "/mnt/disks/linchai_data/gold.jsonl",
      "whitelist_sha256": "3" * 64,
  }
  arguments.update(changes)
  return renderer.render(_base() if base is None else base, **arguments)


class RenderP34JobSetTest(unittest.TestCase):

  def test_proxy_delivers_excess_precision_through_env(self):
    document = _render()
    head = renderer._head(document)
    proxy = renderer._container(head["containers"], "pathways-proxy")
    entries = [e for e in proxy["env"] if e["name"] == "XLA_FLAGS"]
    self.assertEqual(
        entries,
        [{
            "name": "XLA_FLAGS",
            "value": "--xla_allow_excess_precision=false",
        }],
    )
    self.assertFalse([a for a in proxy["args"] if "excess_precision" in a])

  def test_rejects_base_with_raw_proxy_excess_precision_arg(self):
    base = _base()
    head = renderer._head(base)
    proxy = renderer._container(head["containers"], "pathways-proxy")
    proxy["args"].append("--xla_allow_excess_precision=false")
    with self.assertRaisesRegex(ValueError, "raw excess-precision"):
      _render(base=base)

  def test_published_branch_is_the_renderer_default(self):
    self.assertEqual(
        renderer.DEFAULT_SOURCE_BRANCH, "yuxzhang/canon-zero-tim"
    )
    env = renderer._env(_render())
    self.assertEqual(env["CANON_SOURCE_BRANCH"], renderer.DEFAULT_SOURCE_BRANCH)

  def test_render_is_nonmutating_and_closes_attempt_zero(self):
    base = _base()
    frozen = copy.deepcopy(base)
    document = _render(base)
    self.assertEqual(base, frozen)
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
    jobs = document["spec"]["replicatedJobs"]
    self.assertEqual(jobs[0]["template"]["spec"]["backoffLimit"], 0)
    worker = jobs[1]["template"]["spec"]
    self.assertEqual((worker["backoffLimit"], worker["completions"], worker["parallelism"]), (0, 64, 64))
    self.assertEqual(worker["template"]["spec"]["restartPolicy"], "Never")

  def test_rendered_environment_and_command_match_signed_recipe(self):
    document = _render()
    env = renderer._env(document)
    self.assertEqual(env["MIN_TOKEN_BUCKET"], "4096")
    self.assertEqual(env["CANON_LOGPROB_M"], "256")
    self.assertEqual(env["CANON_VJP2_MAX_SEQS"], "1")
    self.assertEqual(env["CANON_PRE_ALIGN_GATE"], "1")
    self.assertTrue(env["CANON_PRE_ALIGN_REPORT"].endswith("pre_alignment.jsonl"))
    self.assertEqual(env["CANON_P34_WHITELIST_SHA256"], "3" * 64)
    command = env["CANON_RUN_CMD"]
    for value in (
        "--batch_size=8",
        "--num_generations=8",
        "--max_prompt_length=4096",
        "--max_response_length=32768",
        "--max_steps=3",
        "--rollout_mesh_dp=16",
        "--rollout_mesh_tp=8",
        "--train_mesh_dp=16",
        "--train_mesh_tp=8",
        "--rollout_vllm_max_num_seqs=4",
        "--max_num_batched_tokens=256",
        "--train_fraction=1.0",
        "--num_epochs=1",
        "--enable_remat=True",
        "--remat_policy=decoder",
        "--num_iterations=1",
        "--beta=0.0",
        "--epsilon=0.2",
        "--epsilon_high=0.28",
        "--off_policy_steps=0",
        "--per_turn_timeout_secs=300",
        "--episode_timeout_secs=5400",
        "--step_timeout_secs=1800",
        "--reward_timeout_secs=1800",
        "--loss_agg_mode=sequence-mean-token-scale",
        "--advantage_estimator=rloo",
        "--learning_rate=1e-6",
        "--b1=0.9",
        "--b2=0.99",
        "--weight_decay=0.01",
        "--max_grad_norm=1.0",
    ):
      self.assertIn(value, command)
    self.assertNotIn("--rollout_vllm_max_num_seqs=64", command)
    self.assertNotIn("--max_num_batched_tokens=8192", command)
    self.assertNotIn("fsdp", command)

  def test_secret_refs_and_pvc_survive_render(self):
    document = _render()
    head = renderer._head(document)
    main = renderer._container(head["containers"], "jax-tpu")
    secret_env = {
        item["name"]: item["valueFrom"]["secretKeyRef"]
        for item in main["env"]
        if "secretKeyRef" in item.get("valueFrom", {})
    }
    self.assertEqual(secret_env["INJECTED_HF_TOKEN"]["key"], "HF_TOKEN")
    self.assertEqual(secret_env["INJECTED_WANDB_API_KEY"]["key"], "WANDB_API_KEY")
    self.assertIn(
        {"name": "p34-data", "mountPath": "/mnt/disks/linchai_data"},
        main["volumeMounts"],
    )

  def test_floating_image_and_bad_digest_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "pinned"):
      _render(client_image="registry.example/tunix:latest")
    with self.assertRaisesRegex(ValueError, "whitelist_sha256"):
      _render(whitelist_sha256="bad")

  def test_invalid_stage_source_and_run_id_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "unknown P34 stage"):
      _render(stage="skip-gates")
    with self.assertRaisesRegex(ValueError, "source_commit"):
      _render(source_commit="abc")
    with self.assertRaisesRegex(ValueError, "run_id"):
      _render(run_id="UPPER")


if __name__ == "__main__":
  unittest.main()
