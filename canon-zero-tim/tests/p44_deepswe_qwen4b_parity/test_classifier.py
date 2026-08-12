"""Positive and negative controls for the dual-topology P44 classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


classifier = _load(
    "p44_parity_classifier",
    ROOT / "canon-zero-tim/tests/p44_deepswe_qwen4b_parity/classify_run.py",
)
artifacts = _load("p44_parity_artifacts", ROOT / "tunix/rl/deepswe_debug.py")


def _trajectory_batch():
  trajectories = []
  rewards = []
  advantages = []
  for index in range(16):
    group, pair = divmod(index, 4)
    reward = float(pair % 2)
    trajectories.append(types.SimpleNamespace(
        group_id=group,
        pair_index=pair,
        traj={
            "status": "SUCCEEDED",
            "trajectory_reward": reward,
            "conversation_text": [
                {"role": "user", "content": f"p{group}"},
                {"role": "assistant", "content": f"a{pair}"},
            ],
        },
    ))
    rewards.append(reward)
    advantages.append(-1.0 if pair % 2 == 0 else 1.0)
  return trajectories, rewards, advantages


def _spec(topology: str) -> dict[str, int | str]:
  return classifier._TOPOLOGY[topology]


def _log(topology: str, stage: str, batches: int) -> str:
  spec = _spec(topology)
  lines = [
      "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)",
      "[P34.PATHWAYS] initialized_once=1 before_jax=1",
      "[P34.CLI] PASS model=Qwen3-4B prompts=4 generations=4",
      "[sync] provenance ok",
      "[P34.DEVICE_INVENTORY] PASS "
      f"devices={spec['total_devices']} host_source=logical_task "
      f"hosts={spec['hosts']} devices_per_host=4 "
      f"rollout_hosts={spec['role_hosts']} trainer_hosts={spec['role_hosts']}",
      "[P34.TOPOLOGY] PASS",
      "[PATHTRACE] CANON_PALLAS_SWIGLU_MPAD=1 M=4096 Mp=4096 "
      "F=1216 Fp=1280 row_padded=0 feature_padded=1",
      "[CANON_P34_WANDB] ONLINE_RUN_PASS",
      f"Prepared token paddings: [{spec['global_m']}]",
      "Precompile worker0 backbone --> "
      f"{{'num_tokens': {spec['global_m']}, 'num_reqs': 16}}",
  ]
  lines.extend("[P44.TRAJECTORY_BATCH]" for _ in range(batches))
  lines.extend("[P44.BATCH_METRICS_JSON]" for _ in range(batches))
  lines.extend(
      "[P44.LOGPS_BATCH] configured_prompts=4 generations=4 "
      "execution_trajectories=16 observed_trajectories=16"
      for _ in range(batches)
  )
  if stage == "rollout-only":
    lines.append("[P44.ROLLOUT_ONLY] PASS")
  return "\n".join(lines)


def _policy() -> dict[str, str]:
  return {
      "id": "deepswe-pilot-alignment-warning-v1",
      "claim_level": "convergence-only",
  }


def _weight(topology: str) -> dict:
  spec = _spec(topology)
  return {
      "verdict": "PASS",
      "equal": True,
      "mesh_shape": {"dp": spec["dp"], "tp": 8},
      "mesh_device_ids": list(range(spec["devices"])),
  }


def _pre() -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "N_action": 10,
      "admission_policy": _policy(),
  }


def _alignment() -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "ratio_finite": True,
      "gradient": {"finite": True},
      "admission_policy": _policy(),
  }


def _update(topology: str, step: int) -> dict:
  spec = _spec(topology)
  limit = 100 * 1024**3
  free = 10 * 1024**3
  snapshot = [
      {
          "device": index,
          "peak_bytes_in_use": limit - free,
          "bytes_limit": limit,
      }
      for index in range(spec["devices"])
  ]
  return {
      "contract_name": spec["contract"],
      "dp_size": spec["dp"],
      "tp_size": 8,
      "global_m": spec["global_m"],
      "verdict": "PASS",
      "commits": 1,
      "train_steps_before": step,
      "train_steps_after": step + 1,
      "gradient_finite": True,
      "gradient_activity": [True] * spec["local_trajectories"],
      "dp_replicas_exact": True,
      "dp_reduction_transactions": spec["local_trajectories"],
      "dp_reduction_rounds_per_transaction": spec["reduction_rounds"],
      "dp_rank_pullbacks_per_transaction": spec["dp"],
      "optimizer_placement": "device-resident",
      "optimizer_memory_kinds_before": ["device"],
      "optimizer_memory_kinds_after": ["device"],
      "optimizer_transaction_valid": True,
      "hbm_before": snapshot,
      "hbm_after_accumulation": snapshot,
      "hbm_after_commit": snapshot,
  }


class P44ClassifierTest(unittest.TestCase):

  def _artifacts(
      self, root: Path, *, topology: str, stage: str, batches: int
  ) -> None:
    for step in range(batches):
      artifacts.persist_batch(
          *_trajectory_batch(),
          expected_step=step,
          output_dir=root,
          model_id="Qwen/Qwen3-4B",
          values={
              "CANON_P43_DEEPSWE_DEBUG": "0",
              "CANON_P44_DEEPSWE_PARITY": "1",
              "CANON_P44_TOPOLOGY": topology,
              "CANON_EXPECT_COMMIT": "1" * 40,
              "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
              "CANON_RUN_ID": "classify",
              "CANON_P34_RUN_STAGE": stage,
          },
      )

  def _classify(self, root: Path, *, topology: str, stage: str):
    updates_count = classifier._STAGE_UPDATES[stage]
    batches = max(1, updates_count)
    spec = _spec(topology)
    return classifier.classify(
        log_text=_log(topology, stage, batches),
        debug_dir=root,
        weight_attestations=[
            _weight(topology) for _ in range(updates_count)
        ],
        pre_alignment=[_pre() for _ in range(updates_count)],
        alignment=[
            _alignment()
            for _ in range(updates_count * spec["local_trajectories"])
        ],
        updates=[_update(topology, step) for step in range(updates_count)],
        stage=stage,
        topology=topology,
    )

  def test_rollout_and_three_update_pass_on_both_topologies(self):
    for topology in ("64", "256"):
      for stage in ("rollout-only", "three-update"):
        with self.subTest(topology=topology, stage=stage):
          with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text).resolve()
            batches = max(1, classifier._STAGE_UPDATES[stage])
            self._artifacts(
                root, topology=topology, stage=stage, batches=batches
            )
            report = self._classify(root, topology=topology, stage=stage)
            self.assertEqual(report["verdict"], "PASS")

  def test_topology_mismatch_is_rejected(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, topology="64", stage="one-update", batches=1)
      report = self._classify(root, topology="256", stage="one-update")
      self.assertIn("manifest_exact", report["failed"])

  def test_missing_runtime_batch_evidence_is_rejected(self):
    topology = "64"
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, topology=topology, stage="rollout-only", batches=1)
      log = _log(topology, "rollout-only", 1).replace(
          "[P44.LOGPS_BATCH] configured_prompts=4 generations=4 "
          "execution_trajectories=16 observed_trajectories=16",
          "",
      )
      report = classifier.classify(
          log_text=log,
          debug_dir=root,
          weight_attestations=[],
          pre_alignment=[],
          alignment=[],
          updates=[],
          stage="rollout-only",
          topology=topology,
      )
      self.assertIn("logps_batch_exact", report["failed"])

  def test_missing_swiglu_feature_padding_evidence_is_rejected(self):
    topology = "256"
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, topology=topology, stage="rollout-only", batches=1)
      log = _log(topology, "rollout-only", 1).replace(
          "[PATHTRACE] CANON_PALLAS_SWIGLU_MPAD=1 M=4096 Mp=4096 "
          "F=1216 Fp=1280 row_padded=0 feature_padded=1",
          "",
      )
      report = classifier.classify(
          log_text=log,
          debug_dir=root,
          weight_attestations=[],
          pre_alignment=[],
          alignment=[],
          updates=[],
          stage="rollout-only",
          topology=topology,
      )
      self.assertIn("swiglu_feature_padding_active", report["failed"])

  def test_nonmonotonic_update_is_rejected(self):
    topology = "256"
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, topology=topology, stage="three-update", batches=3)
      spec = _spec(topology)
      updates = [_update(topology, step) for step in range(3)]
      updates[2]["train_steps_before"] = 9
      report = classifier.classify(
          log_text=_log(topology, "three-update", 3),
          debug_dir=root,
          weight_attestations=[_weight(topology) for _ in range(3)],
          pre_alignment=[_pre() for _ in range(3)],
          alignment=[
              _alignment() for _ in range(3 * spec["local_trajectories"])
          ],
          updates=updates,
          stage="three-update",
          topology=topology,
      )
      self.assertIn("monotonic_train_steps", report["failed"])


if __name__ == "__main__":
  unittest.main()
