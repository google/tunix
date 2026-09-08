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


def _log(
    topology: str,
    stage: str,
    batches: int,
    system_optimization_arm: str | None = None,
) -> str:
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
      "[PATHTRACE] CANON_PALLAS_MPAD=1 M=4096 Mp=4096 padded=0 "
      "K=2560 Kp=2560 N=1216 Np=1280 "
      "contract_padded=0 output_padded=1",
      "[PATHTRACE] CANON_PALLAS_MPAD=1 M=4096 Mp=4096 padded=0 "
      "K=1216 Kp=1280 N=2560 Np=2560 "
      "contract_padded=1 output_padded=0",
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
  if system_optimization_arm is not None:
    lines.append(
        "[P44.V2] system optimization "
        f"arm={system_optimization_arm} topology={topology} strict=1"
    )
    lines.extend(
        "[P59.CHECKED_VMA] enabled=1" for _ in range(3)
    )
    lines.extend("[V1.FIRST_UPDATE] {}" for _ in range(2))
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


def _exact_boundary() -> dict:
  return {
      "valid": True,
      "finite": True,
      "differing_bytes": 0,
      "differing_elements": 0,
  }


def _pre(system_optimization_arm: str | None = None) -> dict:
  if system_optimization_arm is not None:
    return {
        "verdict": "PASS",
        "reds": [],
        "blocking_reds": [],
        "warning_reds": [],
        "reported_reds": [],
        "N_action": 10,
        "admission_policy": {
            "enabled": False,
            "warning_only": False,
            "claim_level": "strict-zero-tim",
        },
        "boundaries": {
            "S_decode_vs_S_prefill": _exact_boundary(),
            "S_prefill_vs_T_old": _exact_boundary(),
        },
    }
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "N_action": 10,
      "admission_policy": _policy(),
  }


def _alignment(system_optimization_arm: str | None = None) -> dict:
  if system_optimization_arm is not None:
    return {
        "verdict": "PASS",
        "reds": [],
        "blocking_reds": [],
        "warning_reds": [],
        "reported_reds": [],
        "admission_policy": {
            "enabled": False,
            "warning_only": False,
            "claim_level": "strict-zero-tim",
        },
        "boundaries": {
            name: _exact_boundary()
            for name in (
                "S_decode_vs_S_prefill",
                "S_prefill_vs_T_old",
                "T_old_vs_T_current",
            )
        },
        "exact": {
            "w_all_exactly_1": True,
            "r_all_exactly_1": True,
            "wr_all_exactly_1": True,
        },
        "ratio_finite": True,
        "clip_hits": 0,
        "tis_hits": 0,
        "gradient": {"finite": True},
    }
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "ratio_finite": True,
      "gradient": {"finite": True},
      "admission_policy": _policy(),
  }


def _update(
    topology: str,
    step: int,
    system_optimization_arm: str | None = None,
) -> dict:
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
  record = {
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
  if system_optimization_arm is not None:
    treatment = system_optimization_arm == "treatment"
    record.update({
        "system_optimization_arm": system_optimization_arm,
        "dp_reduction_transactions": (
            1 if treatment else spec["local_trajectories"]
        ),
        "dp_reduction_visibility": (
            "EXPLICIT_FIXED_TREE_REDUCE_ONCE"
            if treatment
            else "EXPLICIT_FIXED_TREE"
        ),
        "dp_staged_accumulations": (
            spec["local_trajectories"] if treatment else 0
        ),
        "commit_gradient_norm": 2.0 + step,
    })
  return record


class P44ClassifierTest(unittest.TestCase):

  def _artifacts(
      self,
      root: Path,
      *,
      topology: str,
      stage: str,
      batches: int,
      system_optimization_arm: str | None = None,
  ) -> None:
    for step in range(batches):
      artifacts.persist_batch(
          *_trajectory_batch(),
          expected_step=step,
          output_dir=root,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          values={
              "CANON_P43_DEEPSWE_DEBUG": "0",
              "CANON_P44_DEEPSWE_PARITY": "1",
              "CANON_P44_TOPOLOGY": topology,
              "CANON_EXPECT_COMMIT": "1" * 40,
              "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
              "CANON_RUN_ID": "classify",
              "CANON_P34_RUN_STAGE": stage,
              **(
                  {"CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM": system_optimization_arm}
                  if system_optimization_arm is not None
                  else {}
              ),
          },
      )

  def _classify(
      self,
      root: Path,
      *,
      topology: str,
      stage: str,
      system_optimization_arm: str | None = None,
  ):
    updates_count = classifier._STAGE_UPDATES[stage]
    batches = max(1, updates_count)
    spec = _spec(topology)
    return classifier.classify(
        log_text=_log(topology, stage, batches, system_optimization_arm),
        debug_dir=root,
        weight_attestations=[
            _weight(topology) for _ in range(updates_count)
        ],
        pre_alignment=[
            _pre(system_optimization_arm) for _ in range(updates_count)
        ],
        alignment=[
            _alignment(system_optimization_arm)
            for _ in range(updates_count * spec["local_trajectories"])
        ],
        updates=[
            _update(topology, step, system_optimization_arm)
            for step in range(updates_count)
        ],
        stage=stage,
        topology=topology,
        system_optimization_arm=system_optimization_arm,
    )

  def test_rollout_and_three_update_pass_on_both_topologies(self):
    for topology in ("64", "128"):
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
      report = self._classify(root, topology="128", stage="one-update")
      self.assertIn("manifest_exact", report["failed"])

  def test_strict_system_optimization_arms_pass_both_topologies(self):
    for topology in ("64", "128"):
      for arm in ("control", "treatment"):
        with self.subTest(topology=topology, arm=arm):
          with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text).resolve()
            self._artifacts(
                root,
                topology=topology,
                stage="three-update",
                batches=3,
                system_optimization_arm=arm,
            )
            report = self._classify(
                root,
                topology=topology,
                stage="three-update",
                system_optimization_arm=arm,
            )
            self.assertEqual(report["verdict"], "PASS", report)
            self.assertEqual(
                report["claim_level"],
                "strict-zero-tim-system-optimization-arm",
            )

  def test_strict_system_optimization_negatives_are_audible(self):
    topology = "64"
    arm = "treatment"
    spec = _spec(topology)
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(
          root,
          topology=topology,
          stage="three-update",
          batches=3,
          system_optimization_arm=arm,
      )
      pre = [_pre(arm) for _ in range(3)]
      post = [_alignment(arm) for _ in range(3 * spec["local_trajectories"])]
      updates = [_update(topology, step, arm) for step in range(3)]
      cases = (
          (
              "warning_verdict",
              "pre_alignment_nonblocking",
          ),
          (
              "one_differing_byte",
              "alignment_nonblocking",
          ),
          (
              "wrong_reduce_count",
              "fixed_dp_transaction",
          ),
      )
      for name, failed_check in cases:
        with self.subTest(name=name):
          local_pre = [dict(record) for record in pre]
          local_post = [dict(record) for record in post]
          local_updates = [dict(record) for record in updates]
          if name == "warning_verdict":
            local_pre[0]["verdict"] = "PASS_WITH_ALIGNMENT_WARNINGS"
          elif name == "one_differing_byte":
            local_post[0] = {
                **local_post[0],
                "boundaries": {
                    **local_post[0]["boundaries"],
                    "T_old_vs_T_current": {
                        **local_post[0]["boundaries"]["T_old_vs_T_current"],
                        "differing_bytes": 1,
                        "differing_elements": 1,
                    },
                },
            }
          else:
            local_updates[0]["dp_reduction_transactions"] = 2
          report = classifier.classify(
              log_text=_log(topology, "three-update", 3, arm),
              debug_dir=root,
              weight_attestations=[_weight(topology) for _ in range(3)],
              pre_alignment=local_pre,
              alignment=local_post,
              updates=local_updates,
              stage="three-update",
              topology=topology,
              system_optimization_arm=arm,
          )
          self.assertIn(failed_check, report["failed"])

      missing_checked = classifier.classify(
          log_text=_log(topology, "three-update", 3, arm).replace(
              "[P59.CHECKED_VMA] enabled=1", "", 1
          ),
          debug_dir=root,
          weight_attestations=[_weight(topology) for _ in range(3)],
          pre_alignment=pre,
          alignment=post,
          updates=updates,
          stage="three-update",
          topology=topology,
          system_optimization_arm=arm,
      )
      self.assertIn("system_optimization_receipts", missing_checked["failed"])

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
    topology = "128"
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

  def test_missing_matmul_padding_evidence_is_rejected(self):
    topology = "128"
    markers = {
        "matmul_output_padding_active": (
            "[PATHTRACE] CANON_PALLAS_MPAD=1 M=4096 Mp=4096 padded=0 "
            "K=2560 Kp=2560 N=1216 Np=1280 "
            "contract_padded=0 output_padded=1"
        ),
        "matmul_contract_padding_active": (
            "[PATHTRACE] CANON_PALLAS_MPAD=1 M=4096 Mp=4096 padded=0 "
            "K=1216 Kp=1280 N=2560 Np=2560 "
            "contract_padded=1 output_padded=0"
        ),
    }
    for check, marker in markers.items():
      with self.subTest(check=check), tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text).resolve()
        self._artifacts(root, topology=topology, stage="rollout-only", batches=1)
        report = classifier.classify(
            log_text=_log(topology, "rollout-only", 1).replace(marker, ""),
            debug_dir=root,
            weight_attestations=[],
            pre_alignment=[],
            alignment=[],
            updates=[],
            stage="rollout-only",
            topology=topology,
        )
        self.assertIn(check, report["failed"])

  def test_nonmonotonic_update_is_rejected(self):
    topology = "128"
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
