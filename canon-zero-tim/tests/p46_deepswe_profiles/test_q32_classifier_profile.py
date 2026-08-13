"""P46 topology extensions to the existing Qwen3-32B classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = ROOT / "canon-zero-tim/tests/p34_deepswe/classify_run.py"
SPEC = importlib.util.spec_from_file_location("p46_q32_classifier", CLASSIFIER_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import Qwen3-32B classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)

ARTIFACT_SPEC = importlib.util.spec_from_file_location(
    "p46_q32_artifacts", ROOT / "tunix/rl/deepswe_debug.py"
)
if ARTIFACT_SPEC is None or ARTIFACT_SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE artifact writer")
artifacts = importlib.util.module_from_spec(ARTIFACT_SPEC)
sys.modules[ARTIFACT_SPEC.name] = artifacts
ARTIFACT_SPEC.loader.exec_module(artifacts)

CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "p46_q32_contract", ROOT / "tunix/rl/deepswe_contract.py"
)
if CONTRACT_SPEC is None or CONTRACT_SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE workload contract")
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = contract
CONTRACT_SPEC.loader.exec_module(contract)


class P46Q32ClassifierProfileTest(unittest.TestCase):

  def test_active_workload_selects_both_p46_topologies(self):
    small = contract.active_workload({
        "CANON_P46_DEEPSWE_TRAIN": "1",
        "CANON_P46_TOPOLOGY": "64",
    })
    large = contract.active_workload({
        "CANON_P46_DEEPSWE_TRAIN": "1",
        "CANON_P46_TOPOLOGY": "256",
    })
    self.assertIs(small, contract.P46_Q32_64_WORKLOAD)
    self.assertIs(large, contract.P46_Q32_256_WORKLOAD)
    small.validate()
    large.validate()
    self.assertEqual(contract.requested_max_steps({
        "CANON_P46_DEEPSWE_TRAIN": "1",
        "CANON_P46_TOPOLOGY": "64",
        "CANON_P34_RUN_STAGE": "full",
        "CANON_P34_NO_COMMIT": "0",
    }), 1000)

  def test_p46_training_cannot_overlap_a_debug_recipe(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      contract.active_workload({
          "CANON_P46_DEEPSWE_TRAIN": "1",
          "CANON_P46_TOPOLOGY": "64",
          "CANON_P44_DEEPSWE_PARITY": "1",
          "CANON_P44_TOPOLOGY": "64",
      })

  def test_classifier_specs_cover_both_topologies(self):
    small = classifier._profile_spec(topology="64", p46_profile=True)
    large = classifier._profile_spec(topology="256", p46_profile=True)
    self.assertEqual(
        (small["dp"], small["global_m"], small["local_trajectories"]),
        (4, 1024, 16),
    )
    self.assertEqual(
        (large["dp"], large["global_m"], large["local_trajectories"]),
        (16, 4096, 4),
    )
    with self.assertRaisesRegex(ValueError, "fixed to topology 256"):
      classifier._profile_spec(topology="64", p46_profile=False)

  def test_64chip_artifact_manifest_uses_p46_contract(self):
    trajectories = []
    for index in range(64):
      group, pair = divmod(index, 8)
      trajectories.append(types.SimpleNamespace(
          group_id=group,
          pair_index=pair,
          metadata={
              "task_identity": {
                  "instance_id": group,
                  "docker_image": f"repo/image-{group}:latest",
              }
          },
          traj={
              "status": "SUCCEEDED",
              "trajectory_reward": float(pair == 0),
              "conversation_text": [
                  {"role": "user", "content": f"prompt-{group}"},
                  {"role": "assistant", "content": f"answer-{pair}"},
              ],
          },
      ))
    values = {
        "CANON_P34_TRAJECTORY_CAPTURE": "1",
        "CANON_P43_DEEPSWE_DEBUG": "0",
        "CANON_P44_DEEPSWE_PARITY": "0",
        "CANON_DEEPSWE_ONEHOST_SMOKE": "0",
        "CANON_P46_DEEPSWE_TRAIN": "1",
        "CANON_P46_TOPOLOGY": "64",
        "CANON_P34_RUN_STAGE": "full",
        "CANON_EXPECT_COMMIT": "6" * 40,
        "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
        "CANON_RUN_ID": "p46-artifact-64",
        "CANON_P34_DATASET_NAME": "R2E-Gym/R2E-Gym-Subset",
        "CANON_P34_DATASET_REVISION": classifier._DATASET_REVISION,
        "CANON_P34_DATASET_SPLIT": "train",
        "CANON_P34_DATASET_ROWS": "4578",
        "CANON_P34_CLEAN_ROWS": "1851",
        "CANON_P34_WHITELIST_SHA256": classifier._WHITELIST_SHA256,
    }
    with tempfile.TemporaryDirectory() as text:
      root = Path(text).resolve()
      artifacts.persist_batch(
          trajectories,
          [float(index % 8 == 0) for index in range(64)],
          [1.0] * 64,
          expected_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-32B",
          values=values,
      )
      spec = classifier._profile_spec(topology="64", p46_profile=True)
      checks, _ = classifier._artifact_checks(
          root, expected_batches=1, spec=spec
      )
      manifest = json.loads((root / "run_manifest.json").read_text())
    self.assertTrue(all(checks.values()), checks)
    self.assertEqual(manifest["contract_name"], "p46-qwen32b-train-64")
    self.assertEqual(manifest["slice_topology"], "4x4x4")
    self.assertEqual(
        manifest["role_topology"], {"dp": 4, "tp": 8, "devices": 32}
    )


if __name__ == "__main__":
  unittest.main()
