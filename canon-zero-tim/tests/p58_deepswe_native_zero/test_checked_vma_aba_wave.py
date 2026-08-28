#!/usr/bin/env python3
"""Contracts for the three-arm P58 checked-VMA ABA wave."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
SCRIPTS = (
    PKG / "tasks/p58-deepswe-native-zero-comparison/scripts"
)


def _load(name: str, filename: str):
  spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {filename}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


RENDER = _load("p58_checked_vma_aba_render", "render_p58_checked_vma_aba_wave.py")
VERIFY = _load("p58_checked_vma_aba_verify", "verify_p58_checked_vma_aba_wave.py")
CLASSIFY = _load(
    "p58_checked_vma_aba_classify", "classify_p58_checked_vma_aba_wave.py"
)

SOURCE = "1" * 40
IMAGE = "registry.example/tunix@sha256:" + "2" * 64


def _render(root: Path) -> tuple[Path, dict, dict]:
  output = root / "aba"
  receipt = RENDER.render_wave(
      base_path=PKG / "cluster/jobset-64chip.yaml",
      output_dir=output,
      source_commit=SOURCE,
      source_branch="yuxzhang/canon-zero-tim",
      client_image=IMAGE,
      wave_id="p58aba01",
      cpu_nodepool="cpu-np",
      worker_nodepool="tpu-pool",
      model_pvc="model-pvc",
  )
  verification = VERIFY.verify(output)
  return output, receipt, verification


def _arm_result(path: Path, selector: str, *, exact: bool) -> Path:
  result = {
      "schema": "canon.p58.checked-vma-diagnostic.v2",
      "verdict": "PASS",
      "outcome": (
          f"A_B_EXACT_WITH_CHECKED_VMA_{selector.upper()}"
          if exact else
          f"A_B_RED_WITH_CHECKED_VMA_{selector.upper()}"
      ),
      "selector": selector,
      "source_commit": SOURCE,
      "B_C_differing_bytes": 0,
      "backward": 0,
      "optimizer_commits": 0,
  }
  path.write_text(json.dumps(result), encoding="utf-8")
  return path


class P58CheckedVmaAbaWaveTest(unittest.TestCase):

  def test_wave_is_three_independent_matched_zero_commit_jobsets(self):
    with tempfile.TemporaryDirectory() as directory:
      output, receipt, verification = _render(Path(directory))
      self.assertEqual(receipt["arm_order"], ["on-a", "off", "on-b"])
      self.assertEqual(receipt["parallel_capacity"]["aggregate_tpu_chips"], 384)
      self.assertEqual(
          receipt["parallel_capacity"]["aggregate_sandbox_concurrency"], 384
      )
      self.assertEqual(verification["verdict"], "PASS")
      self.assertEqual(len(verification["jobsets"]), 3)
      self.assertEqual(len(verification["persistent_roots"]), 3)
      self.assertEqual(verification["backward"], 0)
      self.assertEqual(verification["optimizer_commits"], 0)

      selectors = []
      for filename in ("01-on-a.yaml", "02-off.yaml", "03-on-b.yaml"):
        document = yaml.safe_load((output / "jobsets" / filename).read_text())
        selectors.append(
            document["metadata"]["labels"][
                "canon.zero-tim/diagnostic-selector"
            ]
        )
      self.assertEqual(selectors, ["on", "off", "on"])

  def test_serialized_yaml_tamper_fails_verification(self):
    with tempfile.TemporaryDirectory() as directory:
      output, _, _ = _render(Path(directory))
      (output / "wave-verify.json").unlink()
      path = output / "jobsets/02-off.yaml"
      document = yaml.safe_load(path.read_text())
      document["metadata"]["labels"]["canon.zero-tim/diagnostic-selector"] = "on"
      path.write_text(yaml.safe_dump(document), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "digest drifted"):
        VERIFY.verify(output)

  def test_aba_classifier_recognizes_causal_reproduction(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      output, _, _ = _render(root)
      result = CLASSIFY.classify(
          wave_verify=output / "wave-verify.json",
          on_a=_arm_result(root / "on-a.json", "on", exact=False),
          off=_arm_result(root / "off.json", "off", exact=True),
          on_b=_arm_result(root / "on-b.json", "on", exact=False),
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["decision"], "CHECKED_VMA_CAUSAL_REPRODUCED")

  def test_aba_classifier_keeps_nonreplicating_on_arm_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      output, _, _ = _render(root)
      result = CLASSIFY.classify(
          wave_verify=output / "wave-verify.json",
          on_a=_arm_result(root / "on-a.json", "on", exact=False),
          off=_arm_result(root / "off.json", "off", exact=True),
          on_b=_arm_result(root / "on-b.json", "on", exact=True),
      )
      self.assertEqual(result["verdict"], "INCONCLUSIVE")
      self.assertEqual(result["decision"], "INCONCLUSIVE_ON_REPLICATION")


if __name__ == "__main__":
  unittest.main()
