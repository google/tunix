#!/usr/bin/env python3
"""Host tests for the immutable e0w5 recovery render contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from validate_m15_e0w5_recovery_render import (
    RenderContractError,
    RUN_ID,
    TARGET_SOURCE,
    validate,
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class E0w5RecoveryRenderTest(unittest.TestCase):

  def setUp(self) -> None:
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.rows = []
    for arm in ("off", "on"):
      name = f"canon-v1-apc-m15-{arm}-{RUN_ID}-{TARGET_SOURCE[:8]}"
      path = self.root / f"jobset-v1-apc-m15-{arm}-{RUN_ID}.yaml"
      document = {
          "apiVersion": "jobset.x-k8s.io/v1alpha2",
          "kind": "JobSet",
          "metadata": {
              "name": name,
              "labels": {
                  "canon.zero-tim/apc-m15-arm": arm,
                  "canon.zero-tim/m15-token-continuity": "exact",
              },
          },
          "spec": {
              "replicatedJobs": [{
                  "template": {"spec": {"template": {"spec": {
                      "containers": [{
                          "name": "jax-tpu",
                          "env": [
                              {"name": "CANON_APC_M15_TARGET_DEBUG", "value": arm},
                              {"name": "CANON_EXPECT_COMMIT", "value": TARGET_SOURCE},
                              {"name": "CANON_M15_TOKEN_CONTINUITY", "value": "exact"},
                              {"name": "CANON_P38_DIAGNOSTIC_ROUNDS", "value": "3"},
                              {"name": "CANON_P38_DURABILITY_PROFILE", "value": "m15-wide-v1"},
                              {"name": "CANON_P38_SEAM_OBSERVER", "value": "layer"},
                              {"name": "CANON_P38_TAIL_OBSERVER", "value": "1"},
                              {"name": "CANON_P38_PRECHECK_ONLY", "value": "1"},
                              {"name": "CANON_P38_CONTROLLED_EXIT", "value": "1"},
                              {"name": "CANON_P33_NO_COMMIT", "value": "1"},
                              {"name": "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY", "value": "0"},
                              {"name": "CANON_VLLM_ENABLE_PREFIX_CACHING", "value": "1" if arm == "on" else "0"},
                              {"name": "CANON_P38_GCS_PREFIX", "value": f"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{name}/attempt-0"},
                          ],
                      }],
                  }}}},
              }],
          },
      }
      path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
      self.rows.append({
          "arm": arm,
          "jobset": name,
          "yaml": path.name,
          "sha256": _sha256(path),
      })
    contract = {
        "schema": "m15-e0v-tito-layer-render-v1",
        "source_commit": TARGET_SOURCE,
        "run_id": RUN_ID,
        "program_identity": "m15-apc-debug-exact-tito-layer-v1",
        "observer": "layer",
        "rounds": 3,
        "zero_backward": True,
        "zero_optimizer_commit": True,
        "b_full_reset_immutable": True,
        "control_and_treatment_differ_only_at_apc": True,
        "tito_exact_both_arms": True,
        "launch_authorized": False,
        "target_executed": False,
        "remote_mutation": False,
        "arms": self.rows,
    }
    (self.root / "RUN_CONTRACT.json").write_text(
        json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
    )
    self._manifest()

  def tearDown(self) -> None:
    self.holder.cleanup()

  def _manifest(self) -> None:
    names = sorted(
        path.name for path in self.root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (self.root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(self.root / name)}  {name}\n" for name in names),
        encoding="ascii",
    )

  def _document(self, arm: str) -> tuple[Path, dict]:
    path = self.root / f"jobset-v1-apc-m15-{arm}-{RUN_ID}.yaml"
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))

  def test_accepts_exact_original_pair(self) -> None:
    result = validate(self.root)
    self.assertEqual(result["source_commit"], TARGET_SOURCE)
    self.assertEqual(set(result["arms"]), {"off", "on"})
    wrapper = Path(__file__).with_name("run_m15_e0w5_gcs_return.sh").read_text(
        encoding="utf-8"
    )
    self.assertIn("run_m15_multiround_gcs_return.sh", wrapper)
    self.assertIn('"$scratch_parent" 1 >"$raw_log"', wrapper)
    self.assertNotIn("kubectl", wrapper)
    self.assertNotRegex(wrapper, r"gcloud storage (cp|rsync).+gs://")
    self.assertNotRegex(wrapper, r"gsutil.+-[mM]?")

  def test_rejects_manifest_tamper(self) -> None:
    path, _ = self._document("off")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with self.assertRaisesRegex(RenderContractError, "hash drifted"):
      validate(self.root)

  def test_rejects_run_identity_drift(self) -> None:
    path = self.root / "RUN_CONTRACT.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["run_id"] = "e0w6"
    path.write_text(json.dumps(value), encoding="utf-8")
    self._manifest()
    with self.assertRaisesRegex(RenderContractError, "run contract drifted"):
      validate(self.root)

  def test_rejects_source_or_jobset_drift(self) -> None:
    path, document = self._document("on")
    document["metadata"]["name"] = "canon-v1-apc-m15-on-reused"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    self._manifest()
    with self.assertRaisesRegex(RenderContractError, "JobSet identity drifted"):
      validate(self.root)

  def test_rejects_difference_beyond_apc(self) -> None:
    path, document = self._document("on")
    document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]["containers"][0]["env"].append({"name": "UNSIGNED_DRIFT", "value": "1"})
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    self.rows[1]["sha256"] = _sha256(path)
    contract_path = self.root / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["arms"] = self.rows
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    self._manifest()
    with self.assertRaisesRegex(RenderContractError, "beyond the signed APC"):
      validate(self.root)


if __name__ == "__main__":
  unittest.main()
