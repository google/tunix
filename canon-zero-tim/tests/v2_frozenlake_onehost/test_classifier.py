"""Tests for the FrozenLake DP2xTP2 one-host classifier and launch contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts"
CLASSIFIER_PATH = SCRIPT_DIR / "classify_frozenlake_dp2tp2.py"
RUNNER_PATH = SCRIPT_DIR / "run_frozenlake_dp2tp2_onehost.sh"
INNER_PATH = SCRIPT_DIR / "run_frozenlake_dp2tp2_inner.sh"
SPEC = importlib.util.spec_from_file_location(
    "v2_frozenlake_onehost_classifier", CLASSIFIER_PATH
)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _boundary() -> dict:
  return {
      "valid": True,
      "finite": True,
      "differing_bytes": 0,
      "differing_elements": 0,
      "max_abs": 0.0,
  }


def _alignment(*, pre: bool, action_tokens: int = 5000) -> dict:
  boundaries = {"T_old_vs_T_current": _boundary()}
  if pre:
    boundaries = {
        "S_decode_vs_S_prefill": _boundary(),
        "S_prefill_vs_T_old": _boundary(),
    }
  return {
      "N_action": action_tokens,
      "verdict": "PASS",
      "blocking_reds": [],
      "reds": [],
      "boundaries": boundaries,
      "hashes": {
          "tokens": "1" * 64,
          "action_mask": "2" * 64,
          "S_decode": "3" * 64,
          "S_prefill": "4" * 64,
          "T_old": "4" * 64,
      },
  }


def _manifest(
    workload: str = "m15", arm: str = "r3", mode: str = "measure"
) -> dict:
  spec = classifier._WORKLOADS[workload]  # pylint: disable=protected-access
  return {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "source_commit": "a" * 40,
      "source_diff_sha256": "b" * 64,
      "image_id_sha256": "c" * 64,
      "model_snapshot_sha256": "d" * 64,
      "dataset_train_sha256": "e" * 64,
      "dataset_test_sha256": "f" * 64,
      "runner_sha256": "0" * 64,
      "workload": workload,
      "workload_name": spec["name"],
      "arm": arm,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp2",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "global_prompts": 4,
      "num_generations": 4,
      "global_trajectories": 16,
      "gradient_groups": 8,
      "local_m": 256,
      "global_m": 512,
      "max_prompt_length": spec["prompt"],
      "max_response_length": spec["response"],
      "max_turns": spec["max_turns"],
      "data_shuffle_seed": 42,
      "vllm_global_seed": 0,
      "vllm_hbm_utilization": spec["vllm_hbm_utilization"],
      "selectors": classifier._ARMS[arm],  # pylint: disable=protected-access
      "reducer_schedule": {
          "kind": "fixed-local-byte-buckets",
          "max_local_bytes": 2 * 1024**3,
      },
      "checked_vma": True,
      "wandb_mode": "disabled",
      "classification_mode": mode,
      "training_capsule": {
          "mode": "none",
          "capture_run": None,
          "sha256": None,
          "model_binding_sha256": None,
      },
      "hbm_stage_diagnostic": mode == "measure" and arm in ("r0", "r0b"),
  }


def _update(arm: str = "r3", workload: str = "m15") -> dict:
  norms = [float(index + 1) for index in range(8)]
  update = {
      "contract_name": f"frozenlake-{workload}-onehost-dp2-tp2",
      "dp_size": 2,
      "tp_size": 2,
      "global_m": 512,
      "verdict": "PASS",
      "mode": "backward-no-commit",
      "microsteps": 8,
      "commits": 0,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "gradient_finite": True,
      "gradient_activity": [True] * 8,
      "micro_gradient_norms": norms,
      "dp_replicas_exact": True,
      "dp_axis": "dp",
      "dp_reduction_transactions": 1 if arm in ("r2", "r3") else 8,
      "dp_reduction_rounds_per_transaction": 2,
      "dp_rank_pullbacks_per_transaction": 2,
      "dp_pullback_invocations_per_transaction": 1,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "optimizer_placement": "pinned-host-offload",
      "optimizer_memory_kinds_before": ["pinned_host"],
      "hbm_before": [
          {"peak_bytes_in_use": 60, "bytes_limit": 100} for _ in range(4)
      ],
      "hbm_after_reverse": [
          {"peak_bytes_in_use": 80, "bytes_limit": 100} for _ in range(4)
      ],
  }
  if arm in ("r2", "r3"):
    update["update_gradient_norm"] = 12.5
  return update


def _retarget_r1_fixture(root: Path, *, workload: str, geometry: str) -> None:
  """Retargets the complete DP2 R1 fixture to another registered geometry."""
  geometry_spec = classifier._GEOMETRIES[geometry]  # pylint: disable=protected-access
  dp_size = geometry_spec["dp"]
  tp_size = geometry_spec["tp"]
  groups = geometry_spec["gradient_groups"]
  manifest_path = root / "run_manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest.update({
      "workload_name": f"frozenlake-{workload}-onehost-dp{dp_size}-tp{tp_size}",
      "model_dir_name": geometry_spec["model_dir_name"],
      "topology": {"dp": dp_size, "tp": tp_size, "devices": 4},
      "gradient_groups": groups,
      "global_m": geometry_spec["global_m"],
      "vllm_hbm_utilization": geometry_spec["vllm_hbm_utilization"][workload],
      "checked_vma": geometry_spec["checked_vma"],
  })
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  alignment_path = root / "alignment.jsonl"
  alignment_path.write_text(
      "".join(
          json.dumps(_alignment(pre=False)) + "\n" for _ in range(groups)
      ),
      encoding="utf-8",
  )
  update_path = root / "updates.json"
  update = json.loads(update_path.read_text(encoding="utf-8"))
  update.update({
      "contract_name": manifest["workload_name"],
      "dp_size": dp_size,
      "tp_size": tp_size,
      "global_m": geometry_spec["global_m"],
      "microsteps": groups,
      "gradient_activity": [True] * groups,
      "micro_gradient_norms": [float(index + 1) for index in range(groups)],
      "dp_reduction_transactions": 0 if dp_size == 1 else groups,
      "dp_reduction_rounds_per_transaction": 0 if dp_size == 1 else 2,
      "dp_rank_pullbacks_per_transaction": dp_size,
  })
  update_path.write_text(json.dumps(update), encoding="utf-8")

  raw_path = root / "raw.log"
  raw_lines = [
      line
      for line in raw_path.read_text(encoding="utf-8").splitlines()
      if not line.startswith(("[P32.DP", "[P59.DP", "[P66.VMA]"))
  ]
  n_real = [3900 + index * 30 for index in range(16)]
  forward_lines = []
  for group in range(groups):
    start = group * dp_size
    values = ", ".join(str(value) for value in n_real[start:start + dp_size])
    forward_lines.append(
        f"[P32.DP{dp_size}] forward_group_issued "
        f"group={group + 1}/{groups} rows=({start}, {start + dp_size}) "
        f"n_real=({values})"
    )
  forward_lines.append(
      f"[P59.DP{dp_size}] reducer_bucket_schedule programs=8 "
      "max_local_bytes=2147483648 peak_local_bytes=2130875392 "
      "total_local_bytes=16382087168"
  )
  if geometry_spec["checked_vma"]:
    forward_lines.append("[P66.VMA] outer_check_enabled program=test")
  raw_path.write_text(
      "\n".join(forward_lines + raw_lines) + "\n", encoding="utf-8"
  )


class FrozenLakeOneHostClassifierTest(unittest.TestCase):

  def _fixture(
      self,
      root: Path,
      *,
      arm: str = "r3",
      mode: str = "measure",
      workload: str = "m15",
  ) -> Path:
    (root / "run_manifest.json").write_text(
        json.dumps(_manifest(workload=workload, arm=arm, mode=mode)),
        encoding="utf-8",
    )
    (root / "runtime.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    (root / "pre_alignment.jsonl").write_text(
        json.dumps(_alignment(pre=True)) + "\n", encoding="utf-8"
    )
    (root / "alignment.jsonl").write_text(
        "".join(json.dumps(_alignment(pre=False)) + "\n" for _ in range(8)),
        encoding="utf-8",
    )
    update = _update(arm, workload)
    diagnostic = mode == "measure" and arm in ("r0", "r0b")
    hbm_stages = []
    if diagnostic:
      # The landed fixture below first crosses into 17 chunks in zero-based
      # group 3; later ties must not move R0's diagnostic group. R0b instead
      # observes r21's independently proven first warm peak group 1.
      hbm_group = 1 if arm == "r0b" else 3
      hbm_chunks = 16 if arm == "r0b" else 17
      for stage_index, stage in enumerate(  # pylint: disable=protected-access
          classifier._expected_hbm_stages(hbm_chunks)
      ):
        hbm_stages.append({
            "stage": stage,
            "group": hbm_group,
            "devices": [
                {
                    "device": device,
                    "bytes_in_use": 60 + stage_index,
                    "peak_bytes_in_use": 70 + stage_index,
                    "bytes_limit": 1000,
                }
                for device in range(4)
            ],
        })
      if arm == "r0b":
        def group_checkpoint(group):
          return {
              "stage": "report_group_after_bucket_0",
              "group": group,
              "devices": [
                  {
                      "device": device,
                      "bytes_in_use": 500 + group,
                      "peak_bytes_in_use": 900 + group,
                      "bytes_limit": 2000,
                  }
                  for device in range(4)
              ],
          }

        def bucket_stages(group, start):
          records = []
          for bucket in range(8):
            for stage_index, stage in enumerate(
                ("before_execute", "after_execute", "after_delete")
            ):
              current = start + bucket * 3 + stage_index
              if stage == "after_delete":
                current -= 1
              records.append({
                  "stage": f"report_bucket_{bucket}_{stage}",
                  "group": group,
                  "devices": [
                      {
                          "device": device,
                          "bytes_in_use": current,
                          "peak_bytes_in_use": 700 + start + bucket * 3
                          + stage_index,
                          "bytes_limit": 2000,
                      }
                      for device in range(4)
                  ],
              })
              if bucket == 0 and stage == "after_execute":
                records.append(group_checkpoint(group))
          return records

        after_model = next(
            index
            for index, item in enumerate(hbm_stages)
            if item["stage"] == "after_model_backward"
        )
        hbm_stages = (
            bucket_stages(0, 100)
            + hbm_stages[: after_model + 1]
            + bucket_stages(1, 300)
            + hbm_stages[after_model + 1 :]
            + [group_checkpoint(group) for group in range(2, 8)]
        )
      update["hbm_stage_receipts"] = hbm_stages
    (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
    marker = (
        "forward_group_done"
        if arm in ("r0", "r0b", "r0c", "r0d")
        else "forward_group_issued"
    )
    lines = [
        f"[P32.DP2] {marker} group={index}/8 rows=(0, 8) "
        f"n_real=({3900 + index * 30}, {4000 + index * 30})"
        for index in range(1, 9)
    ]
    lines.append("[P66.VMA] outer_check_enabled program=test")
    lines.append(
        "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
        "vllm_global_seed=0 per_request_seed=unsupported"
    )
    lines.append("[V2.FL.WANDB] DISABLED_LOCAL_PASS test")
    lines.append(
        "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
        "use_rollout_logps=1 tis_weights=absent"
    )
    lines.append(
        "[P59.DP2] reducer_bucket_schedule programs=8 "
        "max_local_bytes=2147483648 peak_local_bytes=2130875392 "
        "total_local_bytes=16382087168"
    )
    if arm in ("r2", "r3"):
      lines.append(
          "[V2.REDUCE_ONCE.ACCUMULATOR_LOAN] enabled=1 "
          "leaves=399 local_bytes=16382087168 "
          "transition=base-to-staged base_handles_retired=399 "
          "staged_handles_retired=399 check_vma=1 host_transfers=0"
      )
      lines.append(
          "[V2.REDUCE_ONCE.ACCUMULATOR_RESET] enabled=1 "
          "leaves=399 transition=adopted-to-idle "
          "alias_mode=bitwise-zero host_transfers=0"
      )
      lines.extend(
          "[V2.REDUCE_ONCE.REPORT_ACCUMULATE] enabled=1 "
          f"group={group}/8 leaves=399 buckets=8 executables=1 "
          "peak_local_bytes=2130875392 device_dependencies=7 "
          "host_transfers=0"
          for group in range(2, 9)
      )
    lines.extend(
        "[V2.FL.HBM_STAGE] "
        + json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in hbm_stages
    )
    if diagnostic and arm == "r0":
      lines.append(
          "[V2.FL.REPORT_ADJOINT_MEMORY] "
          + json.dumps(
              {
                  "schema": "canon-v2-p59-report-adjoint-memory-v1",
                  "staged_engine_leaves": 399,
                  "trainer_leaves": 399,
                  "argument_size_in_bytes": 25_000_000_000,
                  "output_size_in_bytes": 16_382_087_168,
                  "alias_size_in_bytes": 0,
                  "temp_size_in_bytes": 1_900_000_000,
                  "host_argument_size_in_bytes": 0,
                  "host_output_size_in_bytes": 0,
                  "host_alias_size_in_bytes": 0,
                  "host_temp_size_in_bytes": 0,
              },
              sort_keys=True,
              separators=(",", ":"),
          )
      )
    elif arm in ("r0b", "r0c", "r0d"):
      bucket_template = {
          "source_leaves": 50,
          "target_leaves": 50,
          "local_output_bytes": 2_047_760_896,
          "argument_size_in_bytes": 1_100_000_000,
          "output_size_in_bytes": 2_047_760_896,
          "alias_size_in_bytes": 0,
          "temp_size_in_bytes": 0,
          "host_argument_size_in_bytes": 0,
          "host_output_size_in_bytes": 0,
          "host_alias_size_in_bytes": 0,
          "host_temp_size_in_bytes": 0,
      }
      buckets = [
          {**bucket_template, "bucket": index}
          for index in range(8)
      ]
      buckets[-1]["source_leaves"] = 49
      buckets[-1]["target_leaves"] = 49
      buckets[-1]["local_output_bytes"] = 2_047_760_896
      lines.extend(
          "[P75.REPORT_ADJOINT_BUCKETS] enabled=1 programs=8 "
          "max_local_bytes=2147483648 peak_local_bytes=2047760896 "
          "total_local_bytes=16382087168 host_blocks=8"
          for _ in range(8)
      )
      if diagnostic:
        lines.append(
            "[V2.FL.REPORT_ADJOINT_MEMORY] "
            + json.dumps(
                {
                    "schema": "canon-v2-p75-report-adjoint-buckets-memory-v1",
                    "source_leaves": 399,
                    "target_leaves": 399,
                    "bucket_count": 8,
                    "max_local_bytes": 2 * 1024**3,
                    "peak_local_output_bytes": max(
                        item["local_output_bytes"] for item in buckets
                    ),
                    "total_local_output_bytes": sum(
                        item["local_output_bytes"] for item in buckets
                    ),
                    "host_blocks": 8,
                    "buckets": buckets,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    if arm == "r0c":
      lines.append(
          "[P76.CHUNK_DEPENDENCY] enabled=1 leaves=310 "
          "checked_vma=1 scalar_collectives=1 host_transfers=0"
      )
    if arm == "r0d":
      for chunks in (16, 16, 16, 17, 17, 17, 17, 17):
        lines.append(
            "[P77.CHUNK_BACKPRESSURE] enabled=1 "
            f"pullback_waits={chunks} accumulation_waits={chunks} "
            "leaves=310 "
            "wait_api=block_until_ready host_transfers=0"
        )
    (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    anchor = root / "anchors.json"
    anchor.write_text(
        json.dumps({
            "schema": "canon.v2-frozenlake-onehost.gradient-anchors.v2",
            "anchors": {},
        }),
        encoding="utf-8",
    )
    return anchor

  def _classify(
      self,
      mutate=None,
      *,
      anchor=False,
      anchor_run_id=None,
      arm="r3",
      workload="m15",
      require_anchor=False,
  ) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(
          root,
          arm=arm,
          mode="certify" if require_anchor else "measure",
          workload=workload,
      )
      if mutate is not None:
        mutate(root)
      if anchor:
        registry = json.loads(anchor_path.read_text(encoding="utf-8"))
        update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        key = f"{workload}:dp2-tp2:{arm}"
        registry["anchors"][key] = {
            "run_id": anchor_run_id or (
                manifest.get("training_capsule", {}).get("capture_run")
                if require_anchor
                else "measurement-run-r1"
            ),
            "micro_gradient_norms": update["micro_gradient_norms"],
        }
        if arm in ("r2", "r3"):
          registry["anchors"][key][
              "update_gradient_norm"
          ] = update["update_gradient_norm"]
        if require_anchor:
          registry["anchors"][key][
              "training_capsule_sha256"
          ] = manifest.get("training_capsule", {}).get("sha256")
        anchor_path.write_text(json.dumps(registry), encoding="utf-8")
      return classifier.classify(
          root,
          workload=workload,
          arm=arm,
          docker_exit=0,
          anchor_registry=anchor_path,
          require_anchor=require_anchor,
      )

  def test_clean_unregistered_run_is_measurement_only(self):
    result = self._classify()
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(result["landed_shape"]["max_n_real"], 4240)
    self.assertFalse(result["landed_shape"]["cap_coverage"])
    self.assertEqual(result["receipts"]["p66_outer_check_enabled"], 1)
    self.assertEqual(
        result["receipts"]["reducer_bucket_schedule"]["programs"], 8
    )
    self.assertEqual(
        result["receipts"]["reduce_once_report_accumulate"],
        [
            (group, 8, 399, 8, 1, 2130875392, 7, 0)
            for group in range(2, 9)
        ],
    )
    self.assertEqual(
        result["receipts"]["reduce_once_accumulator_loan"],
        [(399, 16382087168, "base-to-staged", 399, 399, 1, 0)],
    )
    self.assertEqual(
        result["receipts"]["reduce_once_accumulator_reset"],
        [(399, "adopted-to-idle", "bitwise-zero", 0)],
    )
    self.assertEqual(result["gradient"]["update_gradient_norm"], 12.5)

  def test_r1_matrix_geometries_enforce_vma_and_geometry_scoped_anchors(self):
    for geometry, expected_vma in (("dp4-tp1", 0), ("dp1-tp4", 1)):
      with self.subTest(geometry=geometry):
        with tempfile.TemporaryDirectory() as temporary:
          root = Path(temporary)
          anchor_path = self._fixture(root, arm="r1", workload="p45")
          _retarget_r1_fixture(root, workload="p45", geometry=geometry)
          update = json.loads((root / "updates.json").read_text())
          registry = json.loads(anchor_path.read_text())
          registry["anchors"][f"p45:{geometry}:r1"] = {
              "run_id": f"{geometry}-measurement-r1",
              "micro_gradient_norms": update["micro_gradient_norms"],
          }
          anchor_path.write_text(json.dumps(registry), encoding="utf-8")
          result = classifier.classify(
              root,
              workload="p45",
              geometry=geometry,
              arm="r1",
              docker_exit=0,
              anchor_registry=anchor_path,
          )
        self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
        self.assertEqual(result["geometry"], geometry)
        self.assertEqual(
            result["receipts"]["p66_outer_check_enabled"], expected_vma
        )
        self.assertTrue(result["gradient"]["anchor_exact"])

  def test_matrix_geometry_negatives_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "DP1 has no reduce-once arm"):
      with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        anchor_path = self._fixture(root, arm="r2", workload="p45")
        classifier.classify(
            root,
            workload="p45",
            geometry="dp1-tp4",
            arm="r2",
            docker_exit=0,
            anchor_registry=anchor_path,
        )

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(root, arm="r1", workload="p45")
      _retarget_r1_fixture(root, workload="p45", geometry="dp4-tp1")
      manifest_path = root / "run_manifest.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["checked_vma"] = True
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      rejected = classifier.classify(
          root,
          workload="p45",
          geometry="dp4-tp1",
          arm="r1",
          docker_exit=0,
          anchor_registry=anchor_path,
      )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(
        any(reason.startswith("manifest:") for reason in rejected["reasons"])
    )

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(root, arm="r1", workload="p45")
      _retarget_r1_fixture(root, workload="p45", geometry="dp4-tp1")
      manifest_path = root / "run_manifest.json"
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      manifest["vllm_hbm_utilization"] = 0.20
      manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
      rejected = classifier.classify(
          root,
          workload="p45",
          geometry="dp4-tp1",
          arm="r1",
          docker_exit=0,
          anchor_registry=anchor_path,
      )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(
        any(reason.startswith("manifest:") for reason in rejected["reasons"])
    )

  def test_reduce_once_report_accumulate_receipt_is_fail_closed(self):
    def missing(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      lines = raw.splitlines()
      lines.remove(
          "[V2.REDUCE_ONCE.REPORT_ACCUMULATE] enabled=1 "
          "group=4/8 leaves=399 buckets=8 executables=1 "
          "peak_local_bytes=2130875392 device_dependencies=7 "
          "host_transfers=0"
      )
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(missing, arm="r2")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_report_accumulate=")
        for reason in rejected["reasons"]
    ))

    def host_transfer(root: Path) -> None:
      path = root / "raw.log"
      raw = path.read_text(encoding="utf-8")
      path.write_text(
          raw.replace(
              "group=4/8 leaves=399 buckets=8 executables=1 "
              "peak_local_bytes=2130875392 device_dependencies=7 "
              "host_transfers=0",
              "group=4/8 leaves=399 buckets=8 executables=1 "
              "peak_local_bytes=2130875392 device_dependencies=7 "
              "host_transfers=1",
          ),
          encoding="utf-8",
      )

    rejected = self._classify(host_transfer, arm="r3")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_report_accumulate=")
        for reason in rejected["reasons"]
    ))

    ordinary = self._classify(arm="r1")
    self.assertEqual(ordinary["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        ordinary["receipts"]["reduce_once_report_accumulate"], []
    )

  def test_reduce_once_accumulator_loan_receipt_is_fail_closed(self):
    marker = (
        "[V2.REDUCE_ONCE.ACCUMULATOR_LOAN] enabled=1 "
        "leaves=399 local_bytes=16382087168 "
        "transition=base-to-staged base_handles_retired=399 "
        "staged_handles_retired=399 check_vma=1 host_transfers=0"
    )

    def missing(root: Path) -> None:
      path = root / "raw.log"
      lines = path.read_text(encoding="utf-8").splitlines()
      lines.remove(marker)
      path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(missing, arm="r2")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_accumulator_loan=")
        for reason in rejected["reasons"]
    ))

    def host_transfer(root: Path) -> None:
      path = root / "raw.log"
      raw = path.read_text(encoding="utf-8")
      path.write_text(
          raw.replace(marker, marker.replace("host_transfers=0", "host_transfers=1")),
          encoding="utf-8",
      )

    rejected = self._classify(host_transfer, arm="r3")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_accumulator_loan=")
        for reason in rejected["reasons"]
    ))

    ordinary = self._classify(arm="r1")
    self.assertEqual(ordinary["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        ordinary["receipts"]["reduce_once_accumulator_loan"], []
    )

  def test_reduce_once_accumulator_reset_receipt_is_fail_closed(self):
    marker = (
        "[V2.REDUCE_ONCE.ACCUMULATOR_RESET] enabled=1 "
        "leaves=399 transition=adopted-to-idle "
        "alias_mode=bitwise-zero host_transfers=0"
    )

    def missing(root: Path) -> None:
      path = root / "raw.log"
      lines = path.read_text(encoding="utf-8").splitlines()
      lines.remove(marker)
      path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(missing, arm="r2")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_accumulator_reset=")
        for reason in rejected["reasons"]
    ))

    def host_transfer(root: Path) -> None:
      path = root / "raw.log"
      raw = path.read_text(encoding="utf-8")
      path.write_text(
          raw.replace(
              marker,
              marker.replace("host_transfers=0", "host_transfers=1"),
          ),
          encoding="utf-8",
      )

    rejected = self._classify(host_transfer, arm="r3")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("reduce_once_accumulator_reset=")
        for reason in rejected["reasons"]
    ))

    ordinary = self._classify(arm="r1")
    self.assertEqual(ordinary["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        ordinary["receipts"]["reduce_once_accumulator_reset"], []
    )

  def test_reduce_once_update_gradient_norm_is_fail_closed(self):
    def missing(root: Path) -> None:
      path = root / "updates.json"
      update = json.loads(path.read_text(encoding="utf-8"))
      update.pop("update_gradient_norm")
      path.write_text(json.dumps(update), encoding="utf-8")

    rejected = self._classify(missing, arm="r2")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("update_gradient_norm=None", rejected["reasons"])

    def unexpected(root: Path) -> None:
      path = root / "updates.json"
      update = json.loads(path.read_text(encoding="utf-8"))
      update["update_gradient_norm"] = 12.5
      path.write_text(json.dumps(update), encoding="utf-8")

    rejected = self._classify(unexpected, arm="r1")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("unexpected_update_gradient_norm", rejected["reasons"])

  def test_r2_repin_rejects_old_and_one_ulp_update_norms(self):
    candidate = 23.40300178527832
    for bad_norm in (23.403005599975586, 23.403003692626953):
      with self.subTest(bad_norm=bad_norm), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        anchor_path = self._fixture(
            root, arm="r2", mode="measure", workload="p45"
        )
        update_path = root / "updates.json"
        update = json.loads(update_path.read_text(encoding="utf-8"))
        registry = json.loads(anchor_path.read_text(encoding="utf-8"))
        registry["anchors"]["p45:dp2-tp2:r2"] = {
            "run_id": "v2fl_p45_r0d_capsule_20260904_r32",
            "training_capsule_sha256": (
                "99b6dcaba5b816644a02037ef8f4e8ae"
                "0199eb1106a4142e3d076b3f48d8539c"
            ),
            "micro_gradient_norms": update["micro_gradient_norms"],
            "update_gradient_norm": candidate,
        }
        anchor_path.write_text(json.dumps(registry), encoding="utf-8")
        update["update_gradient_norm"] = bad_norm
        update_path.write_text(json.dumps(update), encoding="utf-8")
        rejected = classifier.classify(
            root,
            workload="p45",
            arm="r2",
            docker_exit=0,
            anchor_registry=anchor_path,
            require_anchor=False,
        )
      self.assertEqual(rejected["verdict"], "FAIL")
      self.assertIn("gradient_anchor_bitwise", rejected["reasons"])

  def test_r3_repin_rejects_adjacent_ulp_and_r2_group_order(self):
    candidate = [
        0.0,
        6.509113311767578,
        9.983430862426758,
        0.0,
        0.0,
        0.0,
        0.0,
        37.86933135986328,
    ]
    bad_candidates = [
        candidate[:-1] + [37.869327545166016],
        [
            37.869327545166016,
            6.235235691070557,
            6.509113311767578,
            7.796840667724609,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
    for bad_norms in bad_candidates:
      with (
          self.subTest(bad_norms=bad_norms),
          tempfile.TemporaryDirectory() as tmp,
      ):
        root = Path(tmp)
        anchor_path = self._fixture(
            root, arm="r3", mode="measure", workload="p45"
        )
        update_path = root / "updates.json"
        update = json.loads(update_path.read_text(encoding="utf-8"))
        registry = json.loads(anchor_path.read_text(encoding="utf-8"))
        registry["anchors"]["p45:dp2-tp2:r3"] = {
            "run_id": "v2fl_p45_r0d_capsule_20260904_r32",
            "training_capsule_sha256": (
                "99b6dcaba5b816644a02037ef8f4e8ae"
                "0199eb1106a4142e3d076b3f48d8539c"
            ),
            "micro_gradient_norms": candidate,
            "update_gradient_norm": 23.40300178527832,
        }
        anchor_path.write_text(json.dumps(registry), encoding="utf-8")
        update["micro_gradient_norms"] = bad_norms
        update["update_gradient_norm"] = 23.40300178527832
        update_path.write_text(json.dumps(update), encoding="utf-8")
        rejected = classifier.classify(
            root,
            workload="p45",
            arm="r3",
            docker_exit=0,
            anchor_registry=anchor_path,
            require_anchor=False,
        )
      self.assertEqual(rejected["verdict"], "FAIL")
      self.assertIn("gradient_anchor_bitwise", rejected["reasons"])

  def test_p45_landed_work_uses_fixed_program_count(self):
    def eight_chunks(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = "\n".join(
          line if "n_real=" not in line else (
              line.split("n_real=")[0] + "n_real=(1792, 1793)"
          )
          for line in raw.splitlines()
      ) + "\n"
      (root / "raw.log").write_text(raw, encoding="utf-8")

    result = self._classify(eight_chunks, workload="p45")
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(result["landed_shape"]["max_n_real"], 1793)
    self.assertEqual(result["landed_shape"]["max_group_chunks"], 8)

    def seven_chunks(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = "\n".join(
          line if "n_real=" not in line else (
              line.split("n_real=")[0] + "n_real=(1792, 1792)"
          )
          for line in raw.splitlines()
      ) + "\n"
      (root / "raw.log").write_text(raw, encoding="utf-8")

    rejected = self._classify(seven_chunks, workload="p45")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("landed_group_chunks_max=7", rejected["reasons"])

  def test_exact_post_training_vllm_finalizer_is_the_only_allowed_traceback(self):
    finalizer = "".join((
        "Exception ignored in: <finalize object at 0x7f140c3678c0; dead>\n"
        "Traceback (most recent call last):\n"
        "  File \"/usr/local/lib/python3.12/weakref.py\", line 590, in __call__\n"
        "    return info.func(*info.args, **(info.kwargs or {}))\n",
        " " * 11 + "^" * 44 + "\n",
        "  File \"/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py\", line 441, in _cleanup_instance_caches\n"
        "    for module in model.modules():\n",
        " " * 18 + "^" * 13 + "\n",
        "AttributeError: 'Qwen3ForCausalLM' object has no attribute 'modules'\n",
    ))

    def append_exact(root: Path) -> None:
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[CANON_FROZENLAKE_P27] TRAINING_DONE max_steps=1\n" + finalizer
        )

    accepted = self._classify(append_exact)
    self.assertEqual(accepted["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        accepted["receipts"]["ignored_post_training_finalizers"], 1
    )

    def append_before_success(root: Path) -> None:
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(finalizer)

    before = self._classify(append_before_success)
    self.assertEqual(before["verdict"], "FAIL")
    self.assertIn("traceback", before["reasons"])

    def append_changed_exception(root: Path) -> None:
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[CANON_FROZENLAKE_P27] TRAINING_DONE max_steps=1\n"
            + finalizer.replace("AttributeError:", "RuntimeError:")
        )

    changed = self._classify(append_changed_exception)
    self.assertEqual(changed["verdict"], "FAIL")
    self.assertIn("traceback", changed["reasons"])

    def append_extra(root: Path) -> None:
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[CANON_FROZENLAKE_P27] TRAINING_DONE max_steps=1\n"
            + finalizer
            + "Traceback (most recent call last):\nextra failure\n"
        )

    extra = self._classify(append_extra)
    self.assertEqual(extra["verdict"], "FAIL")
    self.assertIn("traceback", extra["reasons"])

  def test_training_capsule_capture_and_replay_receipts_are_fail_closed(self):
    def invalid_manifest(root: Path) -> None:
      manifest = json.loads((root / "run_manifest.json").read_text())
      manifest["training_capsule"] = []
      (root / "run_manifest.json").write_text(json.dumps(manifest))

    invalid = self._classify(invalid_manifest)
    self.assertEqual(invalid["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("training_capsule=")
        for reason in invalid["reasons"]
    ))

    def capture(root: Path) -> None:
      capsule_path = root / "training_capsule.npz"
      binding_path = root / "training_capsule.npz.model.json"
      capsule_path.write_bytes(b"exact capsule")
      binding_path.write_text('{"schema":"binding"}\n', encoding="utf-8")
      capsule_sha = classifier._sha256(capsule_path)  # pylint: disable=protected-access
      binding_sha = classifier._sha256(binding_path)  # pylint: disable=protected-access
      manifest = json.loads((root / "run_manifest.json").read_text())
      manifest["training_capsule"] = {
          "mode": "capture",
          "capture_run": None,
          "sha256": None,
          "model_binding_sha256": None,
      }
      (root / "run_manifest.json").write_text(json.dumps(manifest))
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[V2.CAPSULE] capture_ready "
            f"path={capsule_path} sha256={capsule_sha} rows=16 arrays=23 "
            "logical_bytes=1234 certification=strict-prealignment-source\n"
            "[V2.CAPSULE] model_bound mode=capture "
            f"capsule_sha256={capsule_sha} binding_sha256={binding_sha} "
            f"model_fingerprint_sha256={'c' * 64} sampled_model=1\n"
        )

    captured = self._classify(capture)
    self.assertEqual(captured["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        captured["receipts"]["training_capsule"]["mode"], "capture"
    )

    def corrupt_capture(root: Path) -> None:
      capture(root)
      with (root / "training_capsule.npz").open("ab") as output:
        output.write(b"corrupt")

    rejected_capture = self._classify(corrupt_capture)
    self.assertEqual(rejected_capture["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("training_capsule_capture_receipts=")
        for reason in rejected_capture["reasons"]
    ))

    def replay(root: Path) -> None:
      manifest = json.loads((root / "run_manifest.json").read_text())
      manifest["training_capsule"] = {
          "mode": "replay",
          "capture_run": "v2-capture-r1",
          "sha256": "d" * 64,
          "model_binding_sha256": "e" * 64,
      }
      (root / "run_manifest.json").write_text(json.dumps(manifest))
      raw_path = root / "raw.log"
      raw_path.write_text(
          raw_path.read_text(encoding="utf-8").replace(
              "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
              "use_rollout_logps=1 tis_weights=absent\n",
              "",
          ),
          encoding="utf-8",
      )
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[V2.CAPSULE] diagnostic_replay_ready "
            f"path=/evidence/training_capsule.npz sha256={'d' * 64} "
            "rows=16 capture_run=v2-capture-r1 "
            f"capture_source={'b' * 40} replay_source={'a' * 40} "
            "rollout=skipped rescore_b=skipped certification=0\n"
            "[V2.CAPSULE] producer_bypass verdict=PASS "
            "environment=0 rollout=0 rescore_b=0\n"
            "[V2.CAPSULE] model_verified mode=replay "
            f"capsule_sha256={'d' * 64} binding_sha256={'e' * 64} "
            f"model_fingerprint_sha256={'c' * 64} sampled_model=1\n"
        )

    replayed = self._classify(
        replay, anchor=True, require_anchor=True
    )
    self.assertEqual(replayed["verdict"], "PASS")
    self.assertEqual(
        replayed["receipts"]["training_capsule"]["sha256"], "d" * 64
    )
    self.assertEqual(replayed["receipts"]["sampler_contract"], 0)

    wrong_capture = self._classify(
        replay,
        anchor=True,
        anchor_run_id="different-capture-r1",
        require_anchor=True,
    )
    self.assertEqual(wrong_capture["verdict"], "FAIL")
    self.assertIn("gradient_anchor_capture_run", wrong_capture["reasons"])

    def missing_bypass(root: Path) -> None:
      replay(root)
      raw = (root / "raw.log").read_text()
      (root / "raw.log").write_text(
          raw.replace(
              "[V2.CAPSULE] producer_bypass verdict=PASS "
              "environment=0 rollout=0 rescore_b=0\n",
              "",
          )
      )

    rejected_replay = self._classify(
        missing_bypass, anchor=True, require_anchor=True
    )
    self.assertEqual(rejected_replay["verdict"], "FAIL")
    self.assertIn(
        "training_capsule_producer_bypass", rejected_replay["reasons"]
    )

    def unexpected_sampler(root: Path) -> None:
      replay(root)
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
            "use_rollout_logps=1 tis_weights=absent\n"
        )

    rejected_sampler = self._classify(
        unexpected_sampler, anchor=True, require_anchor=True
    )
    self.assertEqual(rejected_sampler["verdict"], "FAIL")
    self.assertIn(
        "sampler_receipts=1;expected=0", rejected_sampler["reasons"]
    )

  def test_r0_measure_requires_exact_hbm_stage_receipts(self):
    result = self._classify(arm="r0")
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        [item["stage"] for item in result["receipts"]["hbm_stage_receipts"]],
        list(classifier._expected_hbm_stages(17)),  # pylint: disable=protected-access
    )
    self.assertTrue(all(
        item["group"] == 3
        for item in result["receipts"]["hbm_stage_receipts"]
    ))
    self.assertEqual(
        result["receipts"]["report_adjoint_memory"][
            "output_size_in_bytes"
        ],
        16_382_087_168,
    )

    def remove_one_stage(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      removed = update["hbm_stage_receipts"].pop(2)
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      raw = (root / "raw.log").read_text(encoding="utf-8")
      marker = (
          "[V2.FL.HBM_STAGE] "
          + json.dumps(removed, sort_keys=True, separators=(",", ":"))
          + "\n"
      )
      (root / "raw.log").write_text(raw.replace(marker, ""), encoding="utf-8")

    rejected = self._classify(remove_one_stage, arm="r0")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("hbm_stage_receipts", rejected["reasons"])

    def remove_report_memory(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      lines = [
          line
          for line in raw.splitlines()
          if not line.startswith("[V2.FL.REPORT_ADJOINT_MEMORY] ")
      ]
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(remove_report_memory, arm="r0")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("report_adjoint_memory_receipts=0", rejected["reasons"])

  def test_p45_r0b_measure_requires_checked_bucket_receipts(self):
    result = self._classify(arm="r0b", workload="p45")
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    receipt = result["receipts"]["report_adjoint_memory"]
    self.assertEqual(
        receipt["schema"],
        "canon-v2-p75-report-adjoint-buckets-memory-v1",
    )
    self.assertEqual(receipt["bucket_count"], 8)
    self.assertEqual(receipt["host_blocks"], 8)
    bucket_hbm = [
        item
        for item in result["receipts"]["hbm_stage_receipts"]
        if item["stage"].startswith("report_bucket_")
    ]
    self.assertEqual(len(bucket_hbm), 48)
    group_hbm = [
        item
        for item in result["receipts"]["hbm_stage_receipts"]
        if item["stage"] == "report_group_after_bucket_0"
    ]
    self.assertEqual([item["group"] for item in group_hbm], list(range(8)))
    outer_hbm = [
        item
        for item in result["receipts"]["hbm_stage_receipts"]
        if not item["stage"].startswith("report_bucket_")
        and item["stage"] != "report_group_after_bucket_0"
    ]
    self.assertEqual(
        [item["stage"] for item in outer_hbm],
        list(classifier._expected_hbm_stages(16)),  # pylint: disable=protected-access
    )
    self.assertTrue(all(item["group"] == 1 for item in outer_hbm))

    def remove_bucket_markers(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      lines = [
          line
          for line in raw.splitlines()
          if not line.startswith("[P75.REPORT_ADJOINT_BUCKETS] ")
      ]
      (root / "raw.log").write_text(
          "\n".join(lines) + "\n", encoding="utf-8"
      )

    rejected = self._classify(
        remove_bucket_markers, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn(
        "p75_report_adjoint_bucket_receipts=0", rejected["reasons"]
    )

    def replace_bucket_memory_with_monolithic(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      replacement = (
          "[V2.FL.REPORT_ADJOINT_MEMORY] "
          + json.dumps(
              {
                  "schema": "canon-v2-p59-report-adjoint-memory-v1",
                  "staged_engine_leaves": 399,
                  "trainer_leaves": 399,
                  "argument_size_in_bytes": 1,
                  "output_size_in_bytes": 1,
                  "alias_size_in_bytes": 0,
                  "temp_size_in_bytes": 0,
                  "host_argument_size_in_bytes": 0,
                  "host_output_size_in_bytes": 0,
                  "host_alias_size_in_bytes": 0,
                  "host_temp_size_in_bytes": 0,
              },
              sort_keys=True,
              separators=(",", ":"),
          )
      )
      lines = [
          replacement
          if line.startswith("[V2.FL.REPORT_ADJOINT_MEMORY] ")
          else line
          for line in raw.splitlines()
      ]
      (root / "raw.log").write_text(
          "\n".join(lines) + "\n", encoding="utf-8"
      )

    rejected = self._classify(
        replace_bucket_memory_with_monolithic,
        arm="r0b",
        workload="p45",
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn(
        "bucketed_report_memory_receipts=1", rejected["reasons"]
    )

    def remove_bucket_hbm_stage(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      removed = next(
          item
          for item in update["hbm_stage_receipts"]
          if item["stage"] == "report_bucket_3_after_execute"
      )
      update["hbm_stage_receipts"].remove(removed)
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      raw = (root / "raw.log").read_text(encoding="utf-8")
      marker = (
          "[V2.FL.HBM_STAGE] "
          + json.dumps(removed, sort_keys=True, separators=(",", ":"))
          + "\n"
      )
      (root / "raw.log").write_text(raw.replace(marker, ""), encoding="utf-8")

    rejected = self._classify(
        remove_bucket_hbm_stage, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_bucket_hbm_receipts", rejected["reasons"])

    def corrupt_bucket_hbm_device(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      record = next(
          item
          for item in update["hbm_stage_receipts"]
          if item["stage"] == "report_bucket_4_after_delete"
      )
      record["devices"][3]["device"] = 2
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      lines = []
      replaced = False
      for line in (root / "raw.log").read_text(encoding="utf-8").splitlines():
        if (
            not replaced
            and line.startswith("[V2.FL.HBM_STAGE] ")
            and '"stage":"report_bucket_4_after_delete"' in line
        ):
          line = (
              "[V2.FL.HBM_STAGE] "
              + json.dumps(record, sort_keys=True, separators=(",", ":"))
          )
          replaced = True
        lines.append(line)
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(
        corrupt_bucket_hbm_device, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_bucket_hbm_receipts", rejected["reasons"])

    def reorder_bucket_hbm_stages(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      indices = [
          index
          for index, item in enumerate(update["hbm_stage_receipts"])
          if item["stage"] in (
              "report_bucket_2_before_execute",
              "report_bucket_2_after_execute",
          )
          and item["group"] == 1
      ]
      update["hbm_stage_receipts"][indices[0]], update["hbm_stage_receipts"][
          indices[1]
      ] = (
          update["hbm_stage_receipts"][indices[1]],
          update["hbm_stage_receipts"][indices[0]],
      )
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      lines = (root / "raw.log").read_text(encoding="utf-8").splitlines()
      raw_indices = [
          index
          for index, line in enumerate(lines)
          if '"group":1' in line
          and (
              '"stage":"report_bucket_2_before_execute"' in line
              or '"stage":"report_bucket_2_after_execute"' in line
          )
      ]
      lines[raw_indices[0]], lines[raw_indices[1]] = (
          lines[raw_indices[1]], lines[raw_indices[0]]
      )
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(
        reorder_bucket_hbm_stages, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_bucket_hbm_receipts", rejected["reasons"])

    def corrupt_bucket_hbm_bytes(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      record = next(
          item
          for item in update["hbm_stage_receipts"]
          if item["stage"] == "report_bucket_6_after_execute"
          and item["group"] == 1
      )
      record["devices"][0]["bytes_in_use"] = (
          record["devices"][0]["peak_bytes_in_use"] + 1
      )
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      lines = []
      replaced = False
      for line in (root / "raw.log").read_text(encoding="utf-8").splitlines():
        if (
            not replaced
            and line.startswith("[V2.FL.HBM_STAGE] ")
            and '"group":1' in line
            and '"stage":"report_bucket_6_after_execute"' in line
        ):
          line = (
              "[V2.FL.HBM_STAGE] "
              + json.dumps(record, sort_keys=True, separators=(",", ":"))
          )
          replaced = True
        lines.append(line)
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(
        corrupt_bucket_hbm_bytes, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_bucket_hbm_receipts", rejected["reasons"])

  def test_p45_r0b_requires_all_group_hbm_checkpoints(self):
    result = self._classify(arm="r0b", workload="p45")
    checkpoints = [
        item
        for item in result["receipts"]["hbm_stage_receipts"]
        if item["stage"] == "report_group_after_bucket_0"
    ]
    self.assertEqual([item["group"] for item in checkpoints], list(range(8)))

    def rewrite_hbm_receipts(root: Path, records: list[dict]) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      update["hbm_stage_receipts"] = records
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      lines = [
          line
          for line in (root / "raw.log").read_text(encoding="utf-8").splitlines()
          if not line.startswith("[V2.FL.HBM_STAGE] ")
      ]
      lines.extend(
          "[V2.FL.HBM_STAGE] "
          + json.dumps(item, sort_keys=True, separators=(",", ":"))
          for item in records
      )
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def remove_checkpoint(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      records = update["hbm_stage_receipts"]
      record = next(
          item
          for item in records
          if item["stage"] == "report_group_after_bucket_0"
          and item["group"] == 4
      )
      records.remove(record)
      rewrite_hbm_receipts(root, records)

    rejected = self._classify(
        remove_checkpoint, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_group_hbm_receipts", rejected["reasons"])

    def duplicate_group(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      records = update["hbm_stage_receipts"]
      record = next(
          item
          for item in records
          if item["stage"] == "report_group_after_bucket_0"
          and item["group"] == 5
      )
      record["group"] = 4
      rewrite_hbm_receipts(root, records)

    rejected = self._classify(
        duplicate_group, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_group_hbm_receipts", rejected["reasons"])

    def reorder_groups(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      records = update["hbm_stage_receipts"]
      indices = [
          index
          for index, item in enumerate(records)
          if item["stage"] == "report_group_after_bucket_0"
          and item["group"] in (5, 6)
      ]
      records[indices[0]], records[indices[1]] = (
          records[indices[1]], records[indices[0]]
      )
      rewrite_hbm_receipts(root, records)

    rejected = self._classify(
        reorder_groups, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_group_hbm_receipts", rejected["reasons"])

    def corrupt_bytes(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      records = update["hbm_stage_receipts"]
      record = next(
          item
          for item in records
          if item["stage"] == "report_group_after_bucket_0"
          and item["group"] == 6
      )
      record["devices"][0]["bytes_in_use"] = (
          record["devices"][0]["peak_bytes_in_use"] + 1
      )
      rewrite_hbm_receipts(root, records)

    rejected = self._classify(
        corrupt_bytes, arm="r0b", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("p75_group_hbm_receipts", rejected["reasons"])

  def test_p45_r0c_requires_exact_ticket_without_host_observer(self):
    result = self._classify(arm="r0c", workload="p45")
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(
        result["receipts"]["p76_chunk_dependency"], (310, 1, 1, 0)
    )
    self.assertEqual(result["receipts"]["hbm_stage_receipts"], [])
    self.assertIsNone(result["receipts"]["report_adjoint_memory"])

    def remove_ticket(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      lines = [
          line
          for line in raw.splitlines()
          if not line.startswith("[P76.CHUNK_DEPENDENCY] ")
      ]
      (root / "raw.log").write_text(
          "\n".join(lines) + "\n", encoding="utf-8"
      )

    rejected = self._classify(
        remove_ticket, arm="r0c", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn(
        "p76_chunk_dependency_receipts=[]", rejected["reasons"]
    )

    def claim_host_transfer(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = raw.replace("host_transfers=0", "host_transfers=1")
      (root / "raw.log").write_text(raw, encoding="utf-8")

    rejected = self._classify(
        claim_host_transfer, arm="r0c", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn(
        "p76_chunk_dependency_receipts=[(310, 1, 1, 1)]",
        rejected["reasons"],
    )

  def test_r0b_rejects_the_m15_neighbor(self):
    result = self._classify(arm="r0b", workload="m15")
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("p75_p76_p77_workload=m15", result["reasons"])

  def test_r0c_rejects_the_m15_neighbor(self):
    result = self._classify(arm="r0c", workload="m15")
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("p75_p76_p77_workload=m15", result["reasons"])

  def test_p45_r0d_requires_exact_device_ready_backpressure(self):
    result = self._classify(arm="r0d", workload="p45")
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(len(result["receipts"]["p77_chunk_backpressure"]), 8)
    self.assertEqual(
        [item[0] for item in result["receipts"]["p77_chunk_backpressure"]],
        [16, 16, 16, 17, 17, 17, 17, 17],
    )
    self.assertEqual(
        [item[1] for item in result["receipts"]["p77_chunk_backpressure"]],
        [16, 16, 16, 17, 17, 17, 17, 17],
    )
    self.assertEqual(result["receipts"]["hbm_stage_receipts"], [])

    def remove_wait(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      marker = "[P77.CHUNK_BACKPRESSURE] enabled=1 "
      removed = False
      lines = []
      for line in raw.splitlines():
        if line.startswith(marker) and not removed:
          removed = True
          continue
        lines.append(line)
      (root / "raw.log").write_text(
          "\n".join(lines) + "\n", encoding="utf-8"
      )

    rejected = self._classify(remove_wait, arm="r0d", workload="p45")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("p77_chunk_backpressure_receipts=")
        for reason in rejected["reasons"]
    ))

    def corrupt_wait_count(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = raw.replace(
          "pullback_waits=16 accumulation_waits=16",
          "pullback_waits=15 accumulation_waits=16",
          1,
      )
      (root / "raw.log").write_text(raw, encoding="utf-8")

    rejected = self._classify(
        corrupt_wait_count, arm="r0d", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("p77_chunk_backpressure_receipts=")
        for reason in rejected["reasons"]
    ))

    def claim_host_transfer(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = raw.replace(
          "wait_api=block_until_ready host_transfers=0",
          "wait_api=block_until_ready host_transfers=1",
      )
      (root / "raw.log").write_text(raw, encoding="utf-8")

    rejected = self._classify(
        claim_host_transfer, arm="r0d", workload="p45"
    )
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertTrue(any(
        reason.startswith("p77_chunk_backpressure_receipts=")
        for reason in rejected["reasons"]
    ))

  def test_r0d_rejects_the_m15_neighbor(self):
    result = self._classify(arm="r0d", workload="m15")
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("p75_p76_p77_workload=m15", result["reasons"])

  def test_r0_measure_rejects_a_missing_chunk_boundary(self):
    def remove_chunk_boundary(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      removed = next(
          item
          for item in update["hbm_stage_receipts"]
          if item["stage"].startswith("model_after_pullbacks_chunk_")
      )
      update["hbm_stage_receipts"].remove(removed)
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      raw = (root / "raw.log").read_text(encoding="utf-8")
      marker = (
          "[V2.FL.HBM_STAGE] "
          + json.dumps(removed, sort_keys=True, separators=(",", ":"))
          + "\n"
      )
      (root / "raw.log").write_text(raw.replace(marker, ""), encoding="utf-8")

    rejected = self._classify(remove_chunk_boundary, arm="r0")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("hbm_stage_receipts", rejected["reasons"])

  def test_r0_measure_rejects_group0_receipts_when_a_later_group_is_longer(self):
    def change_group(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      for item in update["hbm_stage_receipts"]:
        item["group"] = 0
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
      raw = (root / "raw.log").read_text(encoding="utf-8")
      lines = [
          line
          for line in raw.splitlines()
          if not line.startswith("[V2.FL.HBM_STAGE] ")
      ]
      lines.extend(
          "[V2.FL.HBM_STAGE] "
          + json.dumps(item, sort_keys=True, separators=(",", ":"))
          for item in update["hbm_stage_receipts"]
      )
      (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rejected = self._classify(change_group, arm="r0")
    self.assertEqual(rejected["verdict"], "FAIL")
    self.assertIn("hbm_stage_receipts", rejected["reasons"])

  def test_registered_anchor_does_not_promote_a_live_measurement(self):
    result = self._classify(anchor=True)
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertTrue(result["gradient"]["anchor_exact"])

  def test_certification_refuses_an_unregistered_anchor(self):
    result = self._classify(require_anchor=True)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("gradient_anchor_unregistered", result["reasons"])
    self.assertIn(
        "certify_requires_training_capsule_replay", result["reasons"]
    )

  def test_one_ulp_anchor_drift_rejects(self):
    def mutate(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      update["micro_gradient_norms"][3] += 0.000001
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    # Register the original expected values after restoring them explicitly.
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(root)
      original = _update()["micro_gradient_norms"]
      registry = json.loads(anchor_path.read_text(encoding="utf-8"))
      registry["anchors"]["m15:dp2-tp2:r3"] = {
          "run_id": "measurement-run-r1",
          "micro_gradient_norms": original,
          "update_gradient_norm": _update()["update_gradient_norm"],
      }
      anchor_path.write_text(json.dumps(registry), encoding="utf-8")
      mutate(root)
      result = classifier.classify(
          root,
          workload="m15",
          arm="r3",
          docker_exit=0,
          anchor_registry=anchor_path,
          require_anchor=False,
      )
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("gradient_anchor_bitwise", result["reasons"])

  def test_alignment_hbm_vma_length_and_reduction_negatives_fire(self):
    def wrong_vllm_hbm_utilization(root: Path) -> None:
      manifest = json.loads(
          (root / "run_manifest.json").read_text(encoding="utf-8")
      )
      manifest["vllm_hbm_utilization"] = 0.20
      (root / "run_manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8"
      )

    def bad_alignment(root: Path) -> None:
      row = _alignment(pre=True)
      row["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] = 4
      (root / "pre_alignment.jsonl").write_text(
          json.dumps(row) + "\n", encoding="utf-8"
      )

    def bad_hbm(root: Path) -> None:
      update = _update()
      update["hbm_after_reverse"][0]["peak_bytes_in_use"] = 91
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    def no_vma(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace("[P66.VMA] outer_check_enabled program=test\n", ""),
          encoding="utf-8",
      )

    def no_seed_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace(
              "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
              "vllm_global_seed=0 per_request_seed=unsupported\n",
              "",
          ),
          encoding="utf-8",
      )

    def no_token_hash(root: Path) -> None:
      row = _alignment(pre=True)
      del row["hashes"]["tokens"]
      (root / "pre_alignment.jsonl").write_text(
          json.dumps(row) + "\n", encoding="utf-8"
      )

    def no_disabled_wandb_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace("[V2.FL.WANDB] DISABLED_LOCAL_PASS test\n", ""),
          encoding="utf-8",
      )

    def no_sampler_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace(
              "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
              "use_rollout_logps=1 tis_weights=absent\n",
              "",
          ),
          encoding="utf-8",
      )

    def no_reducer_bucket_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          "\n".join(
              line
              for line in raw.splitlines()
              if "reducer_bucket_schedule" not in line
          )
          + "\n",
          encoding="utf-8",
      )

    def too_short(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = "\n".join(
          reline if "n_real=" not in reline else reline.split("n_real=")[0] + "n_real=(1000, 2000)"
          for reline in raw.splitlines()
      ) + "\n"
      (root / "raw.log").write_text(raw, encoding="utf-8")

    def wrong_reduction(root: Path) -> None:
      update = _update()
      update["dp_reduction_transactions"] = 8
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    for name, mutation, reason in (
        ("vllm_hbm", wrong_vllm_hbm_utilization, "manifest:"),
        ("alignment", bad_alignment, "pre_alignment_not_exact"),
        ("hbm", bad_hbm, "hbm_headroom_ratio="),
        ("vma", no_vma, "p66_outer_check_receipts=0"),
        ("seed", no_seed_receipt, "seed_receipts=0"),
        ("wandb", no_disabled_wandb_receipt, "disabled_wandb_receipts=0"),
        ("sampler", no_sampler_receipt, "sampler_receipts=0"),
        ("bucket", no_reducer_bucket_receipt, "reducer_bucket_receipts=0"),
        ("hash", no_token_hash, "input_hash_inventory"),
        ("length", too_short, "landed_n_real_max=2000"),
        ("reduction", wrong_reduction, "update:"),
    ):
      with self.subTest(name=name):
        result = self._classify(mutation)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any(item.startswith(reason) for item in result["reasons"]))

  def test_incomplete_hbm_failure_names_the_first_red(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor = self._fixture(root)
      (root / "alignment.jsonl").unlink()
      (root / "updates.json").unlink()
      with (root / "raw.log").open("a", encoding="utf-8") as output:
        output.write(
            "RESOURCE_EXHAUSTED: RuntimeProgramAllocationFailure "
            "loading program 'jit_reduce_local'\n"
        )
      result = classifier.classify(
          root,
          workload="m15",
          arm="r3",
          docker_exit=137,
          anchor_registry=anchor,
      )
    self.assertEqual(result["verdict"], "INCONCLUSIVE")
    self.assertIn("runtime_hbm_exhausted:jit_reduce_local", result["reasons"])

  def test_runner_is_onehost_only_clean_and_observe_only(self):
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    inner = INNER_PATH.read_text(encoding="utf-8")
    self.assertIn("idle_seconds", runner)
    self.assertIn('if [ "$idle_seconds" -ge 120 ]', runner)
    self.assertIn("acceptance launches require a clean", runner)
    self.assertIn("stopping_own=$container", runner)
    self.assertNotIn("kubectl", runner + inner)
    self.assertNotIn("gcloud", runner + inner)
    self.assertNotIn("WANDB_API_KEY", runner + inner)
    self.assertIn("dp_size=4; tp_size=1; model_dir=qwen8b_tp1", runner)
    self.assertIn("dp_size=2; tp_size=2; model_dir=qwen8b_tp2", runner)
    self.assertIn("dp_size=1; tp_size=4; model_dir=qwen8b", runner)
    self.assertIn('--model "$model_dir"', runner)
    self.assertIn("CANON_P66_P59_CHECK_VMA", inner)
    self.assertIn('"backward-no-commit"', inner)
    self.assertIn('"fixed-local-byte-buckets"', inner)
    self.assertIn("measure|certify", runner)
    self.assertIn("--require-anchor", runner)
    self.assertIn("capture:measure", runner)
    self.assertIn("replay:certify", runner)
    self.assertIn('"$evidence_root"/*/training_capsule.npz', runner)
    self.assertIn(
        'capsule_dir_tail="${capsule_dir_name#${run_identity}_}"', runner
    )
    self.assertIn('capsule_capture_run="${capsule_dir_tail#*_}"', runner)
    self.assertIn("CANON_V2_TRAINING_CAPSULE_SHA256", runner + inner)
    self.assertIn('"training_capsule": {', inner)
    self.assertIn('sha256sum -c "$root/SHA256SUMS"', runner)
    terminal = runner[runner.index("classifier_rc=$?") :]
    self.assertLess(
        terminal.index("RED docker=$docker_rc"), terminal.index("seal_evidence")
    )
    self.assertLess(
        terminal.index('echo "[V2.FL.ONEHOST] $verdict evidence=$root"'),
        terminal.rindex("seal_evidence"),
    )


if __name__ == "__main__":
  unittest.main()
