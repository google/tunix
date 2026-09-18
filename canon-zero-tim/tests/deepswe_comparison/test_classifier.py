#!/usr/bin/env python3
"""P58 native-dose and zero-exact run-classifier controls."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest

from tunix.rl import deepswe_debug


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p58_classifier", Path(__file__).with_name("classify_run.py")
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)

_WANDB_PASS = "[CANON_" "P34_WANDB] ONLINE_RUN_PASS\n"


def _values(
    root: Path,
    arm: str,
    topology: str = "128",
    system_optimization_arm: str | None = None,
) -> dict[str, str]:
  values = {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_DEBUG_DIR": str(root),
      "CANON_EXPECT_COMMIT": "1" * 40,
      "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
      "CANON_RUN_ID": "classifier-test",
      "CANON_P34_RUN_STAGE": "three-update",
      "CANON_P34_CLEAN_ROWS": "1012",
      "CANON_P34_WHITELIST_SHA256": classifier._WHITELIST_SHA256,
  }
  if topology != "128" or system_optimization_arm is not None:
    values["CANON_P58_TOPOLOGY"] = topology
  if system_optimization_arm is not None:
    values["CANON_DEEPSWE_SYSTEM_OPTIMIZATION_ARM"] = system_optimization_arm
    values["CANON_P78_SEGMENTED_ACTOR_LOGPS"] = (
        "1"
        if topology == "128" and system_optimization_arm == "treatment"
        else "0"
    )
    if system_optimization_arm == "treatment":
      values.update({
          "CANON_P32_KEEP_TAPE": "stream",
          "CANON_DP_REDUCE_ONCE": "1",
          "CANON_P32_LENGTH_SORT": "1",
      })
  return values


def _batch():
  items = []
  rewards = []
  advantages = []
  for group in range(8):
    for pair in range(16):
      reward = float(pair == 0)
      items.append(types.SimpleNamespace(
          group_id=f"group-{group}",
          pair_index=pair,
          metadata={"task_identity": {"docker_image": f"task-{group}"}},
          traj={
              "status": "SUCCEEDED",
              "trajectory_reward": reward,
              "conversation_text": [],
          },
      ))
      rewards.append(reward)
      advantages.append(1.0 if pair == 0 else -1.0 / 15.0)
  return items, rewards, advantages


def _boundary(differing: int) -> dict:
  return {"valid": True, "finite": True, "differing_bytes": differing}


def _pre(arm: str) -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS" if arm == "native" else "PASS",
      "blocking_reds": [],
      "boundaries": {
          "S_decode_vs_S_prefill": _boundary(1 if arm == "native" else 0),
          "S_prefill_vs_T_old": _boundary(1 if arm == "native" else 0),
      },
  }


def _post(arm: str) -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS" if arm == "native" else "PASS",
      "blocking_reds": [],
      "boundaries": {
          "S_decode_vs_S_prefill": _boundary(1 if arm == "native" else 0),
          "S_prefill_vs_T_old": _boundary(1 if arm == "native" else 0),
          "T_old_vs_T_current": _boundary(1 if arm == "native" else 0),
      },
  }


def _update(
    step: int,
    arm: str,
    topology: str = "128",
    system_optimization_arm: str | None = None,
) -> dict:
  split = topology == "64split"
  record = {
      "contract_name": (
          "p58-qwen4b-tim-64split" if split else "p58-qwen4b-tim-128"
      ),
      "dp_size": 4 if split else 8,
      "tp_size": 8,
      "global_m": 1024 if split else 2048,
      "verdict": "PASS",
      "commits": 1,
      "train_steps_before": step,
      "train_steps_after": step + 1,
      "gradient_finite": True,
      "dp_replicas_exact": True,
      "dp_reduction_transactions": 32 if split else 16,
      "dp_reduction_rounds_per_transaction": 6,
      "dp_rank_pullbacks_per_transaction": 4 if split else 8,
      "optimizer_placement": "device-resident",
  }
  if arm == "native":
    record["dp_reduction_mode"] = "stock-jax-sharded-trainer"
  else:
    record.update({
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 32 if split else 16,
        "dp_reduction_rounds_per_transaction": 6,
        "dp_rank_pullbacks_per_transaction": 4 if split else 8,
    })
  if system_optimization_arm is not None:
    treatment = system_optimization_arm == "treatment"
    transactions = 32 if split else 16
    record.update({
        "system_optimization_arm": system_optimization_arm,
        "dp_reduction_transactions": 1 if treatment else transactions,
        "dp_reduction_visibility": (
            "EXPLICIT_FIXED_TREE_REDUCE_ONCE"
            if treatment else "EXPLICIT_FIXED_TREE"
        ),
        "dp_staged_accumulations": transactions if treatment else 0,
    })
  return record


def _systemopt_log(topology: str, arm: str) -> str:
  split = topology == "64split"
  marker = "64SPLIT" if split else "128"
  dp = 4 if split else 8
  groups = 32 if split else 16
  return (
      _WANDB_PASS
      + f"[P58.{marker}.SYSTEMOPT] arm={arm} stage=three-update "
      + f"topology={topology} strict=1\n"
      + (
          f"[P32.LENGTH_SORT] enabled=1 rows=128 dp={dp} groups={groups} "
          + "permutation_sha256=" + "a" * 64 + "\n"
      ) * (3 if arm == "treatment" else 0)
      + "[P59.CHECKED_VMA] enabled=1\n" * 3
      + "[V1.FIRST_UPDATE]\n" * 2
      + (
          "[P78.ACTOR_LOGPS] segmented_engine_ready "
          "data=8 tp=8 local_M=256 global_M=2048 "
          "workload=p58-qwen4b-tim-128\n"
          + "[P78.ACTOR_LOGPS] deferred_weight_map_ready "
          "source_leaves=311 target_leaves=311 module_programs=39 "
          "mapped_leaf_outputs=0 source=trainer-state\n" * 3
          + "[P78.ACTOR_LOGPS] dispatch batch=128 groups=16 "
          "chunks=80 local_M=256 global_M=2048 outer_jit=0 "
          "d2h_lengths=0 length_guard=device-finite\n" * 3
          + "[P78.ACTOR_LOGPS] program_cache_release "
          "module_programs=39 outputs_ready=1 shared_engine=retained "
          "global_jax_clear_caches=0 chunk_programs=1\n" * 3
          if not split and arm == "treatment"
          else ""
      )
  )


class P58ClassifierTest(unittest.TestCase):

  def _classify(
      self,
      root: Path,
      arm: str,
      topology: str = "128",
      system_optimization_arm: str | None = None,
  ):
    with contextlib.redirect_stdout(io.StringIO()):
      for step in range(3):
        deepswe_debug.persist_batch(
            *_batch(),
            expected_step=step,
            optimizer_step=step,
            output_dir=root,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            values=_values(root, arm, topology, system_optimization_arm),
        )
    return classifier.classify(
        arm=arm,
        stage="three-update",
        log_text=(
            _systemopt_log(topology, system_optimization_arm)
            if system_optimization_arm is not None
            else _WANDB_PASS
        ),
        debug_dir=root,
        weights=[{"verdict": "PASS", "equal": True}],
        pre_alignment=[_pre(arm)],
        alignment=[_post(arm)],
        updates=[
            _update(step, arm, topology, system_optimization_arm)
            for step in range(3)
        ],
        topology=topology,
        system_optimization_arm=system_optimization_arm,
    )

  def test_systemopt_requires_topology_exact_runtime_receipts(self):
    for topology in ("128", "64split"):
      for arm in ("control", "treatment"):
        with (
            self.subTest(topology=topology, arm=arm),
            tempfile.TemporaryDirectory() as directory,
        ):
          root = Path(directory)
          report = self._classify(
              root, "zero", topology, system_optimization_arm=arm
          )
          self.assertEqual(report["verdict"], "PASS")
          self.assertEqual(
              report["length_sort_receipts"],
              3 if arm == "treatment" else None,
          )
          for missing_key in (
              "dp_reduction_visibility",
              "dp_staged_accumulations",
          ):
            bad_updates = [
                _update(step, "zero", topology, arm) for step in range(3)
            ]
            del bad_updates[1][missing_key]
            failed = classifier.classify(
                arm="zero",
                stage="three-update",
                topology=topology,
                system_optimization_arm=arm,
                log_text=_systemopt_log(topology, arm),
                debug_dir=root,
                weights=[{"verdict": "PASS", "equal": True}],
                pre_alignment=[_pre("zero")],
                alignment=[_post("zero")],
                updates=bad_updates,
            )
            self.assertIn(
                "system_optimization_receipts", failed["failed"]
            )

  def test_treatment_rejects_wrong_or_malformed_length_sort_receipts(self):
    corruptions = (
        ("[P32.LENGTH_SORT]", 1),
        (
            "[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=16 "
            "enabled=1 permutation_sha256=" + "a" * 64,
            1,
        ),
        (
            "[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=16 "
            "unknown=1 permutation_sha256=" + "a" * 64,
            1,
        ),
        (
            "[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=15 "
            "permutation_sha256=" + "a" * 64,
            0,
        ),
        (
            "[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=16 "
            "permutation_sha256=not-a-sha256",
            0,
        ),
    )
    for corruption, expected_malformed in corruptions:
      with (
          self.subTest(corruption=corruption),
          tempfile.TemporaryDirectory() as directory,
      ):
        root = Path(directory)
        self._classify(
            root, "zero", "128", system_optimization_arm="treatment"
        )
        good = _systemopt_log("128", "treatment")
        first = next(
            line for line in good.splitlines()
            if line.startswith("[P32.LENGTH_SORT]")
        )
        failed = classifier.classify(
            arm="zero",
            stage="three-update",
            topology="128",
            system_optimization_arm="treatment",
            log_text=good.replace(first, corruption, 1),
            debug_dir=root,
            weights=[{"verdict": "PASS", "equal": True}],
            pre_alignment=[_pre("zero")],
            alignment=[_post("zero")],
            updates=[
                _update(step, "zero", "128", "treatment")
                for step in range(3)
            ],
        )
        self.assertIn("system_optimization_receipts", failed["failed"])
        self.assertEqual(
            failed["length_sort_malformed_receipts"], expected_malformed
        )

  def test_treatment_rejects_missing_or_extra_length_sort_receipts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "zero", "128", system_optimization_arm="treatment")
      good = _systemopt_log("128", "treatment")
      receipt = next(
          line for line in good.splitlines()
          if line.startswith("[P32.LENGTH_SORT]")
      )
      for log_text in (
          good.replace(receipt + "\n", "", 1),
          good + receipt + "\n",
      ):
        with self.subTest(receipts=log_text.count("[P32.LENGTH_SORT]")):
          failed = classifier.classify(
              arm="zero",
              stage="three-update",
              topology="128",
              system_optimization_arm="treatment",
              log_text=log_text,
              debug_dir=root,
              weights=[{"verdict": "PASS", "equal": True}],
              pre_alignment=[_pre("zero")],
              alignment=[_post("zero")],
              updates=[
                  _update(step, "zero", "128", "treatment")
                  for step in range(3)
              ],
          )
          self.assertIn("system_optimization_receipts", failed["failed"])

  def test_128_treatment_rejects_p78_receipt_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "zero", "128", system_optimization_arm="treatment")
      good = _systemopt_log("128", "treatment")
      for marker in (
          "[P78.ACTOR_LOGPS] segmented_engine_ready ",
          "[P78.ACTOR_LOGPS] deferred_weight_map_ready ",
          "[P78.ACTOR_LOGPS] dispatch ",
          "[P78.ACTOR_LOGPS] program_cache_release ",
      ):
        first = next(line for line in good.splitlines() if line.startswith(marker))
        failed = classifier.classify(
            arm="zero",
            stage="three-update",
            topology="128",
            system_optimization_arm="treatment",
            log_text=good.replace(first + "\n", "", 1),
            debug_dir=root,
            weights=[{"verdict": "PASS", "equal": True}],
            pre_alignment=[_pre("zero")],
            alignment=[_post("zero")],
            updates=[
                _update(step, "zero", "128", "treatment")
                for step in range(3)
            ],
        )
        self.assertIn("segmented_actor_logps_receipts", failed["failed"])

  def test_non_128_systemopt_rejects_p78_receipts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(
          root, "zero", "64split", system_optimization_arm="treatment"
      )
      failed = classifier.classify(
          arm="zero",
          stage="three-update",
          topology="64split",
          system_optimization_arm="treatment",
          log_text=(
              _systemopt_log("64split", "treatment")
              + "[P78.ACTOR_LOGPS] segmented_engine_ready data=4 tp=8\n"
          ),
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("zero")],
          alignment=[_post("zero")],
          updates=[
              _update(step, "zero", "64split", "treatment")
              for step in range(3)
          ],
      )
      self.assertIn("segmented_actor_logps_receipts", failed["failed"])

  def test_control_rejects_any_length_sort_marker(self):
    for receipt in (
        "[P32.LENGTH_SORT]",
        "[P32.LENGTH_SORT] enabled=1 rows=128 dp=8 groups=16 "
        "permutation_sha256=" + "a" * 64,
    ):
      with (
          self.subTest(receipt=receipt),
          tempfile.TemporaryDirectory() as directory,
      ):
        root = Path(directory)
        self._classify(root, "zero", "128", system_optimization_arm="control")
        failed = classifier.classify(
            arm="zero",
            stage="three-update",
            topology="128",
            system_optimization_arm="control",
            log_text=_systemopt_log("128", "control") + receipt + "\n",
            debug_dir=root,
            weights=[{"verdict": "PASS", "equal": True}],
            pre_alignment=[_pre("zero")],
            alignment=[_post("zero")],
            updates=[
                _update(step, "zero", "128", "control")
                for step in range(3)
            ],
        )
        self.assertIn("system_optimization_receipts", failed["failed"])

  def test_systemopt_full_admits_only_treatment(self):
    for topology in ("128", "64split"):
      with (
          self.subTest(topology=topology),
          tempfile.TemporaryDirectory() as directory,
      ):
        report = classifier.classify(
            arm="zero",
            stage="full",
            topology=topology,
            system_optimization_arm="treatment",
            log_text="",
            debug_dir=Path(directory),
            weights=[],
            pre_alignment=[],
            alignment=[],
            updates=[],
        )
        self.assertEqual(report["verdict"], "FAIL")
      with self.subTest(topology=topology), self.assertRaises(ValueError):
        classifier.classify(
            arm="zero",
            stage="full",
            topology=topology,
            system_optimization_arm="control",
            log_text="",
            debug_dir=Path("/nonexistent"),
            weights=[],
            pre_alignment=[],
            alignment=[],
            updates=[],
        )

  def test_64split_native_and_zero_use_split_evidence_geometry(self):
    for arm in ("native", "zero"):
      with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
        report = self._classify(Path(directory), arm, topology="64split")
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["topology"], "64split")

  def test_native_requires_a_finite_nonzero_treatment_dose(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      report = self._classify(root, "native")
      self.assertEqual(report["verdict"], "PASS")
      zero_dose = [_pre("zero")]
      failed = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=zero_dose,
          alignment=[_post("zero")],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertIn("registered_treatment_observed", failed["failed"])

  def test_native_b_c_only_is_a_registered_treatment_dose(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "native")
      b_c_only = _pre("native")
      b_c_only["boundaries"]["S_decode_vs_S_prefill"] = _boundary(0)
      report = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[b_c_only],
          alignment=[_post("native")],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertNotIn("registered_treatment_observed", report["failed"])

  def test_native_trainer_program_drift_is_finite_observation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "native")
      drift = _post("native")
      drift["boundaries"]["T_old_vs_T_current"] = _boundary(1)
      report = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("native")],
          alignment=[drift],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertTrue(report["checks"]["native_trainer_program_finite"])

  def test_native_trainer_program_nonfinite_remains_blocking(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "native")
      invalid = _post("native")
      invalid["boundaries"]["T_old_vs_T_current"]["finite"] = False
      failed = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("native")],
          alignment=[invalid],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertIn("alignment_nonblocking_finite", failed["failed"])
      self.assertIn("native_trainer_program_finite", failed["failed"])

  def test_regular_zero_requires_trainer_boundaries_exact(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      report = self._classify(root, "zero")
      self.assertEqual(report["verdict"], "PASS")
      drift = _post("zero")
      drift["boundaries"]["T_old_vs_T_current"] = _boundary(1)
      failed = classifier.classify(
          arm="zero",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("zero")],
          alignment=[drift],
          updates=[_update(step, "zero") for step in range(3)],
      )
      self.assertIn("zero_trainer_boundaries_exact", failed["failed"])

  def test_regular_zero_requires_trainer_repeat_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "zero")
      missing_repeat = _post("zero")
      del missing_repeat["boundaries"]["T_old_vs_T_current"]
      failed = classifier.classify(
          arm="zero",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("zero")],
          alignment=[missing_repeat],
          updates=[_update(step, "zero") for step in range(3)],
      )
      self.assertIn("zero_trainer_boundaries_exact", failed["failed"])

if __name__ == "__main__":
  unittest.main()
