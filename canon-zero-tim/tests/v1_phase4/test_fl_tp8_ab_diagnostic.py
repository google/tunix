from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/v1-phase4-three-full-recipes"
EVIDENCE = TASK / "evidence/v1_hp_three_full_attempt9_20260826"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


CLASSIFIER = _load(
    "classify_fl_tp8_ab_diagnostic", TASK / "scripts/classify_fl_tp8_ab_diagnostic.py"
)
RENDERER = _load(
    "render_fl_tp8_ab_diagnostic", TASK / "scripts/render_fl_tp8_ab_diagnostic.py"
)
PALLAS_MATMUL = _load(
    "p22_pallas_matmul",
    ROOT / "canon-zero-tim/src/engine_shims/p22_pallas_matmul.py",
)
SOURCE = "0a30e6064555bcb42a36cd50b1386aa07f6fea9f"


class FrozenLakeTp8AbDiagnosticTest(unittest.TestCase):

  def test_p67_vma_scope_bypasses_serving_and_retains_p59(self):
    manual = object()
    calls: list[tuple[object, str, str]] = []

    class Mat:
      def __init__(self, varying=()):
        self.varying = frozenset(varying)
        self.unreduced = frozenset()
        self.reduced = frozenset()

    values = (types.SimpleNamespace(mat=Mat()), types.SimpleNamespace(mat=Mat(("model",))))
    context = types.SimpleNamespace(
        axis_names=("attn_head",),
        axis_types=(manual,),
        shape={"attn_head": 8},
    )
    fake_jax = types.SimpleNamespace(
        sharding=types.SimpleNamespace(
            AxisType=types.SimpleNamespace(Manual=manual),
            get_abstract_mesh=lambda: context,
            ManualAxisType=lambda **kwargs: kwargs,
        ),
        typeof=lambda value: types.SimpleNamespace(mat=value.mat),
        lax=types.SimpleNamespace(
            pcast=lambda value, axis, to: calls.append((value, axis, to)) or value
        ),
    )
    env = {
        "CANON_P66_P59_CHECK_VMA": "1",
        "CANON_P67_P66_VMA_P59_ONLY": "1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      self.assertEqual(
          PALLAS_MATMUL.p66_vma_align_operands(fake_jax, *values), values
      )
      self.assertEqual(calls, [])
      context.axis_names = ("data", "model")
      context.axis_types = (manual, manual)
      context.shape = {"data": 8, "model": 8}
      PALLAS_MATMUL.p66_vma_align_operands(fake_jax, *values)
      self.assertEqual(len(calls), 1)
      self.assertEqual(calls[0][1:], ("model", "varying"))

  def test_engine_patch_scopes_embed_and_rpa_vma(self):
    package = ROOT / "canon-zero-tim"
    for relative in (
        "patches/tpu_inference/02-embed.patch",
        "patches/tpu_inference/29-rpa-p66-vma-output.patch",
    ):
      text = (package / relative).read_text(encoding="utf-8")
      self.assertIn("CANON_P67_P66_VMA_P59_ONLY", text)
      self.assertIn("get_abstract_mesh", text)
      self.assertIn("CANON_P59_RANK_PARALLEL_BACKWARD", text)

  def _raw(self, workload: str, arm: str = "p66-off") -> str:
    checked_vma = "0" if arm == "p66-off" else "1"
    p59_only = "0" if arm == "p66-off" else "1"
    return "\n".join((
        f"[V1.FL.AB] profile_resolved arm={arm} workload={workload} dp=8 tp=8 "
        f"checked_vma={checked_vma} vma_p59_only={p59_only} "
        "fixed_ar_gather=1 continue_decode=8 prefix_cache=0 "
        "backward=0 optimizer_commits=0",
        "[CANON_P38] PRECHECK_ROUND_COMPLETE round=1/1 step=0 N_action=1 "
        "verdict=FAIL a_b_differing_bytes=1 backward=0 optimizer_commits=0",
        "[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD rounds=1 step=0",
        "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0",
    )) + "\n"

  def test_attempt9_reds_classify_as_informative_zero_commit(self):
    fixtures = {
        "p45": EVIDENCE / "f46s_p8_pre_alignment.jsonl",
        "m15": EVIDENCE / "m16s_m8_pre_alignment.jsonl",
    }
    for workload, pre_alignment in fixtures.items():
      with self.subTest(workload=workload), tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "run.log"
        raw.write_text(self._raw(workload), encoding="utf-8")
        result = CLASSIFIER.classify(
            raw=raw,
            pre_alignment=pre_alignment,
            workload=workload,
            arm="p66-off",
            output=Path(tmp) / "classification.json",
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["outcome"], "A_B_RED_REPRODUCED")
        self.assertEqual(result["B_C_differing_bytes"], 0)

  def test_classifier_accepts_recovery_and_rejects_bc_drift(self):
    source = json.loads(
        (EVIDENCE / "f46s_p8_pre_alignment.jsonl").read_text().splitlines()[0]
    )
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      raw = root / "run.log"
      raw.write_text(self._raw("p45"), encoding="utf-8")
      source["boundaries"]["S_decode_vs_S_prefill"]["differing_bytes"] = 0
      pre = root / "pre.jsonl"
      pre.write_text(json.dumps(source) + "\n", encoding="utf-8")
      recovered = CLASSIFIER.classify(
          raw=raw, pre_alignment=pre, workload="p45", arm="p66-off",
          output=root / "ok.json"
      )
      self.assertEqual(recovered["outcome"], "ZERO_TIM_RECOVERED")
      source["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] = 1
      pre.write_text(json.dumps(source) + "\n", encoding="utf-8")
      failed = CLASSIFIER.classify(
          raw=raw, pre_alignment=pre, workload="p45", arm="p66-off",
          output=root / "bad.json"
      )
      self.assertEqual(failed["verdict"], "FAIL")

  def test_renderer_preserves_production_geometry_and_zero_commit(self):
    for workload, arm in (
        ("p45", "p66-off"),
        ("p45", "serving-scope"),
        ("m15", "p66-off"),
    ):
      with self.subTest(workload=workload, arm=arm), tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "rendered"
        path = RENDERER.render(
            source_commit=SOURCE,
            run_id=f"{workload}abtest1",
            output_dir=output_dir,
            workload=workload,
            arm=arm,
            base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        )
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        pod = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
        container = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
        env = {item["name"]: item.get("value") for item in container["env"]}
        self.assertEqual(env["CANON_V1_FL_TP8_AB_ARM"], arm)
        expected_vma = "0" if arm == "p66-off" else "1"
        self.assertEqual(env["CANON_P59_CHECKED_VMA"], expected_vma)
        self.assertEqual(env["CANON_P66_P59_CHECK_VMA"], expected_vma)
        self.assertEqual(
            env["CANON_P67_P66_VMA_P59_ONLY"],
            "0" if arm == "p66-off" else "1",
        )
        self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
        self.assertEqual(env["CANON_P38_PRECHECK_ONLY"], "1")
        self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
        command = env["CANON_RUN_CMD"]
        self.assertIn("--mesh_dp=8 --mesh_tp=8", command)
        self.assertIn("--mini_batch_size=32", command)
        self.assertIn("--sampler_is=none", command)
        if workload == "m15":
          self.assertIn("--max_response_length=8192", command)
          self.assertIn("--env_max_steps=15", command)
        else:
          self.assertIn("--max_response_length=2048", command)
          self.assertIn("--env_max_steps=5", command)
        resolved_env = os.environ.copy()
        resolved_env.update({
            item["name"]: item["value"]
            for item in container["env"]
            if "value" in item
        })
        resolved_env["CANON_PKG"] = str(ROOT / "canon-zero-tim")
        resolved_env["WANDB_API_KEY"] = "unit-test-only"
        Path(resolved_env["CANON_STATE"]).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["bash", str(ROOT / "canon-zero-tim/cluster/steps/00_env.sh")],
            env=resolved_env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(
            f"[env] V1 FrozenLake TP8 A/B precheck admitted arm={arm}",
            completed.stdout,
        )

  def test_serving_scope_classifier_requires_its_exact_marker(self):
    source = json.loads(
        (EVIDENCE / "f46s_p8_pre_alignment.jsonl").read_text().splitlines()[0]
    )
    source["boundaries"]["S_decode_vs_S_prefill"]["differing_bytes"] = 0
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      pre = root / "pre.jsonl"
      pre.write_text(json.dumps(source) + "\n", encoding="utf-8")
      raw = root / "run.log"
      raw.write_text(self._raw("p45", "serving-scope"), encoding="utf-8")
      result = CLASSIFIER.classify(
          raw=raw,
          pre_alignment=pre,
          workload="p45",
          arm="serving-scope",
          output=root / "ok.json",
      )
      self.assertEqual(result["verdict"], "PASS")
      self.assertEqual(result["arm"], "serving-scope")
      wrong_raw = root / "wrong.log"
      wrong_raw.write_text(self._raw("p45", "p66-off"), encoding="utf-8")
      wrong = CLASSIFIER.classify(
          raw=wrong_raw,
          pre_alignment=pre,
          workload="p45",
          arm="serving-scope",
          output=root / "wrong.json",
      )
      self.assertEqual(wrong["verdict"], "FAIL")


if __name__ == "__main__":
  unittest.main()
