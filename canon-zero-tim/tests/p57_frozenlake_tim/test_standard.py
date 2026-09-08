"""Native64 trainer-old/no-TIS admission, actual source slices, and receipts.

These CPU checks do not certify a model VJP, convergence, or a TPU launch.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
SOURCE = ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
TREE = ast.parse(SOURCE.read_text())
NAMES = {"_p57_tim_standard_enabled", "_validate_p57_tim_standard",
         "_validate_p57_old_logps_source_request"}
CONTRACT = {"Mapping": Mapping, "alignment": SimpleNamespace(AlignmentGateError=ValueError)}


def block(nodes):
  return compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec")


exec(block([n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]), CONTRACT)
ENABLED = CONTRACT["_p57_tim_standard_enabled"]
REQUEST = CONTRACT["_validate_p57_old_logps_source_request"]
VALIDATE = CONTRACT["_validate_p57_tim_standard"]
GOOD = dict(CANON_P57_TIM_ARM="standard", CANON_P57_RUN_KIND="train",
            CANON_PROFILE_FILE="cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env",
            CANON_P57_INFERENCE_REGIME="stock-fast", CANON_P32_WORKLOAD="frozenlake-dp8-tp8",
            CANON_DP_SIZE="8", CANON_TP_SIZE="8", CANON_P57_EXPECTED_UPDATES="300")
PROCESS = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == "_process_results")
START = next(i for i, n in enumerate(PROCESS.body) if ast.unparse(n) == "rollout_per_token_logps = None")
END = next(i for i, n in enumerate(PROCESS.body) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("self.algo_config.num_iterations > 1"))
SELECTION = block(PROCESS.body[START:END])


def select(source="auto", sampler=None, rollout=((-4., -5.),), trainer=None, step=0):
  if trainer is None:
    trainer = jnp.array([[-2., -3.]])
  scope = dict(jax=jax, jnp=jnp, trainer_old_no_tis=source == "trainer",
      padded_old_logprobs=list(rollout), have_actor_mesh=False,
      deepswe_debug=SimpleNamespace(rollout_only=lambda: False),
      self=SimpleNamespace(algo_config=SimpleNamespace(use_rollout_logps=True, sampler_is=sampler),
          rl_cluster=SimpleNamespace(get_actor_per_token_logps=lambda **kw: trainer,
                                    actor_trainer=SimpleNamespace(train_steps=step))),
      alignment=SimpleNamespace(AlignmentGateError=ValueError), expected_step=0,
      prompt_ids=None, completion_ids=None, pad_value=0, eos_value=1,
      compute_logps_micro_batch_size=8, prompt_mask=None, completion_valid_mask=None,
      host_prompt_lengths=None, host_completion_lengths=None, completion_mask=jnp.ones((1, 2)))
  exec(SELECTION, scope)
  return scope


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


sys.path.insert(0, str(PKG / "cluster"))
RENDER = module("p57_standard_renderer", PKG / "cluster/render_p57_frozenlake_tim.py")
SCRIPTS = PKG / "tasks/p57-frozenlake-tim-causal-study/scripts"
VERIFY = module("p57_standard_manifest", SCRIPTS / "verify_three_arm_manifests.py")
RECEIPTS = module("p57_standard_receipts", SCRIPTS / "classify_standard_receipts.py")


class StandardSelectionTest(unittest.TestCase):
  def test_identity_positive_and_negative(self):
    for changes in ({}, {"CANON_P57_RUN_KIND": "eval"},
                    {"CANON_P57_WORKLOAD_CANDIDATE": "m15", "CANON_P57_DATA_SPLIT": "main"}):
      self.assertTrue(ENABLED({**GOOD, **changes}))
      REQUEST({**GOOD, **changes}, "trainer", "none")
    for name in GOOD:
      with self.subTest(name=name):
        self.assertFalse(ENABLED({**GOOD, name: "wrong"}))
    for source, sampler in (("auto", "none"), ("trainer", "token")):
      with self.assertRaises(ValueError):
        REQUEST(GOOD, source, sampler)
    for arm in ("zero", "mismatch", "is", ""):
      REQUEST({"CANON_P57_TIM_ARM": arm}, "auto", "none")
      with self.assertRaises(ValueError):
        REQUEST({"CANON_P57_TIM_ARM": arm}, "trainer", "none")

  def test_batch_validator_checks_actual_source_and_weights(self):
    good = dict(old_logps_source="trainer", sampler_is=None, use_rollout_logps=True,
                rollout_logps_present=True, trainer_logps_present=True,
                old_logps_are_trainer=True, sampler_is_weights_present=False)
    VALIDATE(**good)
    for name, value in good.items():
      with self.subTest(name=name), self.assertRaises(ValueError):
        VALIDATE(**{**good, name: not value if isinstance(value, bool) else "wrong"})

  def test_actual_selection_keeps_legacy_and_selects_trainer_for_standard(self):
    native, tis, standard = select(), select(sampler="token"), select(source="trainer")
    np.testing.assert_array_equal(native["old_per_token_logps"], [[-4., -5.]])
    self.assertIsNone(native["trainer_per_token_logps"])
    for scope in (tis, standard):
      np.testing.assert_array_equal(scope["old_per_token_logps"], [[-2., -3.]])
      np.testing.assert_array_equal(scope["rollout_per_token_logps"], [[-4., -5.]])
      self.assertIs(scope["old_per_token_logps"], scope["trainer_per_token_logps"])

  def test_capture_rejects_missing_rollout_wrong_shapes_or_policy_drift(self):
    for kwargs in ({"rollout": ()}, {"rollout": ((1.,),)},
                   {"trainer": jnp.ones((1, 1))}, {"step": 1}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        select(source="trainer", **kwargs)

  def test_denominator_is_stop_gradient(self):
    grad = jax.grad(lambda old: select(source="trainer", trainer=old)["old_per_token_logps"].sum())(jnp.ones((1, 2)))
    np.testing.assert_array_equal(grad, np.zeros((1, 2)))


class StandardRendererTest(unittest.TestCase):
  def render(self, directory, **kwargs):
    return RENDER.render_all(base_path=PKG / "cluster/jobset-64chip.yaml",
        output_dir=directory, source_commit="a" * 40, run_id="stdhost",
        campaign_tag="p57-standard", checkpoint_mode="new", expected_updates=300,
        **{"arm": "standard", **kwargs})

  def test_both_workloads_pass_resolved_profile_and_topology_gate(self):
    for workload, kw in (("p45", {}), ("m15", dict(workload_candidate="m15", data_split="main"))):
      with self.subTest(workload=workload), tempfile.TemporaryDirectory() as tmp:
        path, = self.render(Path(tmp), **kw)
        VERIFY.verify(path, wave="standard", workload=workload, source="a" * 40)
        standard = yaml.safe_load(path.read_text())
        native_path, = self.render(Path(tmp) / "native", arm="mismatch", **kw)
        native = yaml.safe_load(native_path.read_text())
        # Only the JobSet-derived head address changes with the distinct run name.
        # All worker configuration, autoscale/exclusive topology included, stays.
        def worker(doc):
          return next(job for job in doc["spec"]["replicatedJobs"] if job["name"] == "pathways-worker")
        self.assertEqual(
            yaml.safe_dump(worker(standard)).replace(standard["metadata"]["name"], "JOBSET"),
            yaml.safe_dump(worker(native)).replace(native["metadata"]["name"], "JOBSET"))

  def test_bad_identity_is_rejected(self):
    for kwargs in (dict(high_performance=True), dict(disable_eval=True),
                   dict(workload_candidate="m15", data_split="selection"), dict(run_kind="calibration")):
      with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
        self.render(Path(tmp), **kwargs)

  def test_manifest_rejects_wrong_source_and_geometry(self):
    for flag, replacement in (("--old_logps_source=trainer", ""),
                              ("--sampler_is=none", "--sampler_is=token"),
                              ("--mesh_dp=8", "--mesh_dp=4"),
                              ("--batch_size=32", "--batch_size=16")):
      with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
        path, = self.render(Path(tmp))
        document = yaml.safe_load(path.read_text())
        env = VERIFY._container(document)["env"]
        item = next(e for e in env if e["name"] == "CANON_RUN_CMD")
        item["value"] = item["value"].replace(flag, replacement)
        path.write_text(yaml.safe_dump(document))
        with self.assertRaises(ValueError):
          VERIFY.verify(path, wave="standard", workload="p45", source="a" * 40)

  def test_default_three_arms_are_unchanged(self):
    self.assertEqual([arm.name for arm in RENDER._ARMS], ["zero", "mismatch"])


def receipt(step, groups):
  return (f"[P57.TIM_STANDARD] PASS step={step} rows={8 * len(groups)} "
          f"groups={','.join(map(str, groups))} old_logps=trainer tis_weights=absent "
          "rollout_logps=present trainer_rescore=training-input policy_version=matched\n")


class StandardReceiptsTest(unittest.TestCase):
  def test_full_and_microgroup_receipts(self):
    for text in (receipt(0, list(range(32))) + receipt(1, list(range(32, 64))),
                 "".join(receipt(group // 32, [group]) for group in reversed(range(64)))):
      result = RECEIPTS.classify(text, expected_updates=2)
      self.assertEqual(result["trajectories"], 512)

  def test_missing_duplicate_wrong_step_and_wrong_source_are_not_zeros(self):
    good = receipt(0, list(range(32)))
    for bad in ("", good + good, good.replace("step=0", "step=1"),
                good.replace("old_logps=trainer", "old_logps=rollout"),
                good.replace("tis_weights=absent", "tis_weights=present"),
                good.replace("rows=256", "rows=255"),
                good + "[P57.TIM_PURITY] PASS\n", receipt(0, list(range(31)))):
      with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
        RECEIPTS.classify(bad, expected_updates=1)


ALGO = ast.parse((ROOT / "tunix/rl/algo_core.py").read_text())


def loss_slice(entry, current, old, advantage=1., weight=None):
  """Execute production ratio/clipping/TIS statements, not a restated formula."""
  fn = next(n for n in ALGO.body if isinstance(n, ast.FunctionDef) and n.name == entry)
  start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "seq_importance_ratio" for t in n.targets))
  end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "sampler_is_weights is not None")
  scope = dict(jax=jax, jnp=jnp, per_token_logps=current, old_per_token_logps=old,
      completion_mask=jnp.ones_like(current), advantages=jnp.array([advantage]),
      train_example=SimpleNamespace(sampler_is_weights=weight), loss_algo="gspo-token",
      epsilon=.003, epsilon_high=.005, epsilon_c=None,
      loss_aggregation_mode="sequence-mean-token-mean", loss_aggregation_kwargs={},
      common=SimpleNamespace(aggregate_loss=lambda *a, **kw: jnp.array(0.)))
  exec(block(fn.body[start:end + 1]), scope)
  return scope["per_token_loss"].sum()


class StandardMathTest(unittest.TestCase):
  def test_three_treatments_differ_as_intended_at_the_first_update(self):
    current, rollout = jnp.log(jnp.array([[.12]])), jnp.log(jnp.array([[.10]]))
    for entry in ("grpo_loss_fn", "grpo_loss_from_precomputed_logps"):
      for old, weight, expected_loss, expected_grad in (
          (current, None, -1., -1.),
          (current, jnp.array([[1.2]]), -1.2, -1.2),
          (rollout, None, -1.005, 0.),
      ):
        with self.subTest(entry=entry, expected_loss=expected_loss):
          loss, grad = jax.value_and_grad(lambda now: loss_slice(entry, now, old, weight=weight))(current)
          np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
          np.testing.assert_allclose(grad, [[expected_grad]], atol=1e-6)

  def test_standard_gradient_sign_and_frozen_denominator(self):
    old = jnp.array([[-2.]])
    for entry in ("grpo_loss_fn", "grpo_loss_from_precomputed_logps"):
      for advantage in (-1., 0., 1.):
        grad = jax.grad(lambda now: loss_slice(entry, now, old, advantage=advantage))(old)
        np.testing.assert_allclose(grad, [[-advantage]], atol=1e-6)
      # A later optimizer iteration must not reset its ratio to one.
      later = loss_slice(entry, old - .1, old)
      np.testing.assert_allclose(later, -np.exp(-.1), atol=1e-6)


if __name__ == "__main__":
  unittest.main()
