"""Pinned-image CPU probe of real FrozenLake geometry admission readers.

No model load, rollout, optimizer commit, cluster connection or remote I/O.
Run in the production image; host-only tests deliberately do not stub these
Tunix imports and call that an installed-runtime certification.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


host = load("t9g_host", Path(__file__).with_name("test_training_geometry.py"))
full = load("t9g_full", ROOT / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/classify_full_recipe.py")
from examples.frozenlake import training_geometry as geometry
from tunix.rl import alignment, canonical_qwen3_adapter, dp_training, dp_workloads, frozenlake_checkpoint
from tunix.rl.agentic import agentic_grpo_learner, token_continuity
from tunix.sft import utils as sft_utils


def main():
  if len(jax.devices()) != 32 or any(device.platform != "cpu" for device in jax.devices()):
    raise RuntimeError("requires exactly 32 forced CPU devices; no TPU admission")
  mesh = Mesh(np.asarray(jax.devices()).reshape(4, 8), ("data", "model"))
  template = jax.device_put(jnp.zeros((32,), jnp.float32), NamedSharding(mesh, P("model")))
  staged_host = np.stack([np.arange(32, dtype=np.float32) + 10.0 * rank for rank in range(4)])
  staged = jax.device_put(staged_host, NamedSharding(mesh, P("data", "model")))
  reducer = dp_training.FixedDPRankGradientReducer(template, dp_size=4, dp_axis="data")
  reduced, report = reducer.finalize_staged(staged)
  np.testing.assert_array_equal(np.asarray(reduced), staged_host.sum(axis=0))
  assert report["dp_size"] == 4 and report["reduction_rounds"] == 4
  assert report["post_reduction_all_finite"] and report["post_reduction_replicas_exact"]
  harness = host.TrainingGeometryTest()
  cases = 0
  for candidate in ("", "m15"):
    for mode in (geometry.LEGACY, geometry.SMALL):
      for debug in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
          path = harness._render(Path(tmp) / "render", candidate, mode)
          env = host.p57._env(host.yaml.safe_load(path.read_text()))
          if debug:
            env.update(CANON_P57_TOKEN_CONTINUITY="exact",
                       CANON_P57_TOKEN_CONTINUITY_DEBUG="record-full")
          state = Path(tmp) / "state"
          result = harness._preflight(env, state)
          if result.returncode:
            # Do not dump env.sh or unrelated inherited environment values.
            raise AssertionError("00_env failed: " + result.stderr)
          resolved = full._resolved_env(state / "env.sh")
          # env.sh deliberately excludes credentials. Test presence only.
          resolved["WANDB_API_KEY"] = "test-only"
          geom = geometry.geometry(mode)
          for key, expected in geom.environment().items():
            assert resolved[key] == expected, key
          workload = dp_workloads.active_workload(resolved)
          dp_workloads.validate_environment(workload, resolved, require_reduction_admission=True)
          assert dp_workloads.requested_max_steps(workload, resolved) == 300
          assert dp_workloads.expected_token_widths(workload, resolved) == (4096, 8192 if candidate else 2048)
          dp_workloads.validate_frozenlake_max_concurrency(workload, geom.trajectories, resolved)
          assert sft_utils.canonical_overflow_safe_clip_max_norm(resolved) == 100.0
          frozenlake_checkpoint.require_p57_fast_no_checkpoint(
              frozenlake_checkpoint.from_env(resolved), resolved)
          assert agentic_grpo_learner._p57_tim_purity_enabled(resolved)
          groups = workload.training_contract().rank_major_reverse_groups()
          assert len(groups) == 32 and all(len(group) == geom.dp for group in groups)
          np.testing.assert_array_equal(np.sort(np.asarray(groups).ravel()), np.arange(geom.trajectories))
          # Averaging per-group rank means then the 32-group accumulator has
          # exactly the same denominator as averaging all trajectories.
          values = np.arange(geom.trajectories, dtype=np.float64)
          assert np.mean([np.mean(values[list(group)]) for group in groups]) == np.mean(values)
          with mock.patch.dict(os.environ, resolved, clear=True):
            assert canonical_qwen3_adapter._canonical_topology_contract() == (geom.dp, 8, 256, geom.global_m)
            assert alignment.gsm8k_ab_report_policy()["warning_boundaries"] == ("S_decode_vs_S_prefill",)
          if debug:
            assert token_continuity.frozenlake_token_continuity(resolved) is not None
            assert token_continuity.frozenlake_token_continuity_debug_mode(resolved) == "record-full"
          if mode == geometry.SMALL:
            for key, wrong in (("CANON_DP_SIZE", "8"), ("CANON_GLOBAL_TRAJECTORIES", "256"),
                               ("MIN_TOKEN_BUCKET", "2048"), ("CANON_P59_CHECKED_VMA", "0")):
              try:
                dp_workloads.validate_environment(workload, {**resolved, key: wrong}, require_reduction_admission=True)
              except ValueError:
                pass
              else:
                raise AssertionError("negative not rejected: " + key)
          cases += 1
  print("P57_TRAIN_GEOMETRY_RUNTIME_PASS cases=" + str(cases) + " dp4tp8_cpu_reducer=1 target=not-run")


if __name__ == "__main__":
  main()
