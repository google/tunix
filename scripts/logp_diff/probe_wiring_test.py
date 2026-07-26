"""CPU wiring test: run the REAL probe main() with tunix/vLLM MOCKED, DISAGGREGATED.

Verifies the probe's plumbing end-to-end on CPU (disjoint two-mesh construction, generate
-> A/B/C flow, cached single-eval per source, decomposition + additivity guard, report
assembly) WITHOUT needing metrax/vLLM/TPU. The numerical result still needs the cluster,
but this proves the data flow + the disaggregated mesh build are correct.

Forces 8 CPU jax devices so build_meshes' real two-mesh (rollout devices[:R], train
devices[R:R+T]) code path runs with rollout(1,1)+train(1,1) = 2 disjoint devices.
Run: python3 probe_wiring_test.py
"""
import os
# MUST precede any jax import (build_meshes imports jax lazily inside P.main).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np
from unittest import mock
import logp_diff_probe as P


class FakeTok:
  pad_token_id, eos_token_id = 0, 2


def _mock_logps(n_prompt, n_gen):
  full = np.arange(n_prompt + n_gen, dtype=np.int32)
  rng = np.random.default_rng(0)
  a = rng.standard_normal(n_gen).astype(np.float32)              # A decode logp
  b = a + rng.standard_normal(n_gen).astype(np.float32) * 0.01   # B ~ A + small (decode effect)
  c = a + rng.standard_normal(n_gen).astype(np.float32) * 0.05   # C ~ A + larger (kernel+mesh)
  return full, a, b, c


def test_probe_main_wiring():
  n_prompt, n_gen = 2048, 512
  full, a, b, c = _mock_logps(n_prompt, n_gen)

  with mock.patch.object(P, "load_prompt_tokens", return_value=(np.arange(n_prompt), FakeTok())), \
       mock.patch.object(P, "run_vllm", return_value=(full, a, object())), \
       mock.patch.object(P, "vllm_prefill_logp", return_value=b), \
       mock.patch.object(P, "tunix_forward_logp", return_value=c):
    rep = P.main(["--rollout_mesh_fsdp", "1", "--rollout_mesh_tp", "1",
                  "--train_mesh_fsdp", "1", "--train_mesh_tp", "1",
                  "--n_prompt", str(n_prompt), "--n_gen", str(n_gen),
                  "--pairs", "A-vs-C,A-vs-B,B-vs-C",
                  "--out", "/tmp/logp_report_wiring.json"])

  comps = rep["comparisons"]
  assert set(comps) == {"A-vs-C", "A-vs-B", "B-vs-C"}, list(comps)
  for k, v in comps.items():
    assert set(v) >= {"mean", "max", "pearson", "n"}, v

  dec = rep["decomposition"]
  # additivity is EXACT per token (A-C == (A-B)+(B-C)) -> residual ~ float noise.
  assert dec["additivity_residual_max"] < 1e-5, dec["additivity_residual_max"]
  # decode effect (A-vs-B) < real total (A-vs-C), per how the mocks are built.
  assert dec["decode(A-vs-B)"]["max"] < dec["real_total(A-vs-C)"]["max"]
  assert dec["lengths"]["A"] == dec["lengths"]["B"] == dec["lengths"]["C"] == n_gen
  print("  [probe_wiring] PASS  disaggregated main() ran on 8-CPU-device 2-mesh, mocked tunix/vLLM")
  print("    real_total(A-vs-C) max = %.4f" % dec["real_total(A-vs-C)"]["max"])
  print("    decode(A-vs-B)     max = %.4f" % dec["decode(A-vs-B)"]["max"])
  print("    kernel+mesh(B-vs-C) max = %.4f" % dec["kernel+mesh(B-vs-C)"]["max"])
  print("    additivity_residual_max = %.2e (align guard)" % dec["additivity_residual_max"])


def test_mesh_sensitivity_rung():
  """--mesh_sensitivity carves a C2 sub-mesh from train chips and adds C-vs-C2."""
  n_prompt, n_gen = 256, 64
  full, a, b, c = _mock_logps(n_prompt, n_gen)
  c2 = c + np.random.default_rng(1).standard_normal(n_gen).astype(np.float32) * 0.02

  with mock.patch.object(P, "load_prompt_tokens", return_value=(np.arange(n_prompt), FakeTok())), \
       mock.patch.object(P, "run_vllm", return_value=(full, a, object())), \
       mock.patch.object(P, "vllm_prefill_logp", return_value=b), \
       mock.patch.object(P, "tunix_forward_logp", side_effect=[c, c2]):
    rep = P.main(["--rollout_mesh_fsdp", "1", "--rollout_mesh_tp", "1",
                  "--train_mesh_fsdp", "1", "--train_mesh_tp", "1",
                  "--n_prompt", str(n_prompt), "--n_gen", str(n_gen),
                  "--mesh_sensitivity", "--pairs", "A-vs-C",
                  "--out", "/tmp/logp_report_wiring2.json"])
  dec = rep["decomposition"]
  assert "tunix_sharding_sensitivity(C-vs-C2)" in dec, list(dec)
  assert dec["lengths"]["C2"] == n_gen
  print("  [mesh_sensitivity] PASS  C2 carved from train sub-mesh, C-vs-C2 max = %.4f"
        % dec["tunix_sharding_sensitivity(C-vs-C2)"]["max"])


def test_dry_run_no_heavy_imports():
  """--dry_run must not import jax/vllm/tunix (CPU boundary)."""
  plan = P.main(["--dry_run"])
  assert plan["flow"].startswith("generate->A")
  assert plan["rollout_mesh"] and plan["train_mesh"]
  print("  [dry_run] PASS  args+plan built (rollout+train mesh), no jax/vllm/tunix import")


if __name__ == "__main__":
  print("Running probe wiring tests (disaggregated, tunix/vLLM mocked):")
  test_dry_run_no_heavy_imports()
  test_probe_main_wiring()
  test_mesh_sensitivity_rung()
  print("\nALL PASS")
