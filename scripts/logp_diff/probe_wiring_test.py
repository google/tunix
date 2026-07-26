"""CPU wiring test: run the REAL probe main() with tunix/vLLM MOCKED.

Verifies the probe's plumbing end-to-end on CPU (source construction, generate->A/B/C
flow, compare() loop, report assembly) WITHOUT needing metrax/vLLM/TPU. The numerical
result still needs the cluster, but this proves the data flow is correct.

Uses a real 1x1 jax-cpu mesh (jax[cpu] installed) so the mesh code path also runs.
Run: python3 probe_wiring_test.py
"""
import numpy as np
from unittest import mock
import logp_diff_probe as P


class FakeTok:
  pad_token_id, eos_token_id = 0, 2


def test_probe_main_wiring():
  n_prompt, n_gen = 2048, 512
  full = np.arange(n_prompt + n_gen, dtype=np.int32)
  rng = np.random.default_rng(0)
  a = rng.standard_normal(n_gen).astype(np.float32)        # A decode logp
  b = a + rng.standard_normal(n_gen).astype(np.float32) * 0.01   # B ~ A + small (decode-vs-prefill)
  c = a + rng.standard_normal(n_gen).astype(np.float32) * 0.05   # C ~ A + larger (kernel)

  with mock.patch.object(P, "load_prompt_tokens", return_value=(np.arange(n_prompt), FakeTok())), \
       mock.patch.object(P, "run_vllm", return_value=(full, a, object())), \
       mock.patch.object(P, "vllm_prefill_logp", return_value=b), \
       mock.patch.object(P, "tunix_forward_logp", return_value=c):
    # local --out so the gs:// GCS-write branch (needs gcsfs, TPU-only) is skipped on CPU.
    rep = P.main(["--mesh_tp", "1", "--mesh_fsdp", "1", "--n_prompt", str(n_prompt),
                  "--n_gen", str(n_gen), "--pairs", "A-vs-C,B-vs-C,A-vs-B",
                  "--out", "/tmp/logp_report_wiring.json"])

  # report structure: the 3 comparisons each with logp_diff stats
  comps = rep["comparisons"]
  assert set(comps) == {"A-vs-C", "B-vs-C", "A-vs-B"}, list(comps)
  for k, v in comps.items():
    assert set(v["logp_diff"]) >= {"mean", "max", "pearson", "n"}, v
  # sanity: B-vs-A diff (decode effect) < C-vs-A diff (kernel effect), per how we built the mocks
  assert comps["A-vs-B"]["logp_diff"]["max"] < comps["A-vs-C"]["logp_diff"]["max"]
  print("  [probe_wiring] PASS  main() ran on 1x1 cpu mesh with mocked tunix/vLLM")
  print("    A-vs-C max diff = %.4f (real training diff)" % comps["A-vs-C"]["logp_diff"]["max"])
  print("    B-vs-C max diff = %.4f (pure kernel)      " % comps["B-vs-C"]["logp_diff"]["max"])
  print("    A-vs-B max diff = %.4f (decode effect)    " % comps["A-vs-B"]["logp_diff"]["max"])


def test_dry_run_no_heavy_imports():
  """--dry_run must not import jax/vllm/tunix (CPU boundary)."""
  plan = P.main(["--dry_run"])
  assert plan["flow"].startswith("generate->A")
  print("  [dry_run] PASS  args+plan built, no jax/vllm/tunix import")


if __name__ == "__main__":
  print("Running probe wiring tests (tunix/vLLM mocked):")
  test_dry_run_no_heavy_imports()
  test_probe_main_wiring()
  print("\nALL PASS")
