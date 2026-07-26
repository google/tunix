"""Phase 2 CPU gate tests for the logp-diff / attribution harness.

Known-answer tests that PROVE the harness/attribution logic is correct — so when
Phase 3 feeds real Qwen3-32B activations, we trust the "which op diverged" verdict.

Run: python3 logp_diff_test.py   (plain asserts; no pytest dep needed)
"""
import numpy as np
import toy_qwen3 as tq
import harness as H

CFG = tq.ToyConfig()
W = tq.init_weights(CFG, seed=0)
TOKENS = np.arange(24) % CFG.vocab  # a 24-token toy sequence


def test_self_diff_zero():
  """C vs C (same model, same input): logp diff == 0 and every activation diff == 0."""
  la, aa = tq.forward(W, CFG, TOKENS)
  lb, ab = tq.forward(W, CFG, TOKENS)
  lp_a, lp_b = H.per_token_logp(la, TOKENS), H.per_token_logp(lb, TOKENS)
  s = H.logp_diff_stats(lp_a, lp_b)
  assert s["max"] == 0.0, s
  attr = H.activation_diff(aa, ab)
  assert attr["first_divergence"] is None, attr
  assert all(md == 0.0 for _, md in attr["per_op"]), attr["per_op"]
  print("  [self_diff_zero] PASS  logp max diff=0, all activations bitwise equal")


def test_perturbed_rmsnorm_caught():
  """Inject a bad RMSNorm at layer 1 -> attribution must point to L1.attn_norm FIRST."""
  _, clean = tq.forward(W, CFG, TOKENS)
  _, bad = tq.forward(W, CFG, TOKENS, perturb_op="rms_norm", perturb_layer=1)
  attr = H.activation_diff(clean, bad, atol=1e-9)
  assert attr["first_divergence"] == "L1.attn_norm", attr["first_divergence"]
  # everything before L1.attn_norm must be identical (0 diff)
  for name, md in attr["per_op"]:
    if name == "L1.attn_norm":
      break
    assert md == 0.0, (name, md)
  print(f"  [perturbed_rmsnorm] PASS  first divergence = {attr['first_divergence']} (correct)")


def test_perturbed_attention_caught():
  """Inject a bad attention at layer 0 -> first divergence must be L0.attn."""
  _, clean = tq.forward(W, CFG, TOKENS)
  _, bad = tq.forward(W, CFG, TOKENS, perturb_op="attention", perturb_layer=0)
  attr = H.activation_diff(clean, bad, atol=1e-9)
  assert attr["first_divergence"] == "L0.attn", attr["first_divergence"]
  print(f"  [perturbed_attention] PASS  first divergence = {attr['first_divergence']} (correct)")


def test_determinism():
  """Toy forward is deterministic: two runs bitwise identical."""
  ok, d = H.determinism_check(lambda t: tq.forward(W, CFG, t)[0], TOKENS)
  assert ok and d == 0.0, (ok, d)
  print("  [determinism] PASS  double-run bitwise identical")


def test_op_isolation():
  """Standalone op isolation: same impl -> 0; perturbed -> >0 (RMSNorm example)."""
  x = np.random.default_rng(1).standard_normal((8, CFG.hidden)).astype(np.float32)
  w = np.ones(CFG.hidden, np.float32)
  same = H.op_isolation(lambda a: tq.rms_norm(a, w, CFG.norm_eps),
                        lambda a: tq.rms_norm(a, w, CFG.norm_eps), x)
  pert = H.op_isolation(lambda a: tq.rms_norm(a, w, CFG.norm_eps),
                        lambda a: tq.rms_norm(a, w, CFG.norm_eps) + 1e-2, x)
  assert same == 0.0 and pert > 0.0, (same, pert)
  print(f"  [op_isolation] PASS  same-impl diff={same}, perturbed diff={pert:.3g}")


def test_report_shape():
  """build_report assembles the structured report the remote probe will reuse."""
  la, aa = tq.forward(W, CFG, TOKENS)
  _, bad = tq.forward(W, CFG, TOKENS, perturb_op="mlp", perturb_layer=0)
  rep = H.build_report(
      H.logp_diff_stats(H.per_token_logp(la, TOKENS), H.per_token_logp(la, TOKENS)),
      H.activation_diff(aa, bad, atol=1e-9))
  assert set(rep) >= {"logp_diff", "attribution"}
  assert rep["attribution"]["first_divergence"] == "L0.mlp"
  print("  [report_shape] PASS  report fields ok; mlp-perturb -> first divergence = L0.mlp")


if __name__ == "__main__":
  tests = [test_self_diff_zero, test_perturbed_rmsnorm_caught,
           test_perturbed_attention_caught, test_determinism,
           test_op_isolation, test_report_shape]
  print(f"Running {len(tests)} CPU gate tests for logp-diff harness:")
  for t in tests:
    t()
  print(f"\nALL {len(tests)} PASS")
