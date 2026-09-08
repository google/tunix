"""Re-pin rule check (tasks/v2_dispatch/GOAL.md, gradient side) from a four-way
fp64_reference.py metrics JSON: A = certified capture, B = new program's capture.
Rule 1 (fp64 equidistance, A as the reference):
  overall rel_l2(B,64) <= 1.05 * rel_l2(A,64); one_minus_cos likewise;
  per leaf kind rel_l2 ratio <= 1.05; per leaf group rel_l2 and one_minus_cos
  ratios <= 1.10; rel_l2(B,A) <= 2 * max(rel_l2(A,64), rel_l2(B,64)).
Rule 2 (guard rails): B<->A overall one_minus_cos <= 1e-3 and every group's
  norm ratio B/A within [0.98, 1.02].
"""
import json, math, re, sys, collections
path = sys.argv[1]
d = json.load(open(path))
a64, b64, ab = d["a_vs_fp64"], d["b_vs_fp64"], d["a_vs_b"]
fails = []
def check(name, ok, detail):
  print(("PASS" if ok else "FAIL"), name, detail)
  if not ok: fails.append(name)
oa, ob, oab = a64["overall"], b64["overall"], ab["overall"]
check("overall rel_l2 ratio <= 1.05", ob["rel_l2"] <= 1.05 * oa["rel_l2"], f"B {ob['rel_l2']:.5f} vs 1.05*A {1.05*oa['rel_l2']:.5f}")
check("overall one_minus_cos ratio <= 1.05", ob["one_minus_cos"] <= 1.05 * oa["one_minus_cos"], f"B {ob['one_minus_cos']:.3e} vs 1.05*A {1.05*oa['one_minus_cos']:.3e}")
check("rel_l2(B,A) <= 2*max(A64,B64)", oab["rel_l2"] <= 2 * max(oa["rel_l2"], ob["rel_l2"]), f"B<->A {oab['rel_l2']:.3e} vs {2*max(oa['rel_l2'], ob['rel_l2']):.3e}")
def kind_of(group):
  if group.startswith("layer["): return re.sub(r"layer\[\d+\]\.", "", group)
  return group
def per_kind(block):
  acc = collections.defaultdict(lambda: [0.0, 0.0])
  for g, m in block["groups"].items():
    k = kind_of(g); acc[k][0] += m["diff_norm"] ** 2; acc[k][1] += m["ref_norm"] ** 2
  return {k: math.sqrt(v[0]) / math.sqrt(v[1]) for k, v in acc.items() if v[1] > 0}
ka, kb = per_kind(a64), per_kind(b64)
worst = max((kb[k] / ka[k], k) for k in ka)
check("per-kind rel_l2 ratio <= 1.05", worst[0] <= 1.05, f"worst {worst[1]} ratio {worst[0]:.4f}; " + " ".join(f"{k}:{kb[k]/ka[k]:.3f}" for k in sorted(ka)))
worst_g = max(((b64["groups"][g]["rel_l2"] / a64["groups"][g]["rel_l2"], g) for g in a64["groups"] if a64["groups"][g]["rel_l2"] > 0))
worst_c = max(((b64["groups"][g]["one_minus_cos"] / a64["groups"][g]["one_minus_cos"], g) for g in a64["groups"] if a64["groups"][g]["one_minus_cos"] > 0))
check("per-group rel_l2 ratio <= 1.10", worst_g[0] <= 1.10, f"worst {worst_g[1]} {worst_g[0]:.4f}")
check("per-group one_minus_cos ratio <= 1.10", worst_c[0] <= 1.10, f"worst {worst_c[1]} {worst_c[0]:.4f}")
check("guard: B<->A overall one_minus_cos <= 1e-3", oab["one_minus_cos"] <= 1e-3, f"{oab['one_minus_cos']:.3e}")
ratios = {g: m["got_norm"] / m["ref_norm"] for g, m in ab["groups"].items() if m["ref_norm"] > 0}
lo, hi = min(ratios.items(), key=lambda kv: kv[1]), max(ratios.items(), key=lambda kv: kv[1])
check("guard: per-group norm ratio B/A in [0.98, 1.02]", lo[1] >= 0.98 and hi[1] <= 1.02, f"min {lo[0]} {lo[1]:.4f}, max {hi[0]} {hi[1]:.4f}")
print("VERDICT:", "RE-PIN ADMISSIBLE" if not fails else "RED: " + ", ".join(fails))
