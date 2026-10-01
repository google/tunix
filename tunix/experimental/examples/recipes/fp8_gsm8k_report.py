"""Per-step and mean metrics for the fp8_gsm8k_35b_v5p.sh runs (<prefix>-{bf16,r8,r8t8}), from W&B.

Usage: python fp8_gsm8k_report.py [-v] [--prefix=igfp8v2]   (needs wandb, pandas and WANDB_API_KEY)
"""
import sys
import wandb

P = "train/"
COLS = {
    "step_s": "orchestrator/step_time_sec",
    "gen_s": "orchestrator/exposed_generation_time",
    "train_s": "orchestrator/policy_training_time",
    "sync_s": "orchestrator/weight_sync_time",
    "compl": "rollout/completion_length_mean",
    "seqs": "rollout/global_valid_seqs",
    "oob": "trainer/tis/is_oob_ratio",
    "ld_absmean": "trainer/sampler_is/token_logdiff_absmean",
    "ld_mean": "trainer/sampler_is/token_logdiff_mean",
    "reward": "rewards/mean",
}
api = wandb.Api(timeout=60)
runs = list(api.runs("google-trellis/trellis-gsm8k", order="-created_at", per_page=20))
verbose = "-v" in sys.argv
prefix = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--prefix=")), "igfp8v2")
summary = []
for v in ("bf16", "r8", "r8t8"):
  rs = [r for r in runs if r.name == f"{prefix}-{v}-gsm8k-v5p"]
  if not rs:
    print(f"== {v}: no run"); continue
  r = rs[0]  # newest
  h = r.history(keys=[P + c for c in COLS.values()], pandas=True, samples=1000)
  print(f"== {v}  {r.state}  created {r.created_at}  steps logged: {len(h)}")
  if h.empty:
    continue
  h = h.rename(columns={P + c: k for k, c in COLS.items()})
  h["gen_tok_s"] = h["compl"] * h["seqs"] / h["gen_s"]
  show = ["_step", "step_s", "gen_s", "train_s", "sync_s", "gen_tok_s", "compl", "oob", "ld_absmean", "ld_mean", "reward"]
  show = [c for c in show if c in h]
  if verbose:
    print(h[show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
  # Step 0 carries compilation; report steady state from step 1 on.
  s = h[h["_step"] >= 1] if (h["_step"] >= 1).any() else h
  summary.append({"variant": v, "n": len(s), **{c: s[c].mean() for c in show if c != "_step"}})
if summary:
  import pandas as pd
  print("\nmean over steps >= 1:")
  print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
