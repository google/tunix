"""Is the canonical forward point biased?  fp64 forward logps (no backward)
vs the captured canonical logps (A) and the stock TPU logps (C): signed mean
difference over action tokens, per row and overall."""
import sys, os, json, pathlib, importlib.util, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/mnt/disks/tunix-data/worktrees/v2_dispatch_0906")
spec = importlib.util.spec_from_file_location("ref", "/mnt/disks/tunix-data/worktrees/v2_dispatch_p9/canon-zero-tim/tests/p61_backward/fp64_reference.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
ref.widen_float32_to_float64() if hasattr(ref, "widen_float32_to_float64") else None
import jax, jax.numpy as jnp, numpy as np
from tunix.models.qwen3 import model as qm
A = pathlib.Path("/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2disp_p13_cap_lane_20260907_r1/train/p61_numerical")
C = pathlib.Path("/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2disp_p13b_stock_dp2_20260907_r1/train/p61_numerical")
G = pathlib.Path("/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-v2p0g4cand_20260903_r4/train/p61_numerical")
config = qm.ModelConfig.qwen3_1p7b()
params = ref.load_capture(G, "model_before")
example = ref.load_example(A)
algo, pad_id, eos_id = ref.load_algo_config(A, 1.0)
graphdef, state = ref.build_model(config, params); del params
def fwd(state, rows):
  logps, _ = ref.common.compute_per_token_logps(graphdef, state, prompt_tokens=rows.prompt_ids, completion_tokens=rows.completion_ids, pad_id=pad_id, eos_id=eos_id, stop_gradient=True, return_entropy=True, temperature=1.0, canonical_actor=False, prompt_mask=rows.prompt_mask, completion_mask=rows.completion_mask)
  return logps
fwd = jax.jit(fwd)
out = []
t0 = time.perf_counter()
for start in range(0, 64, 8):
  out.append(np.asarray(fwd(state, ref.example_rows(example, slice(start, start + 8)))))
  print(f"rows {start}..{start+7} {time.perf_counter()-t0:.0f}s", flush=True)
ref64 = np.concatenate(out, 0).astype(np.float64)
mask = np.asarray(example.completion_mask).astype(bool)
a = np.asarray(ref.load_capture(A, "logps")[""], np.float64)
c = np.asarray(ref.load_capture(C, "stock_logps")[""], np.float64)
def stats(name, x):
  d = (x - ref64)[mask]
  print(f"{name}: signed mean={d.mean():+.5f} mean|d|={np.abs(d).mean():.5f} max|d|={np.abs(d).max():.4f} median={np.median(d):+.5f} frac_negative={np.mean(d<0):.3f} n={mask.sum()}")
stats("A (canonical) - fp64", a)
stats("C (stock TPU) - fp64", c)
d = (a - c)[mask]; print(f"A - C: signed mean={d.mean():+.5f} mean|d|={np.abs(d).mean():.5f}")
# bias vs position: early vs late tokens
pos = np.broadcast_to(np.arange(mask.shape[1])[None, :], mask.shape)
for lo, hi in ((0, 128), (128, 256), (256, 512), (512, 2048)):
  sel = mask & (pos >= lo) & (pos < hi)
  if sel.sum():
    print(f"  completion positions [{lo},{hi}): A-fp64 mean={((a-ref64)[sel]).mean():+.5f}  C-fp64 mean={((c-ref64)[sel]).mean():+.5f} n={sel.sum()}")
