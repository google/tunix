#!/usr/bin/env python3
"""Does bf16 rounding flip an MoE router's top-k expert choice with batch shape?

In a dense model, batch-shape-dependent bf16 rounding perturbs log-probs
smoothly. An MoE adds a discrete failure mode the dense model structurally
cannot show: if rounding shifts router logits across a top-k boundary, that
token is routed through DIFFERENT EXPERTS -- not a slightly wrong number, a
different computation.

This scores the same tokens at different forward batch sizes and compares the
routed expert SETS. Slot order is ignored on purpose: the MoE scatter
accumulates both experts' contributions, so permuting the top-k slots is a
no-op (see tests/rl/router_replay_maxtext_test.py).

  python probe_moe_router.py --dtype bfloat16 --precision default
  python probe_moe_router.py --dtype bfloat16 --precision high
  python probe_moe_router.py --dtype float32  --precision highest
"""

import argparse
import contextlib
import os
import sys
import tempfile

import numpy as np

PAD_ID, EOS_ID = 0, 1


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dtype", default="bfloat16",
                  choices=["bfloat16", "float32"])
  ap.add_argument("--precision", default=None,
                  choices=[None, "default", "high", "highest"])
  ap.add_argument("--num_experts", type=int, default=8)
  ap.add_argument("--top_k", type=int, default=2)
  ap.add_argument("--layers", type=int, default=4)
  ap.add_argument("--prompt_len", type=int, default=64)
  ap.add_argument("--completion_len", type=int, default=192)
  ap.add_argument("--n_seq", type=int, default=8)
  args = ap.parse_args()

  import jax
  import jax.numpy as jnp
  from flax import nnx
  from jax.sharding import Mesh
  from maxtext.configs import pyconfig
  from maxtext.integration.tunix import tunix_adapter
  from maxtext.models import models
  from maxtext.utils import maxtext_utils
  from tunix.rl import common

  seq_len = args.prompt_len + args.completion_len
  base_yml = os.path.join(
      os.path.dirname(maxtext_utils.__file__), "..", "configs", "base.yml")
  cfg = pyconfig.initialize(
      [sys.argv[0], base_yml],
      enable_checkpointing=False, log_config=False,
      skip_jax_distributed_system=True, override_model_config=True,
      model_name="qwen3.5-35b-a3b", attention="dot_product",
      num_experts=args.num_experts, num_experts_per_tok=args.top_k,
      base_emb_dim=128, base_num_query_heads=2, base_num_kv_heads=2,
      head_dim=128, use_mrope=False, partial_rotary_factor=0.25,
      base_mlp_dim=128, base_moe_mlp_dim=128, vocab_size=200,
      max_target_length=seq_len, max_prefill_predict_length=args.prompt_len,
      per_device_batch_size=1.0, run_name="moe_router_probe",
      base_output_directory=os.path.join(tempfile.gettempdir(), "moe_router_probe"),
      base_num_decoder_layers=args.layers, num_decoder_layers=args.layers,
      scan_layers=False, enable_dropout=False,
      weight_dtype=args.dtype, dtype=args.dtype,
  )
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  base = models.Transformer(config=cfg, mesh=mesh, quant=None,
                            model_mode="train", rngs=nnx.Rngs(0))
  model = tunix_adapter.TunixMaxTextAdapter(base, pad_id=PAD_ID)
  graphdef, state = nnx.split(model)

  rng = np.random.default_rng(0)
  prompts = rng.integers(2, 200, size=(args.n_seq, args.prompt_len)).astype(np.int32)
  comps = rng.integers(2, 200, size=(args.n_seq, args.completion_len)).astype(np.int32)

  def routing_and_logps(bs):
    R, L = [], []
    for i in range(0, args.n_seq, bs):
      out = common.compute_per_token_logps(
          graphdef, state,
          prompt_tokens=jnp.asarray(prompts[i:i + bs]),
          completion_tokens=jnp.asarray(comps[i:i + bs]),
          pad_id=PAD_ID, eos_id=EOS_ID, return_routed_experts=True)
      lp, routed = out
      L.append(np.asarray(jax.device_get(lp), np.float64))
      R.append(None if routed is None else np.asarray(jax.device_get(routed)))
    if any(r is None for r in R):
      return np.concatenate(L, 0), None
    return np.concatenate(L, 0), np.concatenate(R, 0)

  prec = (jax.default_matmul_precision(args.precision)
          if args.precision else contextlib.nullcontext())
  print(f"dtype={args.dtype} precision={args.precision} "
        f"experts={args.num_experts} top_k={args.top_k} layers={args.layers} "
        f"seq={seq_len} n={args.n_seq} devices={jax.device_count()}")

  with prec, mesh:
    lp1, r1 = routing_and_logps(1)
    lp4, r4 = routing_and_logps(4)

  if r1 is None or r4 is None:
    # TunixMaxTextAdapter.__call__ is typed -> Tuple[Array, None] and returns
    # `logits, None`: it accepts forced_routed_experts (replay in) but never
    # reports the routing it used (capture out). So flips cannot be counted
    # directly. Fall back to the fingerprint: a flip sends a token through
    # different experts, so |d logp| jumps to O(0.1-1) instead of the smooth
    # ~0.03 of dense bf16 rounding. A heavy tail is the tell.
    print("  routed experts NOT returned (adapter returns None) -- "
          "using the |d logp| distribution as an indirect flip detector")
    mask = (comps != PAD_ID).astype(np.float64)
    d = (np.abs(lp1 - lp4) * mask).ravel()
    d = d[mask.ravel() > 0]
    qs = [50, 90, 99, 99.9]
    print(f"  n scored tokens     : {d.size}")
    print(f"  mean |d|            : {d.mean():.8f}")
    for q in qs:
      print(f"  p{q:<6}            : {np.percentile(d, q):.8f}")
    print(f"  max |d|             : {d.max():.8f}")
    print(f"  max / mean          : {d.max()/max(d.mean(),1e-12):.1f}x"
          f"   (dense bf16 reference ~ 20-50x; a router flip should blow this up)")
    for t in (0.05, 0.2, 0.5, 1.0):
      print(f"  frac |d| > {t:<5}    : {(d > t).mean():.6f}")
    return
  print(f"  routed_experts shape {r1.shape}  (batch, token, layer, top_k)")

  # Compare expert SETS per (sequence, token, layer); slot order is irrelevant.
  s1, s4 = np.sort(r1, axis=-1), np.sort(r4, axis=-1)
  differ = (s1 != s4).any(axis=-1)
  n_slots = differ.size
  n_flip = int(differ.sum())
  print(f"  router positions compared : {n_slots}")
  print(f"  positions whose expert SET changed with batch size: "
        f"{n_flip}  ({100.0 * n_flip / max(n_slots,1):.4f}%)")
  if n_flip:
    per_layer = differ.reshape(-1, differ.shape[-1]).mean(0)
    print(f"  per-layer flip rate: "
          f"{np.array2string(100*per_layer, precision=3)} %")
  # How much do the log-probs move, for context against the dense result.
  mask = (comps != PAD_ID).astype(np.float64)
  d = np.abs(lp1 - lp4) * mask
  print(f"  |logp(bs=1) - logp(bs=4)| pooled = {d.sum()/max(mask.sum(),1):.8f}")


if __name__ == "__main__":
  main()
