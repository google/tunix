#!/usr/bin/env python3
"""Isolate whether trainer log-probs depend on the forward pass's batch size.

No orchestrator, no rollout, no GRPO. Load the model once, take a fixed set of
token sequences, and score the SAME sequence alone and as part of a larger
batch. Any difference is the trainer disagreeing with itself.

  python probe_batch_shape.py --model_dir ... --model_name Qwen3-1.7B --tp 2

Tests, each independently switchable:
  --test logps    per-token log-probs at batch 1 vs 2 vs 4          (reproduce)
  --test same     batch of N identical copies vs batch 1            (control)
  --test split    hidden states vs logits, to locate the divergence (bisect)

Nothing here writes to the repo.
"""

import argparse
import contextlib
import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax                                          # noqa: E402
import jax.numpy as jnp                             # noqa: E402
import numpy as np                                  # noqa: E402
from flax import nnx                                # noqa: E402
from jax.experimental import mesh_utils             # noqa: E402
from jax.sharding import Mesh                       # noqa: E402


def build_mesh(fsdp, tp):
  shape = (fsdp, tp)
  if fsdp * tp != jax.device_count():
    raise SystemExit(
        f"mesh {shape} needs {fsdp * tp} devices, have {jax.device_count()}"
    )
  return Mesh(mesh_utils.create_device_mesh(shape, jax.devices()),
              axis_names=("fsdp", "tp"))


def make_tokens(tokenizer, n_seq, prompt_len, completion_len, seed=0):
  """Realistic-ish token ids: real text, deterministically padded/truncated."""
  rng = np.random.default_rng(seed)
  prompts, comps = [], []
  base = [
      "Natalia sold clips to 48 friends in April, and then half as many in May."
      " How many clips did she sell altogether?",
      "Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes."
      " How much did she earn?",
      "Betty is saving for a wallet that costs $100. She has half. Her parents"
      " give her $15 and her grandparents twice as much. How much more?",
      "James writes a 3-page letter to 2 friends twice a week. How many pages"
      " does he write a year?",
  ]
  for i in range(n_seq):
    text = base[i % len(base)] + f" (variant {i})"
    ids = tokenizer.encode(text)
    if not isinstance(ids, list):
      ids = list(ids)
    p = (ids * ((prompt_len // max(len(ids), 1)) + 1))[:prompt_len]
    # Completions are the part that gets scored; vary their content and the
    # real (unpadded) length the way rollouts do.
    real = int(rng.integers(110, min(200, completion_len)))
    c = list(rng.integers(1000, 40000, size=real))
    c = c + [tokenizer.pad_id] * (completion_len - real)
    prompts.append(p)
    comps.append(c)
  return np.array(prompts, np.int32), np.array(comps, np.int32)


def scored_mask(completion_tokens, pad_id):
  return (completion_tokens != pad_id).astype(np.float32)


LAST_TIMING = {}


def run_logps(common, graphdef, state, prompts, comps, pad_id, eos_id, bs,
              chunk_size=0):
  """Scores rows in groups of `bs`; returns [N, C] log-probs."""
  out = []
  n = prompts.shape[0]
  for i in range(0, n, bs):
    p = jnp.asarray(prompts[i:i + bs])
    c = jnp.asarray(comps[i:i + bs])
    lp = common.compute_per_token_logps(
        graphdef, state, p, c, pad_id=pad_id, eos_id=eos_id,
        chunk_size=chunk_size)
    out.append(np.asarray(jax.device_get(lp), np.float64))
  return np.concatenate(out, 0)


def time_forward(common, graphdef, state, prompts, comps, pad_id, eos_id, bs,
                 reps=5):
  """Wall-clock per forward, after warmup so compilation is excluded."""
  import time
  p = jnp.asarray(prompts[:bs])
  c = jnp.asarray(comps[:bs])
  r = common.compute_per_token_logps(graphdef, state, p, c,
                                     pad_id=pad_id, eos_id=eos_id)
  jax.block_until_ready(r)
  t0 = time.perf_counter()
  for _ in range(reps):
    r = common.compute_per_token_logps(graphdef, state, p, c,
                                       pad_id=pad_id, eos_id=eos_id)
    jax.block_until_ready(r)
  dt = (time.perf_counter() - t0) / reps
  print(f"  forward bs={bs}: {dt * 1000:.1f} ms  ({dt * 1000 / bs:.1f} ms/seq)")
  return dt


def summarize(tag, a, b, mask):
  """Mean |a-b| over scored tokens, plus what the TIS gate actually sees.

  The gate uses seq_geomean = exp(mean_t(delta)), so signed per-token errors
  cancel within a sequence. |delta| is the wrong statistic for judging gate
  impact; the signed per-sequence mean is the right one.
  """
  d = np.abs(a - b) * mask
  per = d.sum(1) / np.maximum(mask.sum(1), 1)
  pooled = d.sum() / max(mask.sum(), 1)
  signed = ((a - b) * mask).sum(1) / np.maximum(mask.sum(1), 1)
  geo_shift = np.exp(signed)
  print(f"  {tag:34s} pooled|d|={pooled:.8f}")
  print(f"  {'':34s}   signed/seq="
        f"{np.array2string(signed, precision=6, max_line_width=200)}")
  print(f"  {'':34s}   seq_geomean shift="
        f"{np.array2string(geo_shift, precision=6, max_line_width=200)}")
  return pooled


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_dir", required=True)
  ap.add_argument("--model_name", default="Qwen3-1.7B")
  ap.add_argument("--tokenizer_path", default=None)
  ap.add_argument("--fsdp", type=int, default=1)
  ap.add_argument("--tp", type=int, default=2)
  ap.add_argument("--n_seq", type=int, default=4)
  ap.add_argument("--prompt_len", type=int, default=512)
  ap.add_argument("--completion_len", type=int, default=768)
  ap.add_argument("--test", default="logps,same",
                  help="comma list of: logps, same, split")
  ap.add_argument("--precision", default=None,
                  choices=[None, "bfloat16", "float32", "tensorfloat32",
                           "highest", "high", "default"],
                  help="jax.default_matmul_precision for the whole probe")
  ap.add_argument("--weights", default=None, choices=[None, "bf16", "fp32"],
                  help="weight storage dtype, independent of activation dtype")
  ap.add_argument("--hp_scope", default=None,
                  choices=[None, "none", "head", "nohead", "all"],
                  help="where to apply fp32 operands + HIGHEST matmul precision")
  ap.add_argument("--model_dtype", default=None, choices=[None, "float32"],
                  help="override the model's activation dtype (default bf16)")
  ap.add_argument("--chunk_size", type=int, default=0,
                  help="compute_per_token_logps chunk_size (reference uses 2048)")
  args = ap.parse_args()

  from tunix.rl import common                       # noqa: E402
  from tunix.experimental.examples.common import models  # noqa: E402
  from transformers import AutoTokenizer            # noqa: E402

  tok_path = args.tokenizer_path or args.model_dir
  hf = AutoTokenizer.from_pretrained(tok_path)
  pad_id = hf.pad_token_id if hf.pad_token_id is not None else 0
  eos_id = hf.eos_token_id if hf.eos_token_id is not None else 1

  class _T:
    def __init__(self, pid):
      self.pad_id = pid
    def encode(self, s): return hf.encode(s, add_special_tokens=False)

  print(f"devices={jax.device_count()}  pad_id={pad_id}  eos_id={eos_id}")
  print(f"precision={args.precision}  chunk_size={args.chunk_size}")
  mesh = build_mesh(args.fsdp, args.tp)
  print(f"mesh={mesh}")

  prec_ctx = (jax.default_matmul_precision(args.precision)
              if args.precision else contextlib.nullcontext())
  # --- selective high precision -------------------------------------------
  # Raising jax precision alone is a no-op when the operands are already bf16,
  # so each scope must ALSO widen its operands. "head" keeps the whole model in
  # bf16 and upgrades only the tied-embedding output projection; "nohead" is the
  # complement, to attribute how much each half contributes.
  if args.hp_scope in ("head", "nohead", "all"):
    from tunix.models.qwen3 import model as q3  # noqa: E402
    _orig_decode = q3.Embedder.decode

    def _fp32_decode(self, x):
      xf = jnp.astype(x, jnp.float32)
      wf = jnp.astype(self.input_embedding.value, jnp.float32)
      return jax.lax.dot_general(
          xf, wf.T, (((xf.ndim - 1,), (0,)), ((), ())),
          precision=jax.lax.Precision.HIGHEST,
          preferred_element_type=jnp.float32)

    def _bf16_decode(self, x):
      xb = jnp.astype(x, jnp.bfloat16)
      wb = jnp.astype(self.input_embedding.value, jnp.bfloat16)
      return jax.lax.dot_general(
          xb, wb.T, (((xb.ndim - 1,), (0,)), ((), ())),
          precision=jax.lax.Precision.DEFAULT)

    if args.hp_scope in ("head", "all"):
      q3.Embedder.decode = _fp32_decode
      print("  output head: fp32 operands + HIGHEST")
    else:  # nohead -- pin the head back down to bf16/DEFAULT
      q3.Embedder.decode = _bf16_decode
      print("  output head: forced bf16 + DEFAULT")
    if args.hp_scope in ("nohead", "all"):
      args.model_dtype = "float32"
      args.precision = "highest"
      print("  everything else: fp32 weights+acts + HIGHEST")
    else:
      print("  everything else: bf16 + DEFAULT (production)")
    prec_ctx = (jax.default_matmul_precision(args.precision)
                if args.precision else contextlib.nullcontext())

  with prec_ctx, mesh:
    if args.model_dtype == "float32":
      # models.create_model hardcodes dtype=jnp.bfloat16 at models.py:76 and the
      # loader casts every tensor to it, so the WEIGHTS are bf16 on device --
      # config.param_dtype only shapes a discarded eval_shape structure. A real
      # fp32 test has to go around create_model and load fp32 from disk.
      from tunix.models.qwen3 import params as qwen3_params_lib  # noqa: E402
      cfg = models._qwen3_config(args.model_name)
      cfg.dtype = jnp.float32                      # activations
      # Weight storage is a separate lever from activation dtype: bf16 weights
      # widened to fp32 still let a 3-pass matmul split the ACTIVATIONS, which
      # may recover most of the accuracy without doubling HBM.
      wdt = jnp.bfloat16 if args.weights == "bf16" else jnp.float32
      cfg.param_dtype = wdt
      model = qwen3_params_lib.create_model_from_safe_tensors(
          args.model_dir, cfg, mesh, dtype=wdt)
      print(f"  weights={wdt.__name__}  activations=float32")
    else:
      model = models.create_model(args.model_name, args.model_dir, mesh)
    graphdef, state = nnx.split(model)

    prompts, comps = make_tokens(
        _T(pad_id), args.n_seq, args.prompt_len, args.completion_len)
    mask = scored_mask(comps, pad_id)
    print(f"scored tokens/seq: {mask.sum(1).astype(int).tolist()}")

    tests = [t.strip() for t in args.test.split(",") if t.strip()]

    if "logps" in tests:
      print("\n=== TEST logps: same sequences, different forward batch size ===")
      ref = run_logps(common, graphdef, state, prompts, comps,
                      pad_id, eos_id, 1, args.chunk_size)
      for bs in (2, 4):
        if args.n_seq % bs:
          continue
        got = run_logps(common, graphdef, state, prompts, comps,
                        pad_id, eos_id, bs, args.chunk_size)
        summarize(f"|logp(bs={bs}) - logp(bs=1)|", got, ref, mask)
      # Self-consistency: rerunning at bs=1 must be bit-identical.
      again = run_logps(common, graphdef, state, prompts, comps,
                        pad_id, eos_id, 1, args.chunk_size)
      d = float(np.abs(again - ref).max())
      print(f"  rerun at bs=1 max|diff| = {d:.3e}"
            f"  {'(deterministic)' if d == 0 else '(NONDETERMINISTIC!)'}")

    if "same" in tests:
      print("\n=== TEST same: N identical copies vs batch 1 ===")
      one_p = prompts[:1]
      one_c = comps[:1]
      one_m = mask[:1]
      ref = run_logps(common, graphdef, state, one_p, one_c,
                      pad_id, eos_id, 1)
      for bs in (2, 4):
        rep_p = np.repeat(one_p, bs, 0)
        rep_c = np.repeat(one_c, bs, 0)
        got = run_logps(common, graphdef, state, rep_p, rep_c,
                        pad_id, eos_id, bs)
        # every row is the same sequence; compare each against the bs=1 answer
        for r in range(bs):
          summarize(f"identical copies bs={bs} row{r}",
                    got[r:r + 1], ref, one_m)

    if "bench" in tests:
      print("\n=== TEST bench: wall-clock cost of this configuration ===")
      from flax import nnx as _nx
      _gd, _st = _nx.split(model)
      for _bs in (1, 4):
        time_forward(common, _gd, _st, prompts, comps, pad_id, eos_id, _bs)

    if "accuracy" in tests:
      print("\n=== TEST accuracy: bf16 at each batch size vs fp32 truth ===")
      test_accuracy(common, models, args, mesh, prompts, comps, mask,
                    pad_id, eos_id)

    if "jit" in tests:
      print("\n=== TEST jit: redo the isolations inside jax.jit ===")
      test_jit(common, model, prompts, comps, mask, pad_id, eos_id,
               args.completion_len)

    if "shape" in tests:
      print("\n=== TEST shape: is it batch, or the matmul's M dimension? ===")
      test_shape(common, model, prompts, comps, mask, pad_id, eos_id,
                 args.completion_len)

    if "recon" in tests:
      print("\n=== TEST recon: hidden + model's own head vs full forward ===")
      test_recon(common, model, prompts, comps, mask, pad_id, eos_id,
                 args.completion_len)

    if "truth" in tests:
      print("\n=== TEST truth: which batch size is actually correct? ===")
      test_truth(common, model, prompts, comps, mask, pad_id, eos_id,
                 args.completion_len)

    if "split" in tests:
      print("\n=== TEST split: hidden states vs final logits ===")
      import inspect
      sig = inspect.signature(model.__call__)
      if "skip_lm_head" not in sig.parameters:
        print("  model has no skip_lm_head; cannot bisect here")
      else:
        def hidden(p, c, bs):
          outs = []
          for i in range(0, p.shape[0], bs):
            pi = jnp.asarray(p[i:i + bs])
            ci = jnp.asarray(c[i:i + bs])
            it, pos, am, sid = common.process_ids(
                pi, ci, pad_id, eos_id, None, None)
            kw = {"positions": pos, "cache": None, "attention_mask": am,
                  "skip_lm_head": True}
            if common.model_call_contains(model, "segment_ids") and sid is not None:
              kw["segment_ids"] = sid
            h, _ = model(it, **kw)
            outs.append(np.asarray(jax.device_get(h), np.float64))
          return np.concatenate(outs, 0)

        h1 = hidden(prompts, comps, 1)
        for bs in (2, 4):
          if args.n_seq % bs:
            continue
          hb = hidden(prompts, comps, bs)
          C = args.completion_len
          hm = mask[:, :, None]
          dh = np.abs(hb[:, -C:, :] - h1[:, -C:, :]) * hm
          print(f"  hidden |bs={bs} - bs=1| pooled="
                f"{dh.sum() / max(hm.sum() * hb.shape[-1], 1):.8f}"
                f"   max={float(np.abs(hb[:, -C:, :] - h1[:, -C:, :]).max()):.6f}")


def test_accuracy(common, models, args, mesh, prompts, comps, mask,
                 pad_id, eos_id):
  """Which batch size is actually correct, against a real fp32 reference.

  fp32 weights + fp32 activations + precision='highest' makes the forward
  batch-invariant to 1.6e-6, so it is a legitimate ground truth. Compare the
  production bf16 path at each batch size against it.
  """
  import jax.numpy as jnp
  from flax import nnx as _nnx
  from tunix.models.qwen3 import params as qwen3_params_lib

  cfg = models._qwen3_config(args.model_name)
  cfg.dtype = jnp.float32
  cfg.param_dtype = jnp.float32
  with jax.default_matmul_precision("highest"):
    ref_model = qwen3_params_lib.create_model_from_safe_tensors(
        args.model_dir, cfg, mesh, dtype=jnp.float32)
    gd, st = _nnx.split(ref_model)
    truth = run_logps(common, gd, st, prompts, comps, pad_id, eos_id, 1)
  del ref_model

  bf = models.create_model(args.model_name, args.model_dir, mesh)
  gdb, stb = _nnx.split(bf)
  print("  production bf16 path vs fp32/highest ground truth:")
  for bs in (1, 2, 4):
    if prompts.shape[0] % bs:
      continue
    got = run_logps(common, gdb, stb, prompts, comps, pad_id, eos_id, bs)
    summarize(f"|bf16 bs={bs} - fp32 truth|", got, truth, mask)


def test_jit(common, model, prompts, comps, mask, pad_id, eos_id, C):
  """Redo the isolation tests INSIDE jax.jit.

  The real path (`common.compute_per_token_logps`) is jitted. Calling the model
  eagerly puts every primitive in its own executable, where the batch axis is a
  free matmul extent no fusion can key on -- so eager batch-invariance is the
  structurally predicted null result and says nothing about the jitted graph.
  These variants share the real path's compilation regime.
  """
  import jax.numpy as jnp
  from flax import nnx as _nnx

  graphdef, state = _nnx.split(model)

  @functools.partial(jax.jit, static_argnames=("pad_id", "eos_id"))
  def hidden_jit(gd, st, p_, c_, pad_id, eos_id):
    m = _nnx.merge(gd, st)
    it, pos, am, sid = common.process_ids(p_, c_, pad_id, eos_id, None, None)
    kw = {"positions": pos, "cache": None, "attention_mask": am,
          "skip_lm_head": True}
    if common.model_call_contains(m, "segment_ids") and sid is not None:
      kw["segment_ids"] = sid
    h, _ = m(it, **kw)
    return h

  @functools.partial(jax.jit, static_argnames=("pad_id", "eos_id", "C"))
  def logps_jit(gd, st, p_, c_, pad_id, eos_id, C):
    """Transformer + projection in ONE jit -- same regime as the real path."""
    m = _nnx.merge(gd, st)
    it, pos, am, sid = common.process_ids(p_, c_, pad_id, eos_id, None, None)
    kw = {"positions": pos, "cache": None, "attention_mask": am,
          "skip_lm_head": True}
    if common.model_call_contains(m, "segment_ids") and sid is not None:
      kw["segment_ids"] = sid
    h, _ = m(it, **kw)
    lg = m.compute_final_logits(h[:, -C - 1:-1, :])
    tgt = it[:, -C:]
    tl = jnp.take_along_axis(lg, tgt[..., None], axis=-1).squeeze(-1)
    return tl.astype(jnp.float32) - jax.nn.logsumexp(
        lg.astype(jnp.float32), axis=-1)

  def batched(fn, bs, *extra):
    outs = []
    for i in range(0, prompts.shape[0], bs):
      r = fn(graphdef, state, jnp.asarray(prompts[i:i + bs]),
             jnp.asarray(comps[i:i + bs]), pad_id, eos_id, *extra)
      outs.append(np.asarray(jax.device_get(r), np.float64))
    return np.concatenate(outs, 0)

  print("  -- hidden states, INSIDE jit --")
  h1 = batched(hidden_jit, 1)
  for bs in (2, 4):
    hb = batched(hidden_jit, bs)
    d = np.abs(hb[:, -C:, :] - h1[:, -C:, :])
    print(f"    hidden |bs={bs} - bs=1|  mean={d.mean():.10f}  max={d.max():.8f}")

  print("  -- transformer + projection in ONE jit --")
  j1 = batched(logps_jit, 1, C)
  for bs in (2, 4):
    jb = batched(logps_jit, bs, C)
    summarize(f"jit logps |bs={bs} - bs=1|", jb, j1, mask)

  print("  -- that same jit vs the real compute_per_token_logps --")
  for bs in (1, 4):
    real = run_logps(common, graphdef, state, prompts, comps, pad_id, eos_id, bs)
    mine = batched(logps_jit, bs, C)
    summarize(f"jit-mine vs real, bs={bs}", mine, real, mask)


def test_shape(common, model, prompts, comps, mask, pad_id, eos_id, C):
  """Pure matmul-shape sensitivity on FIXED inputs.

  Same hidden states, same weights, same target tokens. The only thing that
  changes is the shape of the logits matmul: project all T positions and then
  slice, versus slice to the scored window and project that. If these disagree,
  the bf16 projection's rounding is a function of its own shape -- and batch
  size is simply one factor of that shape.
  """
  import jax.numpy as jnp

  def hidden_for(p_, c_, bs):
    outs = []
    for i in range(0, p_.shape[0], bs):
      it, pos, am, sid = common.process_ids(
          jnp.asarray(p_[i:i + bs]), jnp.asarray(c_[i:i + bs]),
          pad_id, eos_id, None, None)
      kw = {"positions": pos, "cache": None, "attention_mask": am,
            "skip_lm_head": True}
      if common.model_call_contains(model, "segment_ids") and sid is not None:
        kw["segment_ids"] = sid
      h, _ = model(it, **kw)
      outs.append(h)
    return jnp.concatenate(outs, 0)

  h = hidden_for(prompts, comps, 1)        # [N, T, D], bit-identical at any bs
  targets = jnp.asarray(comps)
  T = h.shape[1]
  print(f"  hidden {tuple(h.shape)}, scored window = last {C} of {T}")

  def lp_from_logits(lg, tgt):
    tl = jnp.take_along_axis(lg, tgt[..., None], axis=-1).squeeze(-1)
    norm = jax.nn.logsumexp(lg.astype(jnp.float32), axis=-1)
    return np.asarray(jax.device_get(tl.astype(jnp.float32) - norm), np.float64)

  def wide(bs):    # project all T, then slice  (what the real path does)
    outs = []
    for i in range(0, h.shape[0], bs):
      lg = model.compute_final_logits(h[i:i + bs])[:, -C - 1:-1, :]
      outs.append(lp_from_logits(lg, targets[i:i + bs]))
    return np.concatenate(outs, 0)

  def narrow(bs):  # slice, then project only the scored window
    outs = []
    for i in range(0, h.shape[0], bs):
      lg = model.compute_final_logits(h[i:i + bs, -C - 1:-1, :])
      outs.append(lp_from_logits(lg, targets[i:i + bs]))
    return np.concatenate(outs, 0)

  w1, w4 = wide(1), wide(4)
  n1, n4 = narrow(1), narrow(4)
  print(f"  same inputs, matmul M = B x T:")
  summarize(f"wide  bs=1 [1,{T}] vs bs=4 [4,{T}]", w1, w4, mask)
  summarize(f"narrow bs=1 [1,{C}] vs bs=4 [4,{C}]", n1, n4, mask)
  summarize(f"wide vs narrow at bs=1", w1, n1, mask)
  summarize(f"wide vs narrow at bs=4", w4, n4, mask)


def test_recon(common, model, prompts, comps, mask, pad_id, eos_id, C):
  """Does hidden-then-model's-own-projection reproduce the full forward?

  If yes, the two graphs agree and any earlier gap was my manual matmul.
  If no, including the logits tensor in the compiled graph changes the
  transformer's own numerics.
  """
  import jax.numpy as jnp
  from flax import nnx as _nnx

  def hidden_for(p_, c_, bs):
    outs = []
    for i in range(0, p_.shape[0], bs):
      it, pos, am, sid = common.process_ids(
          jnp.asarray(p_[i:i + bs]), jnp.asarray(c_[i:i + bs]),
          pad_id, eos_id, None, None)
      kw = {"positions": pos, "cache": None, "attention_mask": am,
            "skip_lm_head": True}
      if common.model_call_contains(model, "segment_ids") and sid is not None:
        kw["segment_ids"] = sid
      h, _ = model(it, **kw)
      outs.append(h)
    return jnp.concatenate(outs, 0)

  targets = jnp.asarray(comps)

  def logps_via_model_head(h, bs):
    outs = []
    for i in range(0, h.shape[0], bs):
      hs = h[i:i + bs, -C - 1:-1, :]
      lg = model.compute_final_logits(hs)          # the model's own projection
      tgt = targets[i:i + bs]
      tl = jnp.take_along_axis(lg, tgt[..., None], axis=-1).squeeze(-1)
      norm = jax.nn.logsumexp(lg.astype(jnp.float32), axis=-1)
      outs.append(np.asarray(jax.device_get(
          tl.astype(jnp.float32) - norm), np.float64))
    return np.concatenate(outs, 0)

  graphdef, state = _nnx.split(model)
  for bs in (1, 4):
    h = hidden_for(prompts, comps, bs)
    recon = logps_via_model_head(h, bs)
    full = run_logps(common, graphdef, state, prompts, comps,
                     pad_id, eos_id, bs)
    summarize(f"|recon(bs={bs}) - full(bs={bs})|", recon, full, mask)


def test_truth(common, model, prompts, comps, mask, pad_id, eos_id, C):
  """Which batch size is CORRECT, not just which differs.

  The hidden states are bit-identical across batch sizes, so the only thing
  that varies is the final projection. That lets us build an exact fp32
  reference from the same hidden states and ask whether batch 1 is genuinely
  more accurate or merely happens to round the way the sampler did.
  """
  import jax.numpy as jnp

  def hidden_for(p_, c_, bs):
    outs = []
    for i in range(0, p_.shape[0], bs):
      it, pos, am, sid = common.process_ids(
          jnp.asarray(p_[i:i + bs]), jnp.asarray(c_[i:i + bs]),
          pad_id, eos_id, None, None)
      kw = {"positions": pos, "cache": None, "attention_mask": am,
            "skip_lm_head": True}
      if common.model_call_contains(model, "segment_ids") and sid is not None:
        kw["segment_ids"] = sid
      h, _ = model(it, **kw)
      outs.append(h)
    return jnp.concatenate(outs, 0)

  h = hidden_for(prompts, comps, 1)          # bit-identical at any bs
  hs = h[:, -C - 1:-1, :]                    # positions predicting completions
  targets = jnp.asarray(comps)
  W = model.embedder.input_embedding.value   # tied embedding [V, D]

  def logps_from(hslice, wdt, bs):
    outs = []
    for i in range(0, hslice.shape[0], bs):
      hh = hslice[i:i + bs].astype(wdt)
      lg = jnp.dot(hh, W.astype(wdt).T)
      tgt = targets[i:i + bs]
      tl = jnp.take_along_axis(lg, tgt[..., None], axis=-1).squeeze(-1)
      norm = jax.nn.logsumexp(lg.astype(jnp.float32), axis=-1)
      outs.append(np.asarray(jax.device_get(
          tl.astype(jnp.float32) - norm), np.float64))
    return np.concatenate(outs, 0)

  ref32 = logps_from(hs, jnp.float32, 1)     # exact-ish fp32 projection
  b1 = logps_from(hs, jnp.bfloat16, 1)
  b4 = logps_from(hs, jnp.bfloat16, 4)
  print("  projection only, identical hidden states:")
  summarize("|bf16 bs=1  - fp32 truth|", b1, ref32, mask)
  summarize("|bf16 bs=4  - fp32 truth|", b4, ref32, mask)
  summarize("|bf16 bs=1  - bf16 bs=4 |", b1, b4, mask)
  # The decisive comparison: the REAL path (full forward, lm_head inside the
  # compiled graph) against the same fp32 truth.
  print("\n  full forward (lm_head inside the graph) vs the same fp32 truth:")
  graphdef, state = nnx.split(model)
  f1 = run_logps(common, graphdef, state, prompts, comps, pad_id, eos_id, 1)
  f4 = run_logps(common, graphdef, state, prompts, comps, pad_id, eos_id, 4)
  summarize("|full bs=1 - fp32 truth|", f1, ref32, mask)
  summarize("|full bs=4 - fp32 truth|", f4, ref32, mask)
  summarize("|full bs=1 - proj bs=1  |", f1, b1, mask)
  summarize("|full bs=4 - proj bs=4  |", f4, b4, mask)

  r1 = (np.abs(f1 - ref32) * mask).sum() / max(mask.sum(), 1)
  r4 = (np.abs(f4 - ref32) * mask).sum() / max(mask.sum(), 1)
  print(f"\n  VERDICT (full forward vs fp32 truth):"
        f" bs=1 error={r1:.6f}  bs=4 error={r4:.6f}"
        f"  ratio={r4 / max(r1, 1e-12):.2f}x")
  print("  -> if ratio ~1, NEITHER batch size is more correct;"
        " bs=1 only looked better because it rounds like the sampler.")
  print("  -> if ratio >> 1, bs=1 is genuinely the accurate one.")


if __name__ == "__main__":
  main()
