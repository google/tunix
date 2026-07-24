"""Microbench: what does sequence packing actually cost on TPU splash attention?

Motivating e2e observation (gsm8k, 32 seqs/micro, fsdp=4): packing [4, 8192]
vs unpacking [32, 2048] left total train time FLAT while flash-attention
BACKWARD went 6ms -> 16ms. The JAX source says why: splash's block schedule
comes from a STATIC mask (`_process_mask` takes no segment info; the pallas
grid width is `mask_info.data_next.shape[-1]`), and `segment_ids` is just an
input that zeroes already-computed blocks inside `_apply_mask_and_soft_cap`.
So a packed row costs row_len^2/2 regardless of how many sequences it holds.

That predicts a BUDGET LAW. For a fixed token total T (rows = T / budget):
    attention ~ rows * budget^2 = T * budget      (LINEAR in budget)
    mlp/lm_head ~ rows * budget = T               (independent of budget)
i.e. packing always saves MLP tokens but pays an attention penalty that grows
with the budget -- the optimum is the smallest budget that still fits the
longest single sequence.

Cases (shapes are the PER-CHIP share of the e2e run: 32 seqs / fsdp 4 = 8):
  U      [8, 2048]  unpacked baseline, no segment_ids
  P8192  [1, 8192]  packed, real pack_sequences segment_ids  -> expect ~2x U
  P4096  [2, 4096]  packed                                   -> expect ~1x U
  C      [1, 8192]  packed shape WITHOUT segment_ids         -> expect ~P8192
                    (isolates row length from the segment feature)
  D      [1, 8192]  STATIC block-diagonal NumpyMask          -> expect ~U
                    (prototype for the LocalMask fix: can splash skip when the
                     block structure is known at trace time?)
  E      the same three geometries through a full DecoderLayer (attention+MLP)
         -> expect P8192 ~ U (reproducing the flat e2e total) and P4096 < U.

Everything runs through PRODUCTION code: data comes from
`rl_utils.pack_sequences` (the CL1 FFD packer) and the attention is
`model_lib.Attention` / `model_lib.DecoderLayer`, so the kernel invocation is
the one training actually uses. Kernel names in the xprof traces carry a
`_segmented` suffix (`get_kernel_name(is_segmented=...)`), which is how a
bench trace is matched against the e2e trace symbol-for-symbol.
"""

import argparse
import functools
import statistics
import time

import jax
from jax import numpy as jnp
import numpy as np

from flax import nnx

from tunix.models.qwen3 import model as model_lib
from tunix.rl import common
from tunix.rl import utils as rl_utils


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num_seqs", type=int, default=8,
                 help="Sequences per chip (32 seqs / fsdp 4 in the e2e run).")
  p.add_argument("--seq_len", type=int, default=2048,
                 help="Padded per-sequence length of the unpacked baseline.")
  p.add_argument("--min_tokens", type=int, default=700,
                 help="Real-token sampling range; the default 700-950 lands at")
  p.add_argument("--max_tokens", type=int, default=950,
                 help="a ~20%% dummy_ratio, matching the e2e run.")
  p.add_argument("--budgets", type=str, default="8192,4096",
                 help="Comma-separated packing budgets to bench.")
  p.add_argument("--iters", type=int, default=20,
                 help="Timed iterations per case (median reported).")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--trace_dest", type=str, default="",
                 help="If set, write one xprof trace per case under here.")
  p.add_argument("--trace_iters", type=int, default=3)
  p.add_argument("--model_config", type=str, default="qwen3_1p7b",
                 help="ModelConfig factory: qwen3_1p7b (the gsm8k run) or "
                      "qwen3_8b (FrozenLake).")
  p.add_argument("--skip_layer", action="store_true",
                 help="Skip case E (the full DecoderLayer).")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


# ---------------------------------------------------------------------------
# Data: build TrainExamples and pack them with the production packer.
# ---------------------------------------------------------------------------
def make_examples(num_seqs, seq_len, min_tokens, max_tokens, seed):
  """`num_seqs` TrainExamples, each padded to `seq_len` with a random real length.

  Mirrors the RL producer's layout: prompts are LEFT-padded, completions are
  RIGHT-padded, so `unpad_train_example` inside pack_sequences recovers exactly
  the real tokens.
  """
  rng = np.random.default_rng(seed)
  lens = rng.integers(min_tokens, max_tokens + 1, size=num_seqs)
  half = seq_len // 2

  prompt_ids = np.zeros((num_seqs, half), dtype=np.int32)
  prompt_mask = np.zeros((num_seqs, half), dtype=np.int32)
  completion_ids = np.zeros((num_seqs, half), dtype=np.int32)
  completion_mask = np.zeros((num_seqs, half), dtype=np.int32)

  for i, total in enumerate(lens):
    p_len = int(total) // 2
    c_len = int(total) - p_len
    prompt_ids[i, -p_len:] = rng.integers(1, 1000, size=p_len)  # left pad
    prompt_mask[i, -p_len:] = 1
    completion_ids[i, :c_len] = rng.integers(1, 1000, size=c_len)  # right pad
    completion_mask[i, :c_len] = 1

  return [
      common.TrainExample(
          prompt_ids=jnp.asarray(prompt_ids),
          prompt_mask=jnp.asarray(prompt_mask),
          completion_ids=jnp.asarray(completion_ids),
          completion_mask=jnp.asarray(completion_mask),
          advantages=jnp.zeros((num_seqs,), dtype=jnp.float32),
          ref_per_token_logps=None,
          old_per_token_logps=None,
      )
  ], int(lens.sum())


def pack(examples, budget, pack_size, num_seqs):
  """One packed chunk straight out of the production FFD packer.

  The bench needs ALL sequences in a single [rows, budget] chunk so the case is
  one kernel invocation. FFD may not reach the ceil(total/budget) lower bound,
  so widen the chunk until everything fits rather than silently benching a
  partial chunk.
  """
  for rows in range(pack_size, pack_size + 8):
    chunks = list(
        rl_utils.pack_sequences(
            iter([examples]),
            max_token_budget=budget,
            pack_size=rows,
            sequences_per_update=num_seqs,
        )
    )
    if len(chunks) == 1:
      return chunks[0][0]
  raise ValueError(
      f"pack_sequences needed more than {pack_size + 8} rows at budget {budget}"
  )


def geometry(example, budget, total_real_tokens):
  """(rows, segments-per-row, dummy_ratio) of a packed chunk."""
  seg = np.asarray(example.segment_ids)
  rows = seg.shape[0]
  segs = [int(seg[r].max()) for r in range(rows)]
  dummy = 1.0 - total_real_tokens / float(rows * budget)
  return rows, segs, dummy


def model_inputs(example, pad_id=0, eos_id=0):
  """(positions, attn_mask, segment_ids, seq_len) exactly as training builds them.

  Delegates to the production `common.process_ids`, so each arm gets what
  `compute_per_token_logps` would pass to the model -- crucially including the
  UNPACKED arm's per-position non-pad `segment_ids` (production feeds the 0/1
  mask to splash so pad positions are suppressed; see common.py's process_ids),
  which means both arms hit the *_segmented kernel and the only difference left
  is row length.
  """
  ids, positions, attn_mask, input_seg_ids = common.process_ids(
      example.prompt_ids,
      example.completion_ids,
      pad_id,
      eos_id,
      example.segment_ids,
      example.segment_positions,
  )
  # Packed: process_ids returns attn_mask=None and defers to the packing ids.
  seg = example.segment_ids if example.segment_ids is not None else input_seg_ids
  return positions, attn_mask, seg, ids.shape


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def timed(fn, args, iters, warmup):
  """Median wall time (ms) of `fn(*args)`, warmup excluded."""
  for _ in range(warmup):
    jax.block_until_ready(fn(*args))
  samples = []
  for _ in range(iters):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    samples.append((time.perf_counter() - t0) * 1e3)
  return statistics.median(samples)


def maybe_trace(trace_dest, name, fn, args, iters):
  if not trace_dest:
    return
  with jax.profiler.trace(f"{trace_dest}/splash_bench_{name}"):
    for _ in range(iters):
      jax.block_until_ready(fn(*args))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def build_attention(config, rngs):
  return model_lib.Attention(config=config, rngs=rngs)


def attn_fns(attn, x, segment_pos, attn_mask, segment_ids):
  """(forward, forward+backward) closures over one attention module.

  `attn_mask` is threaded through for fidelity with the production call; the
  splash path ignores it (it only consumes the static causal mask plus
  segment_ids) and the non-flash path uses it.
  """

  @jax.jit
  def fwd(x, segment_pos, attn_mask, segment_ids):
    _, out = attn(x, segment_pos, None, attn_mask, segment_ids)
    return out

  @jax.jit
  def fwd_bwd(x, segment_pos, attn_mask, segment_ids):
    def loss(x):
      _, out = attn(x, segment_pos, None, attn_mask, segment_ids)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  args = (x, segment_pos, attn_mask, segment_ids)
  return (fwd, args), (fwd_bwd, args)


def layer_fns(layer, x, segment_pos, attn_mask, segment_ids):
  @jax.jit
  def fwd(x, segment_pos, attn_mask, segment_ids):
    _, out = layer.block(x, segment_pos, None, attn_mask, segment_ids)
    return out

  @jax.jit
  def fwd_bwd(x, segment_pos, attn_mask, segment_ids):
    def loss(x):
      _, out = layer.block(x, segment_pos, None, attn_mask, segment_ids)
      return jnp.sum(out.astype(jnp.float32))
    return jax.grad(loss)(x)

  args = (x, segment_pos, attn_mask, segment_ids)
  return (fwd, args), (fwd_bwd, args)


def main():
  args = parse_args()
  budgets = [int(b) for b in args.budgets.split(",") if b]

  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(-1, 1), ('fsdp', 'tp'))
  mesh.__enter__()
  print(f"jax devices: {jax.devices()}, using dp={mesh.shape['fsdp']} mesh")
  print(
      f"geometry: {args.num_seqs} seqs x {args.seq_len} padded "
      f"(real {args.min_tokens}-{args.max_tokens}), budgets={budgets}"
  )

  examples, total_real = make_examples(
      args.num_seqs, args.seq_len, args.min_tokens, args.max_tokens, args.seed
  )
  print(f"total real tokens: {total_real}")

  config = getattr(model_lib.ModelConfig, args.model_config)()
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.dtype = jnp.bfloat16
  rngs = nnx.Rngs(params=args.seed)

  d = config.embed_dim
  key = jax.random.PRNGKey(args.seed)

  # ---- inputs per case -----------------------------------------------------
  # Every arm's (positions, attn_mask, segment_ids) comes from the production
  # `process_ids`, so the unpacked arm carries its real ~60% padding and the
  # per-position non-pad segment_ids that training actually feeds to splash.
  cases = []  # (name, x, segment_pos, attn_mask, segment_ids, note)

  pos_u, mask_u, seg_u, shape_u = model_inputs(examples[0])
  x_u = jax.random.normal(key, (*shape_u, d), jnp.bfloat16)
  real_frac = float(np.asarray(seg_u).mean())
  cases.append((
      "U", x_u, pos_u, mask_u, seg_u,
      f"{list(shape_u)} unpacked, real={real_frac:.2f}",
  ))

  for budget in budgets:
    # rows = ceil(total_real / budget) is what the packer will produce; ask for
    # that many rows per chunk so one chunk holds everything.
    rows = -(-total_real // budget)
    ex = pack(examples, budget, rows, args.num_seqs)
    r, segs, dummy = geometry(ex, budget, total_real)
    pos_p, mask_p, seg_p, shape_p = model_inputs(ex)
    x_p = jax.random.normal(key, (*shape_p, d), jnp.bfloat16)
    note = f"[{r}, {budget}] packed, segs/row={segs}, dummy={dummy:.3f}"
    cases.append((f"P{budget}", x_p, pos_p, mask_p, seg_p, note))
    if budget == max(budgets):
      # C: same shape, segment_ids dropped -> isolates row length from the
      # segment feature itself.
      cases.append((f"C{budget}", x_p, pos_p, mask_p, None,
                    f"[{r}, {budget}] packed shape, NO segment_ids"))

  # ---- attention-only ------------------------------------------------------
  attn = build_attention(config, rngs)
  print()
  print("=" * 78)
  print("CASE A/B/C -- attention only (production model_lib.Attention)")
  print("=" * 78)
  print(f"{'case':<10} {'shape':<38} {'fwd ms':>9} {'fwd+bwd ms':>11}")
  results = {}
  for name, x, pos, mask, seg, note in cases:
    (f_fwd, a_fwd), (f_fb, a_fb) = attn_fns(attn, x, pos, mask, seg)
    t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup)
    t_fb = timed(f_fb, a_fb, args.iters, args.warmup)
    results[name] = (t_fwd, t_fb)
    print(f"{name:<10} {note:<38} {t_fwd:>9.2f} {t_fb:>11.2f}")
    maybe_trace(args.trace_dest, f"attn_{name}", f_fb, a_fb, args.trace_iters)

  # ---- full decoder layer (attention + MLP) --------------------------------
  if not args.skip_layer:
    layer = model_lib.DecoderLayer(config=config, rngs=rngs)
    print()
    print("=" * 78)
    print("CASE E -- full DecoderLayer (attention + MLP)")
    print("=" * 78)
    print(f"{'case':<10} {'shape':<38} {'fwd ms':>9} {'fwd+bwd ms':>11}")
    layer_results = {}
    for name, x, pos, mask, seg, note in cases:
      if name.startswith("C"):
        continue  # C only matters for the attention isolation
      (f_fwd, a_fwd), (f_fb, a_fb) = layer_fns(layer, x, pos, mask, seg)
      t_fwd = timed(f_fwd, a_fwd, args.iters, args.warmup)
      t_fb = timed(f_fb, a_fb, args.iters, args.warmup)
      layer_results[name] = (t_fwd, t_fb)
      print(f"{name:<10} {note:<38} {t_fwd:>9.2f} {t_fb:>11.2f}")
      maybe_trace(args.trace_dest, f"layer_{name}", f_fb, a_fb, args.trace_iters)
  else:
    layer_results = {}

  # ---- verdicts ------------------------------------------------------------
  print()
  print("=" * 78)
  print("VERDICTS")
  print("=" * 78)
  base = results["U"][1]
  for name, (_, t_fb) in results.items():
    if name == "U":
      continue
    print(f"  [V1] attn bwd {name}/U = {t_fb / base:.2f}x"
          f"   (predicted: P8192~2.0, P4096~1.0, C~P8192)")
  if layer_results:
    lbase = layer_results["U"][1]
    for name, (_, t_fb) in layer_results.items():
      if name == "U":
        continue
      print(f"  [V3] layer bwd {name}/U = {t_fb / lbase:.2f}x"
            f"   (predicted: P8192~1.0 (flat), P4096<1.0 (win))")
  print()
  print("  [V2] match the xprof kernel names against the e2e trace: packed runs"
        " use *_segmented_{fwd,dq,dkv}; unpacked drop the suffix.")
  if args.trace_dest:
    print(f"       traces: {args.trace_dest}/splash_bench_*")
  print("  [V4] case D (static block-diagonal mask) is not wired yet -- add it"
        " once V1 confirms the row-length law.")


if __name__ == "__main__":
  main()
