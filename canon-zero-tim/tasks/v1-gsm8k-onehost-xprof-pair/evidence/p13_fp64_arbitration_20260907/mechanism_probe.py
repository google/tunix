"""Phase 13c mechanism probe: fp64 reference with ONE emulated bf16 mechanism.

--probe softmax_bf16: the attention softmax's backward computed as a bf16
kernel would (probabilities, incoming cotangent, row-sum and the product all
rounded to bf16), everything else fp64.  Compares the probe gradient with
the certified capture A and with the pure fp64 reference profile.
"""
import argparse
import os, importlib.util, json, os, pathlib, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
script = pathlib.Path("/mnt/disks/tunix-data/worktrees/v2_dispatch_0906/canon-zero-tim/tests/p61_backward/fp64_reference.py")
spec = importlib.util.spec_from_file_location("fp64_reference", script)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
import jax, jax.numpy as jnp, numpy as np
from tunix.models.qwen3 import model as qm

parser = argparse.ArgumentParser()
parser.add_argument("--probe", required=True, choices=("softmax_bf16", "softmax_out_bf16", "attn_bf16", "chunked_attn", "dhidden_bf16", "carriers", "chunk_accum"))
parser.add_argument("--capture-a", type=pathlib.Path, required=True)
parser.add_argument("--params-from", type=pathlib.Path, required=True)
parser.add_argument("--out", type=pathlib.Path, required=True)
args = parser.parse_args()

def bf16(x):
  return x.astype(jnp.bfloat16).astype(x.dtype)

_true_softmax = jax.nn.softmax

@jax.custom_vjp
def probe_softmax(x):
  return _true_softmax(x, axis=-1)

def probe_fwd(x):
  p = _true_softmax(x, axis=-1)
  return p, p

def probe_bwd(p, g):
  if args.probe in ("softmax_bf16", "attn_bf16"):
    pb, gb = bf16(p), bf16(g)
    s = bf16(jnp.sum(pb * gb, axis=-1, keepdims=True))
    return (bf16(pb * (gb - s)),)
  # softmax_out_bf16: only the probabilities are bf16 (as a kernel storing P in bf16)
  pb = bf16(p)
  s = jnp.sum(pb * g, axis=-1, keepdims=True)
  return (pb * (g - s),)

probe_softmax.defvjp(probe_fwd, probe_bwd)

def patched_softmax(x, axis=-1, **kwargs):
  assert axis == -1
  return probe_softmax(x)

if args.probe in ("softmax_bf16", "softmax_out_bf16"):
  jax.nn.softmax = patched_softmax  # the model looks this attribute up at call time

# attn_bf16: the whole attention core (scores einsum, softmax, P@V einsum)
# keeps its fp64 forward, but each backward is computed as a bf16 kernel:
# operands and cotangents rounded to bf16, products accumulated in f32 and
# stored bf16.
_true_einsum = jnp.einsum
ATTN_EQS = ("BTHGD,BSHD->BHGTS", "BHGTS,BSHD->BTHGD")

def bf16_einsum_bwd(eq, a, b, g):
  ab, bb, gb = bf16(a), bf16(b), bf16(g)
  _, pullback = jax.vjp(lambda x, y: _true_einsum(eq, x, y), ab, bb)
  da, db = pullback(gb)
  return bf16(da), bf16(db)

def make_probe_einsum(eq):
  @jax.custom_vjp
  def op(a, b):
    return _true_einsum(eq, a, b)
  def fwd(a, b):
    return _true_einsum(eq, a, b), (a, b)
  def bwd(res, g):
    a, b = res
    return bf16_einsum_bwd(eq, a, b, g)
  op.defvjp(fwd, bwd)
  return op

_probe_einsums = {eq: make_probe_einsum(eq) for eq in ATTN_EQS}

def patched_einsum(eq, *operands, **kwargs):
  if args.probe == "attn_bf16" and eq in _probe_einsums and len(operands) == 2 and not kwargs:
    return _probe_einsums[eq](*operands)
  return _true_einsum(eq, *operands, **kwargs)

# chunked_attn: the model's GQA core is replaced by a chunked attention whose
# K/V live in a bf16 cache written chunk by chunk (M=256 rows like the canonical
# bucket) and whose per-chunk output is rounded to bf16 -- so under autodiff the
# cache cotangent (dcache) is carried between chunks in bf16 and the attention
# output cotangent is bf16, exactly the canonical reverse's carry structure.
# Everything else (q/k/v of the current chunk, scores, softmax) stays fp64.
PROBE_CHUNK = int(os.environ.get("P13_PROBE_CHUNK", "256"))
PROBE_CROSSCHUNK = os.environ.get("P13_PROBE_CROSSCHUNK", "carry")
PROBE_CACHE_DTYPE = os.environ.get("P13_PROBE_CACHE_DTYPE", "bf16")  # bf16 | wide (no rounding)
PROBE_OUT_ROUND = os.environ.get("P13_PROBE_OUT_ROUND", "bf16")  # bf16 | none

def _probe_chunked_attn(query_proj, key_proj, value_proj, attn_mask, scale, b, t, kh, qh, d):
  import jax.numpy as _jnp
  from tunix.models.qwen3 import model as _model
  g = qh // kh
  bf16 = _jnp.bfloat16
  cdt = bf16 if PROBE_CACHE_DTYPE == "bf16" else key_proj.dtype
  if PROBE_CACHE_DTYPE == "fwdround":
    # forward sees bf16-rounded K/V (as the canonical engine's bf16 activations do)
    # but the cache and hence the dcache carry stay wide: isolates the backward
    # carry dtype from the forward rounding
    @jax.custom_vjp
    def _round_fwd_only(x):
      return x.astype(bf16).astype(x.dtype)
    def _rfo_fwd(x):
      return x.astype(bf16).astype(x.dtype), None
    def _rfo_bwd(_, g):
      return (g,)
    _round_fwd_only.defvjp(_rfo_fwd, _rfo_bwd)
    key_proj = _round_fwd_only(key_proj)
    value_proj = _round_fwd_only(value_proj)
  cache_k = _jnp.zeros((b, t, kh, d), cdt)
  cache_v = _jnp.zeros((b, t, kh, d), cdt)
  outs = []
  for c0 in range(0, t, PROBE_CHUNK):
    c1 = min(c0 + PROBE_CHUNK, t)
    cache_k = cache_k.at[:, c0:c1].set(key_proj[:, c0:c1].astype(cdt))
    cache_v = cache_v.at[:, c0:c1].set(value_proj[:, c0:c1].astype(cdt))
    if PROBE_CROSSCHUNK == "stop" and c0 > 0:
      # context keys/values from earlier chunks are constants: no cotangent
      # flows back across the chunk boundary (models a lost dcache carry)
      kk = _jnp.concatenate([jax.lax.stop_gradient(cache_k[:, :c0]), cache_k[:, c0:c1]], axis=1)
      vv = _jnp.concatenate([jax.lax.stop_gradient(cache_v[:, :c0]), cache_v[:, c0:c1]], axis=1)
      kk = kk.astype(query_proj.dtype); vv = vv.astype(query_proj.dtype)
    else:
      kk = cache_k[:, :c1].astype(query_proj.dtype)
      vv = cache_v[:, :c1].astype(query_proj.dtype)
    qc = query_proj[:, c0:c1].reshape((b, c1 - c0, kh, g, d))
    s_ = _jnp.einsum("BTHGD,BSHD->BHGTS", qc, kk) * scale
    if attn_mask is not None:
      s_ = _jnp.where(attn_mask[:, None, None, c0:c1, :c1], s_, _model.K_MASK)
    p_ = _true_softmax(s_, axis=-1)
    o_ = _jnp.einsum("BHGTS,BSHD->BTHGD", p_, vv).reshape((b, c1 - c0, qh, d))
    outs.append(o_.astype(bf16).astype(query_proj.dtype) if PROBE_OUT_ROUND == "bf16" else o_)
  return _jnp.concatenate(outs, axis=1)

if args.probe in ("chunked_attn", "carriers") or (args.probe == "chunk_accum" and os.environ.get("P13_PROBE_WITH_CHUNKED_ATTN") == "1"):
  import inspect, textwrap
  from tunix.models.qwen3 import model as _model
  src = textwrap.dedent(inspect.getsource(_model.Attention.block))
  # dedent inside block(): the GQA block sits at 6 spaces in the original file,
  # i.e. 4 after textwrap.dedent of the method
  start = src.index("# GQA")
  start = src.rfind("\n", 0, start) + 1
  end = src.index("qkv = qkv.reshape((b, t, qh, d))")
  end = src.index("\n", end) + 1
  indent = src[start:src.index("#", start)]
  replacement = (indent + "qkv = _PROBE_CHUNKED_ATTN(query_proj, key_proj, value_proj, attn_mask, "
                 "self.scale, b, t, kh, qh, d)\n")
  new_src = src[:start] + replacement + src[end:]
  ns = dict(_model.__dict__)
  ns["_PROBE_CHUNKED_ATTN"] = _probe_chunked_attn
  exec(compile(new_src, "<probe_chunked_attn>", "exec"), ns)
  _model.Attention.block = ns["block"]
  print(f"[probe:chunked_attn] Attention.block patched; chunk={PROBE_CHUNK} crosschunk={PROBE_CROSSCHUNK} cache={PROBE_CACHE_DTYPE} out_round={PROBE_OUT_ROUND}", flush=True)

if args.probe == "attn_bf16":
  jnp.einsum = patched_einsum
  jax.numpy.einsum = patched_einsum
  jax.nn.softmax = patched_softmax
  args_probe_softmax_mode = "softmax_bf16"

log = lambda m: print(f"[probe:{args.probe}] {m}", flush=True)
config = qm.ModelConfig.qwen3_1p7b()
# dhidden_bf16 / carriers: the residual-stream cotangent is rounded to bf16 at
# every decoder-layer boundary (forward identity), emulating the canonical
# reverse's per-layer program hand-off of dhidden in the activation dtype.
if args.probe in ("dhidden_bf16", "carriers"):
  from tunix.models.qwen3 import model as _model

  @jax.custom_vjp
  def _round_cotangent(x):
    return x

  def _rc_fwd(x):
    return x, None

  def _rc_bwd(_, g):
    return (g.astype(jnp.bfloat16).astype(g.dtype),)

  _round_cotangent.defvjp(_rc_fwd, _rc_bwd)
  _orig_layer_call = _model.DecoderLayer.__call__

  def _patched_layer_call(self, x, *a, **kw):
    out = _orig_layer_call(self, x, *a, **kw)
    if isinstance(out, tuple):
      cache, hidden = out
      return cache, _round_cotangent(hidden)
    return _round_cotangent(out)

  _model.DecoderLayer.__call__ = _patched_layer_call
  print(f"[probe:{args.probe}] DecoderLayer.__call__ patched: cotangent rounded to bf16 at every layer boundary", flush=True)

# ---------------------------------------------------------------------------
# chunk_accum: emulate the canonical reverse's per-row, per-chunk weight
# gradient accumulation in bf16.  For each row the loss cotangent is split by
# packed-position chunk (CH tokens, the engine's per-rank bucket); each chunk's
# gradient is rounded to bf16 and the chunks are summed in bf16 (as
# _p70_grad_tree_add does), then rows are summed in f64 and scaled.  The same
# per-chunk gradients summed unrounded in f64 give the control.
# Runs INSTEAD of ref.reference_gradient (see the early exit below).
if args.probe == "chunk_accum":
  from tunix.rl import common as _common
  _orig_cptl = _common.compute_per_token_logps
  _CURRENT_MASK = {"m": None}

  @jax.custom_vjp
  def _mask_cotangent(x, m):
    return x

  def _mc_fwd(x, m):
    return x, m

  def _mc_bwd(m, g):
    return (g * m.astype(g.dtype), None)

  _mask_cotangent.defvjp(_mc_fwd, _mc_bwd)

  def _patched_cptl(*a, **kw):
    out = _orig_cptl(*a, **kw)
    m = _CURRENT_MASK["m"]
    if m is None:
      return out
    if isinstance(out, tuple):
      return tuple(_mask_cotangent(o, m) if hasattr(o, "shape") and o.shape == m.shape else o for o in out)
    return _mask_cotangent(out, m)

  _common.compute_per_token_logps = _patched_cptl
  ref.common.compute_per_token_logps = _patched_cptl

  root = args.capture_a
  params = ref.load_capture(args.params_from or root, "model_before")
  example = ref.load_example(root).replace(old_per_token_logps=None)
  algo, pad_id, eos_id = ref.load_algo_config(root, 1.0)
  graphdef, state = ref.build_model(config, params)
  del params
  temperature = float(getattr(algo, "temperature", 1.0))

  def unreduced_sum(state, rows, mask):
    _CURRENT_MASK["m"] = mask
    logps, entropy = _common.compute_per_token_logps(
        graphdef, state, prompt_tokens=rows.prompt_ids, completion_tokens=rows.completion_ids,
        pad_id=pad_id, eos_id=eos_id, stop_gradient=False, return_entropy=True,
        temperature=temperature, canonical_actor=False,
        prompt_mask=rows.prompt_mask, completion_mask=rows.completion_mask)
    _CURRENT_MASK["m"] = None
    output = ref.algo_core.grpo_loss_from_precomputed_logps(logps, entropy, rows, algo)
    return output.primary_loss.unreduced_sum, logps

  grad_fn = jax.jit(jax.value_and_grad(unreduced_sum, has_aux=True))
  scale = ref.batch_scale(example, algo)
  CH = PROBE_CHUNK
  prompt_len = np.asarray(example.prompt_mask).sum(axis=1).astype(np.int64)
  comp_mask = np.asarray(example.completion_mask).astype(bool)
  n_rows = int(comp_mask.shape[0]); Lc = int(comp_mask.shape[1])
  bf16 = jnp.bfloat16
  acc_bf16_total = None   # sum over rows of (bf16 sum over chunks)
  acc_wide_total = None   # sum over rows of (f64 sum over chunks): control
  started = time.perf_counter(); passes = 0
  for r in range(n_rows):
    rows = ref.example_rows(example, slice(r, r + 1))
    packed_pos = prompt_len[r] + np.arange(Lc)
    chunk_ids = packed_pos // CH
    chunks = sorted(set(chunk_ids[comp_mask[r]].tolist()))
    row_bf16 = None; row_wide = None
    for c in chunks:
      mask = jnp.asarray(((chunk_ids == c) & comp_mask[r]).astype(np.float64)[None, :])
      (value, logps), grads = grad_fn(state, rows, mask)
      jax.block_until_ready(grads); passes += 1
      g_bf16 = jax.tree.map(lambda x: x.astype(bf16), grads)
      row_bf16 = g_bf16 if row_bf16 is None else jax.tree.map(jnp.add, row_bf16, g_bf16)
      row_wide = grads if row_wide is None else jax.tree.map(jnp.add, row_wide, grads)
    row_bf16 = jax.tree.map(lambda x: x.astype(jnp.float64), row_bf16)
    acc_bf16_total = row_bf16 if acc_bf16_total is None else jax.tree.map(jnp.add, acc_bf16_total, row_bf16)
    acc_wide_total = row_wide if acc_wide_total is None else jax.tree.map(jnp.add, acc_wide_total, row_wide)
    log(f"row {r} chunks={chunks} prompt_len={int(prompt_len[r])} passes={passes} elapsed={time.perf_counter() - started:.0f}s")

  def to_dict(tree):
    tree = jax.tree.map(lambda g: g * scale, tree)
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=ref._is_variable)[0]
    return {jax.tree_util.keystr(path): np.asarray(leaf[...] if ref._is_variable(leaf) else leaf) for path, leaf in flat}

  gradient_a = ref.load_capture(root, "gradient")
  report = {"probe": args.probe, "chunk": CH,
            "a_vs_bf16_chunk_accum": ref.compare_trees(to_dict(acc_bf16_total), gradient_a),
            "a_vs_wide_control": ref.compare_trees(to_dict(acc_wide_total), gradient_a)}
  args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
  for key in ("a_vs_bf16_chunk_accum", "a_vs_wide_control"):
    o = report[key]["overall"]
    log(f"{key}: rel_l2={o['rel_l2']:.3e} one_minus_cos={o['one_minus_cos']:.3e} norm_ratio_error={o['norm_ratio_error']:.3e} (A/probe norm ratio {o['got_norm'] / o['ref_norm']:.4f})")
  raise SystemExit(0)

result = ref.reference_gradient(args.capture_a, config, 4, None, log, temperature=1.0, zero_tim_point=True, params_from=args.params_from)
gradient_a = ref.load_capture(args.capture_a, "gradient")
report = {"probe": args.probe, "a_vs_probe": ref.compare_trees(result["gradient"], gradient_a),
          "logps_validation": ref.validate_logps(result["logps"], args.capture_a, result["example"], result["rows"])}
args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
o = report["a_vs_probe"]["overall"]
log(f"A vs probe: rel_l2={o['rel_l2']:.3e} one_minus_cos={o['one_minus_cos']:.3e} norm_ratio_error={o['norm_ratio_error']:.3e} (A/probe norm ratio {o['got_norm']/o['ref_norm']:.4f})")
