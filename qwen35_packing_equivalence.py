"""Checks that packed and unpacked assembly produce the same loss and gradients.

The A/B in `qwen35_packing_validation.md` §3 compares two training runs that each
generate their own rollouts, so their gradients legitimately differ and the
comparison is statistical. This script removes that noise: it builds one
trajectory set, feeds the identical set through `SequencePackedBatchAssembler`
and `PaddedBatchAssembler`, and runs both through the same engine holding the
same weights. With packing correct the two must agree to numerical tolerance.

One engine and one set of weights are used for both arms, so weight identity is
structural rather than something the caller has to arrange.

Gradients are compared through per-parameter sketches rather than by holding two
full pytrees, which would not fit at 35B:

  norm        catches any change in magnitude
  sum         catches a change in mean
  alt_sum     sum(g[even]) - sum(g[odd]), catches a change in direction that
              leaves norm and sum intact

Usage:
  python3 qwen35_packing_equivalence.py \
      --maxtext_model_name qwen3-0.6b --mesh_fsdp 8 --mesh_tp 1 \
      --maxtext_ckpt_path gs://... \
      --max_seq_token_per_tpu 4096 --trainer_fsdp 8 --trainer_dp 1 \
      --num_generations 16 --mini_batch_size 256 --train_micro_batch_size 8 \
      --length_csv /tmp/packed.csv
"""

import argparse
import json
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("equivalence")


def parse_args(argv):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--maxtext_model_name", default="qwen3-0.6b")
  p.add_argument("--maxtext_ckpt_path", default="",
                 help="Orbax params-only checkpoint. Empty means random init, "
                      "which is still a valid comparison because both arms use "
                      "the same engine instance.")
  p.add_argument("--maxtext_output_directory", default="/tmp/packing_equivalence")
  p.add_argument("--mesh_fsdp", type=int, default=1)
  p.add_argument("--mesh_tp", type=int, default=1)
  p.add_argument("--mesh_expert", type=int, default=1)
  p.add_argument("--max_prompt_length", type=int, default=512)
  p.add_argument("--max_response_length", type=int, default=1024)
  p.add_argument("--max_seq_token_per_tpu", type=int, default=4096)
  p.add_argument("--trainer_fsdp", type=int, default=8)
  p.add_argument("--trainer_dp", type=int, default=1)
  p.add_argument("--num_generations", type=int, default=16)
  p.add_argument("--rollouts_per_update", type=int, default=256,
                 help="Trajectories per optimizer update. The assembler takes "
                      "prompt groups, so this is divided by --num_generations; "
                      "the launcher's MINI_BATCH_SIZE uses the other convention.")
  p.add_argument("--train_micro_batch_size", type=int, default=8)
  p.add_argument("--num_trajectories", type=int, default=0,
                 help="Trajectories to build. Default: one optimizer update.")
  p.add_argument("--length_csv", default="",
                 help="Trajectory CSV to draw a realistic completion-length "
                      "distribution from. Without it, lengths are lognormal.")
  p.add_argument("--chars_per_token", type=float, default=3.6,
                 help="Used only to convert CSV character lengths to token counts.")
  p.add_argument("--pad_id", type=int, default=0)
  p.add_argument("--eos_id", type=int, default=-1,
                 help="Defaults to --pad_id, matching distributed_rl_engine.")
  p.add_argument("--vocab_size", type=int, default=151936)
  p.add_argument("--old_logp_sigma", type=float, default=0.02,
                 help="Jitter added to the measured old-policy logps. Keep it "
                      "well inside log(1 +/- epsilon) so most tokens land in "
                      "the unclipped branch, which is the only branch that "
                      "carries a gradient.")
  p.add_argument("--temperature", type=float, default=1.0)
  p.add_argument("--beta", type=float, default=0.0)
  p.add_argument("--epsilon", type=float, default=0.2)
  p.add_argument("--loss_agg_mode", default="sequence-mean-token-mean")
  p.add_argument("--float32", action="store_true",
                 help="Run the model in float32 with exact matmuls. MaxText "
                      "defaults to bfloat16 for activations and weights, whose "
                      "ulp at a randomly initialized model's logp scale of "
                      "-241 is 0.5 -- larger than any packing difference, so "
                      "the bf16 comparison is uninformative there. Not needed "
                      "with --maxtext_ckpt_path, where logps are near -1.")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="/tmp/packing_equivalence.json")
  p.add_argument("--rtol", type=float, default=2e-2,
                 help="Relative tolerance for the verdict. bf16 accumulation "
                      "over a different microbatch partition does not reproduce "
                      "bitwise, so this is loose by design.")
  return p.parse_args(argv)


def completion_lengths(args, n, rng):
  """Draws n completion token counts, matching production when a CSV is given."""
  if args.length_csv:
    import ast
    import pandas as pd

    def assistant_text(s):
      try:
        msgs = ast.literal_eval(s)
      except (ValueError, SyntaxError):
        return None
      if not isinstance(msgs, list):
        return None
      t = [m.get("content", "") for m in msgs
           if isinstance(m, dict) and m.get("role") == "assistant"]
      return "\n".join(t) if t else None

    d = pd.read_csv(args.length_csv, engine="python")
    gen = d.completion.astype(str).map(assistant_text).dropna()
    lens = (gen.str.len() / args.chars_per_token).round().astype(int).values
    lens = lens[lens > 0]
    log.info("length pool from %s: %d completions, median %d tokens",
             args.length_csv, len(lens), int(np.median(lens)))
    out = rng.choice(lens, size=n, replace=True)
  else:
    out = rng.lognormal(mean=np.log(300), sigma=0.7, size=n).astype(int)
  return np.clip(out, 1, args.max_response_length)


def build_trajectories(args):
  """Builds one deterministic set of unbatched payloads, shared by both arms."""
  from tunix.experimental.common import datatypes

  rng = np.random.default_rng(args.seed)
  n = args.num_trajectories or args.rollouts_per_update
  c_lens = completion_lengths(args, n, rng)
  p_lens = rng.integers(32, args.max_prompt_length + 1, size=n)

  payloads = []
  for i in range(n):
    pl, cl = int(p_lens[i]), int(c_lens[i])
    payloads.append(datatypes.RLTrainerPayload(
        prompt_ids=rng.integers(1, args.vocab_size, size=pl, dtype=np.int32),
        prompt_mask=np.ones(pl, dtype=np.float32),
        completion_ids=rng.integers(1, args.vocab_size, size=cl, dtype=np.int32),
        completion_mask=np.ones(cl, dtype=np.float32),
        advantages=np.float32(rng.normal()),
        # Placeholders. measure_old_logps() replaces them with the policy's own
        # per-token logps before either arm runs.
        old_per_token_logps=np.zeros(cl, dtype=np.float32),
        ref_per_token_logps=np.zeros(cl, dtype=np.float32),
        # batch_assembly._extract_trajectory_id reads this exact key; any other
        # spelling yields an empty id for every trajectory.
        metadata={"traj_id": f"traj_{i:05d}"},
    ))
  log.info("built %d trajectories: completion tokens median %d, max %d; "
           "total content tokens %d",
           n, int(np.median(c_lens)), int(c_lens.max()),
           int((p_lens + c_lens).sum()))
  return payloads


def assemble(payloads, *, packed, args):
  """Runs one trajectory set through one assembler and returns its microbatches."""
  from tunix.experimental.orchestrator import batch_assembly

  cfg = batch_assembly.BatchConfig(
      pad_id=args.pad_id,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      max_seq_token_per_tpu=args.max_seq_token_per_tpu if packed else None,
      trainer_fsdp=args.trainer_fsdp,
      trainer_dp=args.trainer_dp,
  )
  asm = batch_assembly.create_batch_assembler(
      group_size=args.num_generations,
      mini_batch_size=max(1, args.rollouts_per_update // args.num_generations),
      train_micro_batch_size=args.train_micro_batch_size,
      batch_config=cfg,
  )
  log.info("%s arm: %s", "packed" if packed else "unpacked", type(asm).__name__)
  batches = list(asm.feed(payloads)) + list(asm.flush())
  seen = [t for b in batches for t in b.trajectory_ids]
  want = sorted(p.metadata["traj_id"] for p in payloads)
  if sorted(seen) != want:
    raise ValueError(
        f"assembler emitted {len(seen)} trajectory ids for {len(payloads)} "
        f"payloads, and they do not match the ids fed in")
  total = len(seen)
  log.info("  %d microbatches, %d trajectories, shapes %s", len(batches), total,
           sorted({tuple(np.asarray(b.payload.completion_ids).shape) for b in batches}))
  return batches


def policy_logps(engine, batches, args):
  """Runs the policy forward and returns each trajectory's per-token logps.

  Works for either layout. An unpacked row holds exactly one trajectory, so its
  `completion_mask` selects that trajectory's tokens directly. A packed row
  holds several, so the mask is split by `segment_ids`, taking the segments in
  the order they first appear -- the order `trajectory_ids` lists them in.

  Args:
    engine: The engine whose weights define the policy.
    batches: Microbatches from one arm.
    args: Parsed flags.

  Returns:
    `{traj_id: per-token logps}`, one entry per trajectory in `batches`.
  """
  import flax.nnx as nnx
  import jax
  import jax.numpy as jnp
  from tunix.rl import common

  graphdef, state = nnx.split(engine._model)  # pylint: disable=protected-access
  eos_id = args.pad_id if args.eos_id < 0 else args.eos_id

  @jax.jit
  def run(prompt_ids, completion_ids, segment_ids, segment_positions):
    return common.compute_per_token_logps(
        graphdef, state,
        prompt_tokens=prompt_ids, completion_tokens=completion_ids,
        pad_id=args.pad_id, eos_id=eos_id, stop_gradient=True,
        segment_ids=segment_ids, segment_positions=segment_positions,
        temperature=args.temperature)

  out = {}
  for b in batches:
    p = b.payload
    seg = getattr(p, "segment_ids", None)
    lp = np.asarray(jax.device_get(run(
        jnp.asarray(p.prompt_ids), jnp.asarray(p.completion_ids),
        None if seg is None else jnp.asarray(seg),
        None if seg is None else jnp.asarray(p.segment_positions))))
    mask = np.asarray(p.completion_mask) > 0
    ids = list(b.trajectory_ids)
    if seg is None:
      for row, tid in enumerate(ids):
        out[tid] = lp[row][mask[row]]
      continue
    seg = np.asarray(seg)
    cursor = 0
    for row in range(lp.shape[0]):
      live = seg[row][mask[row]]
      for s in dict.fromkeys(live):  # first-appearance order
        out[ids[cursor]] = lp[row][mask[row]][live == s]
        cursor += 1
    if cursor != len(ids):
      raise ValueError(
          f"recovered {cursor} segments from a packed microbatch that lists "
          f"{len(ids)} trajectories")
  return out


def measure_old_logps(engine, batches, payloads, args, rng):
  """Replaces each trajectory's old logps with the policy's own, plus jitter.

  GRPO's gradient flows only through the unclipped branch,
  `-advantage * exp(logp - old_logp)`. Synthetic old logps drawn from a guessed
  distribution put that ratio outside the clip band for every token, which has
  exactly zero gradient, so both arms return zero and agree vacuously. A
  randomly initialized Qwen3-0.6B, for instance, has per-token logps near -228
  spanning 400 nats, so no single constant can stand in for them.

  Production never sees that: the old policy is one optimizer step behind, so
  the ratio starts at 1. Measuring the policy's own logps reproduces it, at the
  cost of one forward pass over the unpacked microbatches.

  Args:
    engine: The engine whose weights define the policy.
    batches: Unpacked microbatches, which map one row to one trajectory.
    payloads: The unbatched payloads the microbatches were assembled from.
    args: Parsed flags.
    rng: Source of the jitter.

  Returns:
    A new payload list, in the order given, carrying measured old logps.
  """
  measured = policy_logps(engine, batches, args)

  out = []
  for p in payloads:
    tid = p.metadata["traj_id"]
    lp = measured[tid]
    if lp.shape[0] != p.completion_ids.shape[0]:
      raise ValueError(
          f"{tid}: measured {lp.shape[0]} logps for a completion of "
          f"{p.completion_ids.shape[0]} tokens; the completion mask and the "
          "unbatched payload disagree")
    jitter = rng.normal(0.0, args.old_logp_sigma, size=lp.shape[0])
    out.append(p.replace(
        old_per_token_logps=(lp + jitter).astype(np.float32),
        ref_per_token_logps=(lp + jitter).astype(np.float32)))

  flat = np.concatenate([np.asarray(p.old_per_token_logps) for p in out])
  log.info("measured old logps over %d trajectories: mean %.3f, min %.3f, "
           "max %.3f", len(out), flat.mean(), flat.min(), flat.max())
  return out


def compare_logps(packed, unpacked):
  """Compares per-token logps between the two layouts, trajectory by trajectory."""
  shared = sorted(set(packed) & set(unpacked))
  diffs, worst = [], (0.0, None)
  for tid in shared:
    a, b = packed[tid], unpacked[tid]
    if a.shape != b.shape:
      raise ValueError(f"{tid}: packed has {a.shape[0]} tokens, unpacked {b.shape[0]}")
    d = np.abs(a - b)
    diffs.append(d)
    if d.max() > worst[0]:
      worst = (float(d.max()), tid)
  flat = np.concatenate(diffs)
  ref = np.concatenate([unpacked[t] for t in shared])
  print("\nper-token logps, packed vs unpacked, same trajectories")
  print(f"  {len(shared)} trajectories, {flat.size} tokens")
  print(f"  logp scale        mean {ref.mean():.4f}, min {ref.min():.4f}")
  print(f"  absolute diff     mean {flat.mean():.4e}, p99 "
        f"{np.quantile(flat, 0.99):.4e}, max {flat.max():.4e} ({worst[1]})")
  print(f"  implied ratio     exp(max diff) = {np.exp(flat.max()):.4f}")
  return {"trajectories": len(shared), "tokens": int(flat.size),
          "mean_abs_diff": float(flat.mean()), "max_abs_diff": float(flat.max()),
          "p99_abs_diff": float(np.quantile(flat, 0.99))}


def grad_leaves(tree):
  """Flattens a gradient tree into `{parameter path: array}`."""
  import jax

  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
    if not hasattr(leaf, "shape") or leaf.size == 0:
      continue
    out["/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)] = leaf
  return out


def run_arm(engine, batches, label):
  """Feeds every microbatch through fwd_bwd and returns the accumulated grads."""
  import jax

  engine._accumulated_grads = None  # pylint: disable=protected-access
  engine._accumulated_denominator = None  # pylint: disable=protected-access
  engine._cached_losses = []  # pylint: disable=protected-access

  for i, b in enumerate(batches):
    engine.fwd_bwd(b.payload)
    if i == 0 or (i + 1) % 10 == 0:
      log.info("  %s: microbatch %d/%d", label, i + 1, len(batches))

  grads = engine._reduced_accumulated_grads()  # pylint: disable=protected-access
  denom = engine._accumulated_denominator  # pylint: disable=protected-access

  # The two arms split the same trajectories into different numbers of
  # microbatches, so a mean of per-microbatch means is not comparable between
  # them. The pooled sum/denominator is, and it is what the optimizer sees.
  sums, denoms = [], []
  for l in engine._cached_losses:  # pylint: disable=protected-access
    sums.append(float(jax.device_get(l.unreduced_sum)))
    denoms.append(float(jax.device_get(l.denominator)))
  pooled = sum(sums) / sum(denoms) if sum(denoms) else float("nan")
  denom_v = float(jax.device_get(denom)) if denom is not None else float("nan")
  log.info("  %s: pooled loss %.8f over %d microbatches, loss denominator %.6g, "
           "grad denominator %.6g",
           label, pooled, len(sums), sum(denoms), denom_v)
  return {
      "microbatches": len(batches),
      "grad_denominator": denom_v,
      "loss_denominator": sum(denoms),
      "loss_sums": sums,
      "loss_denominators": denoms,
      "pooled_loss": pooled,
      "grads": grad_leaves(grads),
  }


def compare(a, b, rtol):
  """Compares two arms and prints the largest relative disagreements."""
  print("\n" + "=" * 74)
  print("packed vs unpacked, identical trajectories and identical weights")
  print("=" * 74)
  print(f"  {'':<18}{'packed':>14} {'unpacked':>16}")
  print(f"  microbatches      {a['microbatches']:>14d} {b['microbatches']:>16d}")
  print(f"  grad denominator  {a['grad_denominator']:>14.6g} {b['grad_denominator']:>16.6g}")
  print(f"  loss denominator  {a['loss_denominator']:>14.6g} {b['loss_denominator']:>16.6g}")
  print(f"  pooled loss       {a['pooled_loss']:>14.8f} {b['pooled_loss']:>16.8f}")

  dl = abs(a["pooled_loss"] - b["pooled_loss"])
  rel_l = dl / max(abs(a["pooled_loss"]), 1e-12)
  print(f"  loss difference   {dl:.3e} absolute, {rel_l:.3e} relative")

  # Exact per-parameter comparison. Both gradient trees are sharded over the
  # same mesh as the weights, so holding the two together costs one extra
  # parameter-sized buffer rather than a host-side copy.
  import jax.numpy as jnp

  shared = sorted(set(a["grads"]) & set(b["grads"]))
  only = set(a["grads"]) ^ set(b["grads"])
  if only:
    print(f"  WARNING: {len(only)} parameter paths present in only one arm")

  rows, num, den = [], 0.0, 0.0
  for k in shared:
    x = jnp.asarray(a["grads"][k], dtype=jnp.float32).reshape(-1)
    y = jnp.asarray(b["grads"][k], dtype=jnp.float32).reshape(-1)
    nx, ny = float(jnp.linalg.norm(x)), float(jnp.linalg.norm(y))
    d = float(jnp.linalg.norm(x - y))
    cos = float(jnp.dot(x, y)) / (nx * ny) if nx and ny else float("nan")
    rows.append((d / max(nx, ny, 1e-30), cos, k, nx, ny, x.size))
    num += d ** 2
    den += max(nx, ny) ** 2
  rows.sort(reverse=True)
  whole = (num ** 0.5) / (den ** 0.5) if den else 0.0

  print(f"\n  {len(shared)} parameter paths, exact relative L2 distance")
  print(f"    whole tree      ||packed - unpacked|| / ||unpacked|| = {whole:.3e}")
  print(f"    {'rel L2':>10} {'cosine':>12}  {'||packed||':>12} {'||unpacked||':>13}  path")
  for rel, cos, k, nx, ny, _ in rows[:10]:
    print(f"    {rel:>10.3e} {cos:>12.9f}  {nx:>12.6g} {ny:>13.6g}  {k[:40]}")

  # Two zero gradients agree trivially. That happens when every token falls
  # outside the GRPO clip band, which has no gradient.
  live = [r for r in rows if r[3] or r[4]]
  if len(live) < len(rows):
    print(f"\n  {len(rows) - len(live)} of {len(rows)} parameter paths have "
          "zero gradient in both arms")

  max_rel = rows[0][0] if rows else 0.0
  worst_cos = min((r[1] for r in live), default=float("nan"))
  print(f"\n  worst per-parameter relative L2: {max_rel:.3e}  (tolerance {rtol:.0e})")
  print(f"  worst per-parameter cosine:      {worst_cos:.9f}")
  if not live:
    print("  VERDICT: INVALID -- every gradient is zero, so the arms agree")
    print("           trivially. Every token is outside the GRPO clip band;")
    print("           see measure_old_logps and --old_logp_sigma.")
    return {"pass": False, "invalid": "all gradients zero"}
  ok = max_rel <= rtol and rel_l <= rtol
  if ok:
    print("  VERDICT: PASS -- packed and unpacked assembly agree")
  else:
    print(f"  VERDICT: FAIL -- disagreement exceeds {rtol:.0e}")
  return {
      "pass": ok,
      "loss_rel_diff": rel_l,
      "whole_tree_rel_l2": whole,
      "worst_rel_l2": max_rel,
      "worst_cosine": worst_cos,
      "per_parameter": [
          {"path": k, "rel_l2": rel, "cosine": cos, "norm_packed": nx,
           "norm_unpacked": ny, "size": size}
          for rel, cos, k, nx, ny, size in rows
      ],
  }


def main(argv):
  args = parse_args(argv)
  import jax
  from tunix.experimental.orchestrator import algorithm_adapter
  from tunix.rl import algorithm_config
  from tunix.utils import maxtext_utils

  log.info("jax devices: %d x %s", jax.device_count(), jax.devices()[0].platform)

  if args.float32:
    # build_maxtext_config takes no dtype argument and the HyperParameters it
    # returns are read-only, so the overrides go in on the way through.
    pyconfig = maxtext_utils.maxtext_modules()[0]
    base_initialize = pyconfig.initialize
    pyconfig.initialize = lambda argv, *a, **kw: base_initialize(
        list(argv) + ["dtype=float32", "weight_dtype=float32",
                      "matmul_precision=highest"], *a, **kw)
    log.info("float32 mode: dtype, weight_dtype and matmul_precision overridden")

  payloads = build_trajectories(args)

  # max_target_length follows the packing budget, which is the wider of the two
  # arms, so the same engine can run the unpacked microbatches as well.
  cfg = maxtext_utils.build_maxtext_config(
      model_name=args.maxtext_model_name,
      train_micro_batch_size=args.train_micro_batch_size,
      mesh_fsdp=args.mesh_fsdp,
      mesh_tp=args.mesh_tp,
      mesh_expert=args.mesh_expert,
      num_devices=jax.device_count(),
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      load_parameters_path=args.maxtext_ckpt_path,
      base_output_directory=args.maxtext_output_directory,
      max_seq_token_per_tpu=args.max_seq_token_per_tpu,
  )
  mesh = maxtext_utils.create_maxtext_mesh(cfg)
  log.info("mesh: %s", mesh)
  engine = maxtext_utils.create_maxtext_engine(
      cfg, mesh=mesh, tokenizer_pad_id=args.pad_id, wrap_with_tunix_adapter=True)

  # Same wiring distributed_rl_engine.py:555-571 applies to the trainer worker.
  # Without it fwd_bwd falls back to MaxText's pre-train loss, which does not
  # understand an RLTrainerPayload.
  eos_id = args.pad_id if args.eos_id < 0 else args.eos_id
  algo = algorithm_adapter.GRPOAdapter(
      algo_config=algorithm_config.GRPOConfig(
          num_generations=args.num_generations,
          epsilon=args.epsilon,
          beta=args.beta,
          temperature=args.temperature,
          loss_agg_mode=args.loss_agg_mode,
      ),
      mini_batch_size=max(1, args.rollouts_per_update // args.num_generations),
      train_micro_batch_size=args.train_micro_batch_size,
      max_packed_len=args.max_seq_token_per_tpu,
      max_response_length=args.max_response_length,
  )
  engine.with_loss_fn(algo.loss_fn(), has_aux=True)
  engine.with_gen_model_input_fn(
      algo.build_gen_model_input_fn(pad_id=args.pad_id, eos_id=eos_id))

  with mesh:
    # Measured against the unpacked arm, whose rows map one to one onto
    # trajectories, then both arms are assembled from the result so they carry
    # the same old logps.
    payloads = measure_old_logps(
        engine, assemble(payloads, packed=False, args=args), payloads, args,
        np.random.default_rng(args.seed + 1))
    packed_batches = assemble(payloads, packed=True, args=args)
    unpacked_batches = assemble(payloads, packed=False, args=args)

    # The forward pass on its own, before any loss aggregation. If the two
    # layouts disagree here, nothing downstream can agree, and the cause is the
    # packed attention mask or position ids rather than the loss reduction.
    logp_stats = compare_logps(
        policy_logps(engine, packed_batches, args),
        policy_logps(engine, unpacked_batches, args))

    packed = run_arm(engine, packed_batches, "packed")
    unpacked = run_arm(engine, unpacked_batches, "unpacked")
    verdict = compare(packed, unpacked, args.rtol)

  with open(args.out, "w") as f:
    json.dump({
        "args": vars(args),
        "logps": logp_stats,
        "gradients": verdict,
        "packed": {k: v for k, v in packed.items() if k != "grads"},
        "unpacked": {k: v for k, v in unpacked.items() if k != "grads"},
        "pass": verdict["pass"],
    }, f, indent=2)
  log.info("wrote %s", args.out)
  return 0 if verdict["pass"] else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
