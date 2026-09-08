#!/usr/bin/env python3
"""fp64 reference gradient for a P61 full-tree capture (tasks/v2_dispatch Phase 13).

Re-derives the first update's committed gradient in float64 on the CPU from
the captured example and the captured parameters, with the stock tunix
Qwen3 forward (``common.compute_per_token_logps``, ``canonical_actor=False``)
and the very loss seam the canonical path uses
(``algo_core.grpo_loss_from_precomputed_logps``), then measures each
captured gradient (the certified path, and a sealed variant) against it:
per leaf, per leaf group and overall -- rel_l2, one_minus_cos,
norm_ratio_error -- plus the two captures against each other.  The
reference validates itself first: its per-token logps must agree with the
engine's captured logps to bf16 order on the action tokens.

usage: fp64_reference.py --capture-a A/p61_numerical [--capture-b B/p61_numerical]
                         --out metrics.json [--rows-per-step 4] [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402


_REAL_FLOAT32 = jax.numpy.float32


def widen_float32_to_float64():
  """Makes every ``jnp.float32`` the tunix model, logps helper and loss seam
  spell resolve to float64 in this process.  Those functions pin float32
  explicitly (norms, softmax, the log-softmax, the loss's casts), which is
  right for the trainer and wrong for a reference: the attribute is looked
  up on the module at call time, so rebinding it is enough."""
  jax.numpy.float32 = jax.numpy.float64
  jnp.float32 = jnp.float64


def restore_float32():
  """Undoes ``widen_float32_to_float64`` (a later pass in another dtype)."""
  jax.numpy.float32 = _REAL_FLOAT32
  jnp.float32 = _REAL_FLOAT32


COMPUTE_DTYPE = os.environ.get("P61_REFERENCE_DTYPE", "float64")
COMPUTE_DTYPE_MAIN = "fp64"
if COMPUTE_DTYPE == "float64":
  widen_float32_to_float64()
elif COMPUTE_DTYPE not in ("float32", "bfloat16"):
  raise SystemExit(f"P61_REFERENCE_DTYPE must be float64, float32 or bfloat16: {COMPUTE_DTYPE!r}")

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from tunix.models.qwen3 import model as qwen3_model  # noqa: E402
from tunix.rl import algo_core  # noqa: E402
from tunix.rl import common  # noqa: E402

EXAMPLE_FIELDS = (
    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
    "advantages", "ref_per_token_logps", "old_per_token_logps",
    "sampler_is_weights",
)


def _is_variable(value):
  return isinstance(value, nnx.Variable)


def load_capture(root: Path, name: str) -> dict[str, np.ndarray]:
  """Loads one P61 capture (``manifest.json`` + ``leaf_*.npy``) by path."""
  manifest = json.loads((root / name / "manifest.json").read_text())
  return {
      leaf["path"]: np.load(root / name / leaf["file"], allow_pickle=False)
      for leaf in manifest["leaves"]
  }


def load_example(root: Path, dtype=None):
  dtype = dtype or jnp.dtype(COMPUTE_DTYPE if COMPUTE_DTYPE != "bfloat16" else "float32")
  leaves = load_capture(root, "example")
  fields = {}
  for name in EXAMPLE_FIELDS:
    value = leaves.get(f".{name}")
    if value is None:
      fields[name] = None
    elif np.issubdtype(value.dtype, np.floating):
      fields[name] = jnp.asarray(value, dtype)
    else:
      fields[name] = jnp.asarray(value)
  return common.TrainExample(**fields)


def load_algo_config(root: Path, temperature: float | None = None):
  record = json.loads((root / "algo_config.json").read_text())
  fields = dict(record["algo_config"])
  if temperature is not None:
    fields["temperature"] = float(temperature)
  if fields.get("temperature") is None:
    raise ValueError(
        "the capture's algo_config.json has no temperature; pass --temperature "
        "(the rollout config's value the loss path uses)"
    )
  return types.SimpleNamespace(**fields), int(record["pad_id"]), int(record["eos_id"])


def set_compute_dtype(name: str):
  """Switches the process's compute dtype between passes."""
  global COMPUTE_DTYPE
  if name == "float64":
    widen_float32_to_float64()
  else:
    restore_float32()
  COMPUTE_DTYPE = name


def build_model(config: qwen3_model.ModelConfig, params: dict[str, np.ndarray], dtype=None):
  """A Qwen3 model whose parameters are the captured ones, in ``dtype``,
  computing in ``dtype`` (the config's activation and parameter dtypes).
  ``bfloat16`` mimics the trainer's mixed precision: bf16 parameters and
  activations, the norms, softmax, log-softmax and loss in float32."""
  import dataclasses  # pylint: disable=g-import-not-at-top

  dtype = dtype or jnp.dtype(COMPUTE_DTYPE)
  config = dataclasses.replace(config, dtype=dtype, param_dtype=dtype)
  abstract = nnx.eval_shape(lambda: qwen3_model.Qwen3(config, rngs=nnx.Rngs(0)))
  graphdef, state = nnx.split(abstract)
  missing = []

  def fill(path, variable):
    key = jax.tree_util.keystr(path)
    value = params.get(key)
    if value is None:
      missing.append(key)
      return variable
    # An abstract variable holds a ShapeDtypeStruct: read its shape through
    # the pytree leaf, not through indexing.
    shape = tuple(jax.tree_util.tree_leaves(variable)[0].shape)
    if tuple(value.shape) != shape:
      raise ValueError(f"{key}: captured shape {value.shape} != model {shape}")
    return variable.replace(jnp.asarray(value, dtype))

  state = jax.tree_util.tree_map_with_path(fill, state, is_leaf=_is_variable)
  if missing:
    raise ValueError(f"capture lacks {len(missing)} model leaves, e.g. {missing[:3]}")
  count = len(jax.tree_util.tree_leaves(state, is_leaf=_is_variable))
  if count != len(params):
    raise ValueError(f"model has {count} leaves, capture {len(params)}")
  return graphdef, state


def example_rows(example, rows):
  return jax.tree.map(lambda v: v[rows] if v is not None and v.ndim >= 1 else v, example)


def make_grad_fn(graphdef, algo, pad_id, eos_id):
  temperature = float(getattr(algo, "temperature", 1.0))

  def unreduced_sum(state, example):
    logps, entropy = common.compute_per_token_logps(
        graphdef, state, prompt_tokens=example.prompt_ids,
        completion_tokens=example.completion_ids, pad_id=pad_id,
        eos_id=eos_id, stop_gradient=False, return_entropy=True,
        temperature=temperature, canonical_actor=False,
        prompt_mask=example.prompt_mask, completion_mask=example.completion_mask,
    )
    output = algo_core.grpo_loss_from_precomputed_logps(logps, entropy, example, algo)
    return output.primary_loss.unreduced_sum, logps

  return jax.jit(jax.value_and_grad(unreduced_sum, has_aux=True))


def batch_scale(example, algo):
  """The loss scale of the whole batch (a function of the masks only)."""
  zeros = jnp.zeros(example.completion_ids.shape, jnp.float32)
  output = algo_core.grpo_loss_from_precomputed_logps(zeros, zeros, example, algo)
  return float(output.primary_loss.compute_scale())


def metrics(reference: np.ndarray, got: np.ndarray) -> dict[str, float]:
  ref = reference.astype(np.float64).reshape(-1)
  val = got.astype(np.float64).reshape(-1)
  ref_norm = float(np.linalg.norm(ref))
  got_norm = float(np.linalg.norm(val))
  diff_norm = float(np.linalg.norm(val - ref))
  if ref_norm == 0.0:
    rel_l2 = 0.0 if got_norm == 0.0 else math.inf
  else:
    rel_l2 = diff_norm / ref_norm
  if ref_norm == 0.0 or got_norm == 0.0:
    one_minus_cos = 0.0 if ref_norm == got_norm else 1.0
  else:
    cosine = float(np.dot(ref, val)) / (ref_norm * got_norm)
    one_minus_cos = max(0.0, 1.0 - max(-1.0, min(1.0, cosine)))
  norm_ratio_error = math.inf if ref_norm == 0.0 else abs(got_norm / ref_norm - 1.0)
  return {
      "rel_l2": rel_l2, "one_minus_cos": one_minus_cos,
      "norm_ratio_error": norm_ratio_error, "ref_norm": ref_norm,
      "got_norm": got_norm, "diff_norm": diff_norm,
  }


def leaf_group(path: str) -> str:
  if path.startswith("['layers']"):
    rest = path[len("['layers']"):]
    index = rest[1:rest.index("]")]
    tail = rest[rest.index("]") + 1:]
    kind = tail.split("'")[1] if "'" in tail else tail
    return f"layer[{index}].{kind}"
  return path.split("'")[1] if "'" in path else path


def compare_trees(reference: dict[str, np.ndarray], got: dict[str, np.ndarray]):
  per_leaf, groups = {}, {}
  all_ref, all_got = [], []
  for path, ref in reference.items():
    value = got[path]
    per_leaf[path] = metrics(ref, value)
    groups.setdefault(leaf_group(path), ([], []))
    groups[leaf_group(path)][0].append(ref.reshape(-1))
    groups[leaf_group(path)][1].append(value.reshape(-1))
    all_ref.append(ref.reshape(-1))
    all_got.append(value.reshape(-1))
  per_group = {
      name: metrics(np.concatenate(r), np.concatenate(g))
      for name, (r, g) in groups.items()
  }
  overall = metrics(np.concatenate(all_ref), np.concatenate(all_got))
  return {"overall": overall, "groups": per_group, "leaves": per_leaf}


def reference_gradient(root: Path, config, rows_per_step: int, max_rows: int | None, log,
                       temperature: float | None = None, zero_tim_point: bool = True,
                       params_from: Path | None = None):
  params_root = params_from or root
  if params_from is not None:
    # Another capture's parameters may stand in only when they are the
    # same values: every leaf's data hash must match this capture's manifest.
    own = {leaf["path"]: leaf["data_sha256"] for leaf in
           json.loads((root / "model_before" / "manifest.json").read_text())["leaves"]}
    theirs = {leaf["path"]: leaf["data_sha256"] for leaf in
              json.loads((params_from / "model_before" / "manifest.json").read_text())["leaves"]}
    if own != theirs:
      raise ValueError(f"--params-from {params_from} holds different parameters than {root}")
    log(f"parameters from {params_from} (all {len(own)} leaf hashes match this capture's manifest)")
  params = load_capture(params_root, "model_before")
  example = load_example(root)
  if zero_tim_point:
    # The certified run sits exactly at importance ratio 1: its trainer
    # logps are the rollout logps bit for bit (strict zero-TIM), so no
    # token is clipped and the policy-gradient weight is -adv exactly.
    # An fp64 forward's logps differ from the bf16 rollout logps by
    # bf16 noise, which would move ratios and clips token by token and
    # measure that noise instead of the backward's arithmetic.  Evaluate
    # the objective at the same point: old := stop_gradient(own logps),
    # which is the seam's own rule when old is absent.  The KL reference
    # logps stay the captured constants (a different model's values).
    example = example.replace(old_per_token_logps=None)
  algo, pad_id, eos_id = load_algo_config(root, temperature)
  total_rows = int(example.completion_ids.shape[0])
  rows = total_rows if max_rows is None else min(max_rows, total_rows)
  log(f"model leaves={len(params)} rows={rows}/{total_rows} rows_per_step={rows_per_step} "
      f"beta={getattr(algo, 'beta', None)} epsilon={getattr(algo, 'epsilon', None)} "
      f"agg={getattr(algo, 'loss_agg_mode', None)} T={getattr(algo, 'temperature', None)}")
  graphdef, state = build_model(config, params)
  del params
  grad_fn = make_grad_fn(graphdef, algo, pad_id, eos_id)
  scale = batch_scale(example, algo)
  accumulator = None
  losses = []
  logps_rows = []
  started = time.perf_counter()
  for start in range(0, rows, rows_per_step):
    stop = min(start + rows_per_step, rows)
    (value, logps), grads = grad_fn(state, example_rows(example, slice(start, stop)))
    jax.block_until_ready(grads)
    losses.append(float(value))
    logps_rows.append(np.asarray(logps))
    accumulator = grads if accumulator is None else jax.tree.map(jnp.add, accumulator, grads)
    log(f"rows {start}..{stop - 1} unreduced_sum={float(value):.9e} elapsed={time.perf_counter() - started:.0f}s")
  accumulator = jax.tree.map(lambda g: g * scale, accumulator)
  flat = jax.tree_util.tree_flatten_with_path(accumulator, is_leaf=_is_variable)[0]
  gradient = {
      jax.tree_util.keystr(path): np.asarray(leaf[...] if _is_variable(leaf) else leaf)
      for path, leaf in flat
  }
  return {
      "gradient": gradient, "scale": scale, "losses": losses,
      "logps": np.concatenate(logps_rows, axis=0), "rows": rows,
      "example": example, "seconds": time.perf_counter() - started,
  }


def validate_logps(reference_logps: np.ndarray, root: Path, example, rows: int):
  captured = load_capture(root, "logps")[""]
  captured = np.asarray(captured, np.float64)[:rows]
  mask = np.asarray(example.completion_mask)[:rows].astype(bool)
  if captured.shape != reference_logps.shape:
    raise ValueError(f"logps shape {captured.shape} != reference {reference_logps.shape}")
  diff = np.abs(captured - reference_logps)[mask]
  return {
      "action_tokens": int(mask.sum()),
      "mean_abs_diff": float(diff.mean()),
      "max_abs_diff": float(diff.max()),
      "mean_abs_logp": float(np.abs(reference_logps[mask]).mean()),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--capture-a", type=Path, required=True)
  parser.add_argument("--capture-b", type=Path)
  parser.add_argument("--capture-b-name", default="gradient",
                      help="which tree of capture B is the gradient (e.g. stock_gradient)")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--rows-per-step", type=int, default=4)
  parser.add_argument("--max-rows", type=int)
  parser.add_argument("--model", default="qwen3_1p7b")
  parser.add_argument("--temperature", type=float, help="overrides/supplies the loss temperature")
  parser.add_argument("--params-from", type=Path,
                      help="a capture whose model_before holds the same parameters (hash-checked)")
  parser.add_argument("--also-dtype", choices=("bfloat16", "float32"),
                      help="after the main pass, redo the reference in this dtype in the same "
                           "process and report its distance to the main pass and to the captures "
                           "(same semantics, precision only)")
  parser.add_argument(
      "--captured-old-logps", action="store_true",
      help="keep the captured rollout logps as old (ratios and clips move with "
           "the fp64 forward's own logps); default evaluates at ratio 1",
  )
  args = parser.parse_args()
  log = lambda message: print(f"[fp64_reference] {message}", flush=True)  # noqa: E731
  config = getattr(qwen3_model.ModelConfig, args.model)()
  global COMPUTE_DTYPE_MAIN
  COMPUTE_DTYPE_MAIN = "fp64" if COMPUTE_DTYPE == "float64" else COMPUTE_DTYPE
  result = reference_gradient(args.capture_a, config, args.rows_per_step, args.max_rows, log,
                              temperature=args.temperature,
                              zero_tim_point=not args.captured_old_logps,
                              params_from=args.params_from)
  report = {
      "schema": "canon-p61-fp64-reference-v1",
      "compute_dtype": COMPUTE_DTYPE,
      "zero_tim_point": not args.captured_old_logps,
      "temperature": args.temperature,
      "capture_a": str(args.capture_a), "capture_b": str(args.capture_b) if args.capture_b else None,
      "rows": result["rows"], "scale": result["scale"], "losses": result["losses"],
      "seconds": result["seconds"],
      "logps_validation": validate_logps(result["logps"], args.capture_a, result["example"], result["rows"]),
  }
  if args.max_rows is None:
    gradient_a = load_capture(args.capture_a, "gradient")
    report["a_vs_fp64"] = compare_trees(result["gradient"], gradient_a)
    if args.capture_b:
      gradient_b = load_capture(args.capture_b, args.capture_b_name)
      report["b_vs_fp64"] = compare_trees(result["gradient"], gradient_b)
      report["a_vs_b"] = compare_trees(gradient_a, gradient_b)
  if args.also_dtype and args.max_rows is None:
    set_compute_dtype(args.also_dtype)
    second = reference_gradient(args.capture_a, config, args.rows_per_step, args.max_rows, log,
                                temperature=args.temperature,
                                zero_tim_point=not args.captured_old_logps,
                                params_from=args.params_from)
    tag = args.also_dtype
    report[f"{tag}_logps_validation"] = validate_logps(
        second["logps"], args.capture_a, second["example"], second["rows"])
    # The main pass is the reference in every comparison below.
    report[f"{tag}_vs_{COMPUTE_DTYPE_MAIN}"] = compare_trees(result["gradient"], second["gradient"])
    report[f"a_vs_{tag}"] = compare_trees(second["gradient"], gradient_a)
    if args.capture_b:
      report[f"b_vs_{tag}"] = compare_trees(second["gradient"], gradient_b)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
  log(f"logps validation {report['logps_validation']}")
  for key in sorted(report):
    if key.endswith(("_vs_fp64", "_vs_b")) or "_vs_" in key:
      value = report[key]
      if isinstance(value, dict) and "overall" in value:
        overall = value["overall"]
        log(f"{key}: rel_l2={overall['rel_l2']:.3e} one_minus_cos={overall['one_minus_cos']:.3e} "
            f"norm_ratio_error={overall['norm_ratio_error']:.3e}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
