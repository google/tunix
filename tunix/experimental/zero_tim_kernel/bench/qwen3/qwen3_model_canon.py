# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical Qwen3 model integrating Zero-TIM batch-invariant kernels.

This module provides `Qwen3Canon`, a drop-in counterpart to
`tunix.models.qwen3.model.Qwen3` (same parameter tree and call signature;
the one interface difference is the logits dtype, see "Logits contract"
below) that replaces individual operations with Zero-TIM kernels from
`tunix.experimental.zero_tim_kernel` to eliminate training-inference
mismatch (TIM):

1. `use_canon_rmsnorm`: Fixed-BF=128 left-to-right f32 accumulation RMSNorm
   (`pallas_rmsnorm` + `canonical_vjp.rmsnorm`) for `input_layernorm`,
   `post_attention_layernorm`, `q_norm`, `k_norm`, and `final_norm`.
2. `use_canon_qkv_proj`: Fixed-tile padded matmul + canonical VJP
   (`padded_matmul` + `canonical_vjp.matmul`) for `q_proj`, `k_proj`, `v_proj`.
3. `use_canon_rope`: Barriered float32 RoPE rotation preventing FMA fusion drift
   between decode (`T=1`), prefill (`T=L`), and `fwd+bwd`.
4. `use_canon_attention`: Fixed-tile exp2 online softmax attention
   (`canon_attention.flash_fwd_pallas`, A4): every query row is processed as
   part of a fixed `128-row x 128-key` Pallas tile program in key order with
   strict no-op masking (`rpa_diff` semantics), making KV-cached decode
   bitwise identical to cacheless prefill and learner forward whatever `T`,
   `S` or `B` are. (`canon_attention.ATTENTION_FWD` can select the earlier
   XLA `lax.scan` forward for A/B; it holds that contract only when the
   prefill length pads to the decode pad, see `canon_attention`. Its third
   value, `RPA_FWD`, runs this switch's attention on the engine's ragged
   paged attention kernel through `rpa_backend` -- the cross-engine
   measurement setting, see `rpa_attention_cached` /
   `rpa_attention_cacheless`.)
5. `use_canon_o_proj`: Fixed-tile padded matmul + canonical VJP for `o_proj`.
6. `use_canon_mlp_proj`: Fixed-tile padded matmul + canonical VJP for
   `gate_proj`, `up_proj`, and `down_proj`.
7. `use_canon_swiglu`: Padded Pallas SwiGLU (`padded_swiglu` +
   `canonical_vjp.swiglu`).
8. `use_fixed_lm_head`: P38 `FIXED_M=256` padded/chunked output head with
   ascending-chunk `lax.scan` VJP (`fixed_lm_head`).
9. `use_canon_logsoftmax`: Fixed 1024-tile 3-stage log-softmax / gathered
   logprobs (`canonical_logsoftmax`).
10. `use_fixed_order_reduce`: Rank-0-to-(TP-1) f32 TP reduction
    (`fixed_order_reduce`) when running over a tensor-parallel mesh.
11. `use_canon_norm_matmul`: P56.4.6 fused RMSNorm-prologue + fixed-tile matmul
    (`pallas_norm_matmul`) for `input_layernorm -> QKV` and
    `post_attention_layernorm -> Gate/Up`; the normalized rows live in VMEM
    scratch instead of an HBM tensor (requires 1, 2 and 6 at the same site).

All 11 switches are individually configurable via `CanonKernelConfig` to
support fine-grained per-kernel numeric and performance ablation studies while
sharing the exact same `nnx.Param` state tree as stock `Qwen3`.

Five module-level HBM / latency switches are numerically neutral in the forward
(bitwise, see
`qwen3_model_canon_test.test_optimization_switch_is_forward_bitwise_neutral_tp4`
and `test_forward_tp_sum_scatter_threshold_is_bitwise_tp4`) and exist only so
their cost can be A/B'd:

- `CHECKPOINT_ATTN_CORE`: `jax.checkpoint` around the cacheless
  `q_norm / k_norm -> RoPE -> attention` core during `Remat=NONE` training.
- `VOCAB_PARALLEL_LOGSOFTMAX`: canonical log-softmax / `token_logprobs` on the
  `P(None, 'tp')`-sharded `[M, V/TP]` logits without the `[M, V]` all-gather.
  It is only taken when every rank's `V/TP` shard is a whole 1024-column tile
  (`vocab_parallel_logsoftmax_admitted`), which makes it bitwise the library
  program; other shapes (Qwen3's `V=151936` at TP=2/4/8) use the gathered
  library path.
- `canon_attention.ATTENTION_BWD`: the backward of `canonical_online_attention`
  (a dense flash backward or the Pallas flash backward kernel; JAX autodiff
  of the XLA scan for the `XLA_FWD` forward). The forward kernel is the same
  object in every mode.
- `ROPE_FUSE_QK`: the canonical RoPE of the query and key heads as one program
  (`apply_rope_canonical_qk`) instead of two.
- `FORWARD_TP_SUM_MODE` / `FORWARD_TP_SUM_SCATTER_MIN_M`: the collective form
  (`fixed_order_reduce` GATHER / SCATTER / RING, identical operands in the
  identical association order) of the contract-parallel TP sums of `o_proj` /
  `down_proj`: GATHER below `FORWARD_TP_SUM_SCATTER_MIN_M = 4096` rows
  (`M = B * T` of the site: every decode step and e.g. all `B=4` benchmark
  programs), SCATTER from there on (Step 12).

`ROPE_PAD_ROWS` (the 64-row sequence padding inside the canonical RoPE body)
is deliberately NOT one of them: without it the `T=1` decode step and the
`T=L` prefill no longer agree bitwise on TPU (see its definition), so it is a
Zero-TIM requirement that only looks like a knob.

`HARD_ROUNDING` (env `ZERO_TIM_HARD_ROUNDING`, default `HARD_ROUNDING_ALL`)
is a bitmask that writes the `astype(bf16)` boundaries of four XLA-glue site
groups (RoPE, TP-sum adds, residual adds, the XLA attention scan) as
`lax.reduce_precision` rounding points that `--xla_allow_excess_precision`
cannot move. It is bitwise neutral under `--xla_allow_excess_precision=false`
and, with the default, makes the forward the same bits under either flag
value; `0` is the plain `astype` glue the attribution started from (see its
definition). The attention bit only matters for the `XLA_FWD` forward: the
Pallas forward kernel materializes its operands and its output.

Logits contract: `Qwen3Canon.compute_final_logits` returns the LM head's
native dtype (`config.dtype`, bf16; stock returns the same values upcast to
f32). `compute_log_softmax` / `compute_token_logprobs` / `next_token_logprobs`
accept either and upcast exactly where f32 is needed, so no f32 copy of the
logits is materialized in front of the Pallas log-softmax.

TP gradient convention: every `jax.shard_map` in this module runs with
`check_vma=False` and the learner differentiates OUTSIDE the maps
(`score_sequence_fwd_bwd`). JAX's transpose rule for such a map divides the
cotangent of an output whose spec leaves the TP axis unmentioned by the axis
size and `psum`s the cotangent of every input whose spec leaves it unmentioned.
`_fixed_order_tp_sum_with_vjp`, `_column_parallel_local` and
`_shard_map_with_local_vjp` are written against exactly that rule (see their
docstrings); `qwen3_model_canon_test.test_param_grads_match_stock_tp4` pins the
parameter gradients to stock `Qwen3` at TP=4.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.interpreters import pxla
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel import canonical_vjp
from tunix.experimental.zero_tim_kernel import fixed_lm_head
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul
from tunix.experimental.zero_tim_kernel import padded_swiglu
from tunix.experimental.zero_tim_kernel import pallas_norm_matmul
from tunix.experimental.zero_tim_kernel import pallas_rmsnorm
from tunix.experimental.zero_tim_kernel.bench.qwen3 import canon_attention
from tunix.experimental.zero_tim_kernel.bench.qwen3 import rpa_backend
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen3 import model as qwen3_stock
from tunix.utils import compat

pl = canonical_logsoftmax.pl
pltpu = canonical_logsoftmax.pltpu
K_MASK = qwen3_stock.K_MASK
LayerCache = qwen3_stock.LayerCache
Cache = qwen3_stock.Cache
RematConfig = qwen3_stock.RematConfig
ShardingConfig = qwen3_stock.ShardingConfig
ModelConfig = qwen3_stock.ModelConfig
shard = qwen3_stock.shard


def _remat_is(config: ModelConfig, kind: RematConfig) -> bool:
  """Whether `config.remat_config` is `kind` (stock accepts the enum or its value)."""
  return config.remat_config == kind or config.remat_config == kind.value


LOG2_E = canon_attention.LOG2_E

# Fixed KV-block granularity (`bkv_csz`) of `canonical_online_attention`; the
# rationale (fixed, not small: `bkv=512` mirrors `rpa_canonical`'s
# `(128, 512, 128, 512)` bundle and turns every `S <= 512` attention into one
# scan step) is documented next to the definition in `canon_attention`.
DEFAULT_ATTENTION_KV_BLOCK_SIZE = (
    canon_attention.DEFAULT_ATTENTION_KV_BLOCK_SIZE
)

# Forward reducer for the contract-parallel TP sums of `o_proj` / `down_proj`.
#
# All `fixed_order_reduce` modes add the identical operands in the identical
# association order `((p0 + p1) + p2) + p3` and are bitwise equal
# (`fixed_order_reduce_test.test_modes_are_bitwise_equal_on_random_operands`),
# so the choice is a pure latency / HBM knob and never touches program
# identity (Zero-TIM). GATHER = one `all_gather` of the bf16 `[M, N]` partial +
# rank-ordered adds on every rank. SCATTER (`all_to_all` + rank-ordered adds of
# one `M/TP` row block + tiled `all_gather`) moves 2x fewer ICI bytes, but a
# same-process A/B on 4x v5p (TP=4, B=4; SCATTER for `M >= 256` vs GATHER
# everywhere, interleaved, 2 repeats) measured:
#   * latency: prefill `M=1024` `-0.3 ms` (`5.45` vs `5.75`, -5%); every other
#     shape (decode, `M=256/896` prefill, Fwd+Bwd) within the `±0.2 ms` noise;
#   * Temp HBM: Sample `L=8, T=256` `+9.5 MB` (`126.89` vs `117.36`, the
#     `M=896` prefill phase), prefill `L=8, M=256` `+2.0 MB` (`31.87` vs
#     `29.90`); XLA keeps more buffers live around the two collectives than
#     around the single gather, which outweighs the smaller stack.
# A `-5%` prefill gain that costs `+8%` Sample HBM is not worth it on the
# rollout path, so the small sites keep GATHER; the `M >= 2048` re-evaluation
# (Step 12) made SCATTER the default from `FORWARD_TP_SUM_SCATTER_MIN_M` rows
# on, see below.
FORWARD_TP_SUM_MODE = fixed_order_reduce.GATHER

# Step 12: the `M >= 2048` re-evaluation of SCATTER. Contract-parallel TP sums
# over at least this many rows (`M = B * T` of the site) use SCATTER instead of
# `FORWARD_TP_SUM_MODE`; smaller sites (every decode step, every `B=4`
# benchmark program) are unchanged. The GATHER form moves `(TP-1) * M * N * 2`
# bytes into every rank per site, i.e. `48 MB` at `M=8192, N=1024, TP=4` for
# each of the 16 `o_proj` / `down_proj` sites of an 8-layer forward; SCATTER
# moves `2(TP-1)/TP` of one partial. Rows the TP degree does not divide fall
# back to GATHER inside `fixed_order_reduce` (static shape decision, program
# identity unchanged). `None` = never (the Step-11 program; the forced-GATHER
# arm of the A/B test).
# Measured on 4x v5p (Step 12, `test_forward_tp_sum_scatter_tpu_ab`, `L=8,
# B=32, T=256`, same process): `4096` vs `None` = Sample `43.94 -> 41.82 ms`,
# Prefill `14.38 -> 11.73`, Fwd+Bwd `31.64 -> 29.16`; prefill and Fwd+Bwd
# logprobs bitwise EQUAL on TPU; Code smaller in all three programs; Arena
# `+0.03 / +0.02 / +4.00 MB` (the Fwd+Bwd `+4.00` is exactly one `[M/TP, N]`
# bf16 shard). Accepted as the default with that `+4 MB` as the known cost;
# at `M=4096` (`B=16`) the Prefill arena is `7.9 MB` smaller. The Zero-TIM
# suite's `B=4, T=64` workloads never reach the threshold, so the A/B test is
# the bitwise guard of this switch: it re-asserts GATHER == SCATTER at `B=32`
# on every TPU round.
FORWARD_TP_SUM_SCATTER_MIN_M: int | None = 4096


def _forward_tp_sum_mode(rows: int) -> str:
  """Reducer form of a contract-parallel TP sum over `rows` (= `B * T`) rows."""
  min_m = FORWARD_TP_SUM_SCATTER_MIN_M
  if min_m is not None and rows >= min_m:
    return fixed_order_reduce.SCATTER
  return FORWARD_TP_SUM_MODE


# Candidate (10): Checkpoint the cacheless attention core
# (`q_norm / k_norm -> RoPE -> canonical_online_attention`) during `Remat=NONE`
# training so only `(q_raw, k_raw, value_proj)` is kept across layers instead
# of also retaining post-norm/post-RoPE `q_f32` (f32), `sp=512`-padded
# `k_blocks`/`v_blocks`/`mask_blocks`, `acc_final` (f32), and RoPE `sin/cos`.
CHECKPOINT_ATTN_CORE = True

# Candidate (5b): Vocab-parallel canonical log-softmax / `token_logprobs` on
# `P(None, 'tp')`-sharded `[M, V/TP]` logits. Eliminates the full-vocabulary
# `[M, V]` f32 `all_gather` in Sample, Prefill, and Fwd+Bwd, runs Stage 1 tile
# reductions only on the local `V/TP` columns, `all_gather`s only the tiny
# `(M, tiles_local)` f32 tile summaries before Stage 2, and keeps the backward
# residual and `grad_logits` sharded as `[M, V/TP]`.
VOCAB_PARALLEL_LOGSOFTMAX = True

# D1: rotate the query and the key heads in ONE canonical RoPE program
# (`apply_rope_canonical_qk`: concatenate along the head axis, one
# `shard_map`, split) instead of one program per tensor. RoPE never mixes
# heads and every product / sum / cast is the same IEEE f32 element-wise op,
# so the bits are identical (XLA already shares the position `sin / cos`
# between the two tensors; the merge halves the RoPE fusion groups per layer,
# which is where the launch-bound Sample path pays). Same-process A/B on 4x
# v5p (`L=8, B=4, T=256`, TP=4): Sample `22.03 -> 21.32 ms` (stock `14.3`
# either way), Prefill / Fwd+Bwd within noise, Code `34.2 -> 30.7 MB`, arena
# not larger. Accepted as the default (Step 11) with one known cost: in the
# cross-run comparison the `L=2` Fwd+Bwd (`Remat=NONE`) arena grew by
# `+0.22 MB` (`T=64`) / `+1.14 MB` (`T=256`), the only HBM cells of 15 that
# did not shrink; the gain is a fixed per-call cost, so it does not touch the
# `B=16/32` steady-state throughput gap (D0).
ROPE_FUSE_QK = True

# `ROPE_PAD_ROWS`: rows the sequence axis is padded to inside
# `_apply_rope_canonical_local` before the barriered products (`None`: no
# padding). NOT a free latency knob: the padding is what keeps the XLA:TPU
# element-wise RoPE program of a `T=1` decode step bitwise the `T=L` prefill
# one. Measured on 4x v5p (Step 11, `qwen3_0p6b_2l`, TP=4, same process,
# `test_rope_glue_switches_tpu_ab`): with `None`, drift regimes 1a-1c
# (decode vs prefill) show `max |diff| = 1.6e-2 .. 3.1e-2` with only `6-8%`
# bitwise-equal logprobs, while regime 2 (B=1 vs B=4 prefill) and 3a (fwd vs
# fwd+bwd) stay `0`; the CPU bitwise tests cannot see this. Keep 64 (the
# branch's value); the switch only exists so the A/B can be re-run.
ROPE_PAD_ROWS: int | None = 64

# `HARD_ROUNDING` (default: all four groups): which groups of XLA-glue bf16
# boundaries of the forward are written as `lax.reduce_precision(x_f32, 8, 7)`
# instead of `astype(bf16)` (`canon_attention.hard_round` / `hard_upcast`).
#
# Why: the Pallas kernels materialize their bf16 outputs, so their rounding
# points are fixed; the XLA glue between them (`astype` casts of RoPE, the
# attention scan, the rank-ordered TP-sum adds, the residual adds) is subject
# to `--xla_allow_excess_precision`. Under its default (`true`) XLA may drop an
# `f32 -> bf16 -> f32` convert pair and keep bf16 intermediates in f32 inside a
# fusion, so WHERE a rounding happens depends on the fusion decisions of each
# program: two programs over different `M = B * T` can round at different
# points and no longer agree bitwise. Measured (TPU, TP=4, `qwen3_0p6b_2l`,
# default flags, plain `astype` glue = mask `0`): `canon_all_enabled` regimes
# 1a / 3a stay `0`, but the `B=4, T=64` prefill program drifts from every
# other one (1b / 2 `max |diff| = 3.1e-2`, `0%` bitwise-equal logprobs; 1c
# `1.6e-2`, `6%`); under `--xla_allow_excess_precision=false` every regime is
# `0`. `reduce_precision` is the same round-to-nearest-even expressed as an
# arithmetic op XLA keeps under either flag value, so routing a site group
# through it pins its rounding points.
#
# Attribution (`test_reduce_precision_glue_tpu_ab`, same setup): under the
# default flags `ROPE + TP_SUM` restores every regime to `0` AND makes the
# `B=4, T=64` prefill and fwd+bwd logprobs bit-identical to the strict-flag
# program (equal sha256 fingerprints); `ALL` and the complements without
# RESIDUAL / ATTENTION do the same (those two bits are no-ops on top: XLA
# already rounds there). The drift itself is the bf16 add chain of
# `fixed_order_reduce` carried in f32 (in the small programs through the
# residual add), so `only TP_SUM` (and even `only RESIDUAL`, which pins the
# chain's last rounding) already gives `0` drift, but those programs stay
# flag-dependent: `only TP_SUM` / `ALL - ROPE` share one fingerprint, `only
# ROPE` / `ALL - TP_SUM` another, neither the strict one, because the elided
# RoPE `sin / cos` bf16 round trip moves the bits of every program uniformly.
# Under the strict flag every mask is bitwise the unmodified program (`0` vs
# `ALL` on CPU in `test_optimization_switch_is_forward_bitwise_neutral_tp4`;
# all 11 masks in the strict-flag branch of
# `test_reduce_precision_glue_tpu_ab`), so the default is `ALL`: the forward
# no longer depends on the flag, and `0` remains for re-running the
# attribution. The RESIDUAL / ATTENTION bits are kept on as defence in depth:
# they add elementwise `reduce_precision` ops to fusions that already round
# there (not separately timed; the scaling tables are taken with the default).
HARD_ROUNDING_ROPE = 1  # `_apply_rope_canonical_local` casts (in, sin/cos, out)
HARD_ROUNDING_TP_SUM = 2  # the rank-ordered adds of `fixed_order_reduce`
HARD_ROUNDING_RESIDUAL = 4  # the two residual adds of `CanonDecoderLayer`
HARD_ROUNDING_ATTENTION = 8  # `online_softmax_scan` casts (`XLA_FWD` only)
HARD_ROUNDING_ALL = (
    HARD_ROUNDING_ROPE
    | HARD_ROUNDING_TP_SUM
    | HARD_ROUNDING_RESIDUAL
    | HARD_ROUNDING_ATTENTION
)
HARD_ROUNDING_ENV = 'ZERO_TIM_HARD_ROUNDING'


def _hard_rounding_from_env() -> int:
  """Reads the `ZERO_TIM_HARD_ROUNDING` bitmask (default `HARD_ROUNDING_ALL`).

  Returns:
    The bitmask of `HARD_ROUNDING_*` site groups to hard-round; `0` is the
    plain `astype` glue.

  Raises:
    RuntimeError: If the value is not an integer in `[0, HARD_ROUNDING_ALL]`
      (fail closed).
  """
  raw = os.environ.get(HARD_ROUNDING_ENV, '')
  if not raw:
    return HARD_ROUNDING_ALL
  try:
    mask = int(raw)
  except ValueError as e:
    raise RuntimeError(
        f'{HARD_ROUNDING_ENV} must be an integer bitmask in'
        f' [0, {HARD_ROUNDING_ALL}], got {raw!r}'
    ) from e
  if not 0 <= mask <= HARD_ROUNDING_ALL:
    raise RuntimeError(
        f'{HARD_ROUNDING_ENV} must be in [0, {HARD_ROUNDING_ALL}], got {mask}'
    )
  return mask


HARD_ROUNDING: int = _hard_rounding_from_env()


def _hard(site: int) -> bool:
  """Whether the `HARD_ROUNDING` site group `site` is on (read at trace time)."""
  return bool(HARD_ROUNDING & site)


def _cast(x: jax.Array, dtype: jax.typing.DTypeLike, hard: bool) -> jax.Array:
  """Producer-side `astype(dtype)`; a flag-proof rounding point when `hard`."""
  if hard:
    return canon_attention.hard_round(x, dtype)
  return x.astype(dtype)


def _upcast(x: jax.Array, hard: bool) -> jax.Array:
  """Consumer-side `astype(f32)`; re-asserts a bf16 rounding when `hard`."""
  if hard:
    return canon_attention.hard_upcast(x)
  return x.astype(jnp.float32)


def _hard_add(a: jax.Array, b: jax.Array) -> jax.Array:
  """`a + b` (in the promoted dtype) with operands and sum hard-rounded.

  The result dtype follows `jnp.result_type(a, b)` exactly like the `a + b`
  this replaces, so the switch cannot change a program's dtypes: at every site
  today both operands are `config.dtype` (bf16); an f32 operand would keep the
  sum in f32 (`hard_round` of an f32 to f32 is the identity).

  Args:
    a: Left operand (the running accumulator in the TP-sum chain).
    b: Right operand.

  Returns:
    The hard-rounded sum.
  """
  out_dtype = jnp.result_type(a, b)
  total = canon_attention.hard_upcast(a) + canon_attention.hard_upcast(b)
  return canon_attention.hard_round(total, out_dtype)


def _residual_add(a: jax.Array, b: jax.Array, hard: bool) -> jax.Array:
  """The decoder layer's residual add (`a + b`), hard-rounded when `hard`."""
  return _hard_add(a, b) if hard else a + b


ABLATION_SWITCH_NAMES: tuple[str, ...] = (
    'use_canon_rmsnorm',
    'use_canon_qkv_proj',
    'use_canon_rope',
    'use_canon_attention',
    'use_canon_o_proj',
    'use_canon_mlp_proj',
    'use_canon_swiglu',
    'use_fixed_lm_head',
    'use_canon_logsoftmax',
    'use_fixed_order_reduce',
    'use_canon_norm_matmul',
)


@dataclasses.dataclass(slots=True, frozen=True)
class CanonKernelConfig:
  """Per-kernel switches for Zero-TIM canonical execution and ablation tests.

  `use_canon_norm_matmul` is a pure fusion switch: it only takes effect at a
  site where `use_canon_rmsnorm` and the site's projection switch
  (`use_canon_qkv_proj` / `use_canon_mlp_proj`) are also on, so the Add-One-In
  row `only_use_canon_norm_matmul` is by construction identical to
  `canon_all_disabled` (bitwise under the strict XLA flag: as an `only_*` row
  it carries the hard-rounded residual adds of `HARD_ROUNDING_RESIDUAL`, which
  `canon_all_disabled` leaves to XLA), and the Leave-One-Out row
  `without_use_canon_norm_matmul` is the un-fused two-kernel chain whose
  `|LP - Canon|` must be `0.000e+00`.
  """

  use_canon_rmsnorm: bool = True
  use_canon_qkv_proj: bool = True
  use_canon_rope: bool = True
  use_canon_attention: bool = True
  use_canon_o_proj: bool = True
  use_canon_mlp_proj: bool = True
  use_canon_swiglu: bool = True
  use_fixed_lm_head: bool = True
  use_canon_logsoftmax: bool = True
  use_fixed_order_reduce: bool = True
  use_canon_norm_matmul: bool = True
  attention_kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE
  interpret: bool | None = None

  def resolve_interpret(self) -> bool:
    if self.interpret is not None:
      return self.interpret
    return jax.default_backend() == 'cpu'

  def any_enabled(self) -> bool:
    """Whether at least one ablation switch is on (`all_disabled` is stock)."""
    return any(getattr(self, name) for name in ABLATION_SWITCH_NAMES)

  @classmethod
  def all_enabled(
      cls,
      *,
      attention_kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    return cls(
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def all_disabled(
      cls,
      *,
      attention_kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    kwargs = {name: False for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def only(
      cls,
      switch_name: str,
      *,
      attention_kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    """Enables ONLY `switch_name` (Add-One-In ablation)."""
    if switch_name not in ABLATION_SWITCH_NAMES:
      raise ValueError(
          f'Unknown switch {switch_name!r}; valid: {ABLATION_SWITCH_NAMES}'
      )
    kwargs = {name: name == switch_name for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )

  @classmethod
  def without(
      cls,
      switch_name: str,
      *,
      attention_kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
      interpret: bool | None = None,
  ) -> CanonKernelConfig:
    """Enables all canonical kernels EXCEPT `switch_name` (Leave-One-Out)."""
    if switch_name not in ABLATION_SWITCH_NAMES:
      raise ValueError(
          f'Unknown switch {switch_name!r}; valid: {ABLATION_SWITCH_NAMES}'
      )
    kwargs = {name: name != switch_name for name in ABLATION_SWITCH_NAMES}
    return cls(
        **kwargs,
        attention_kv_block_size=attention_kv_block_size,
        interpret=interpret,
    )


def _round_up(x: int, tile: int) -> int:
  return ((x + tile - 1) // tile) * tile


def resolve_or_build_contract(
    config: ModelConfig,
    tp_size: int = 1,
) -> model_contracts.ModelContract:
  """Returns a registered `ModelContract` or builds a compatible one."""
  for contract in model_contracts.CONTRACTS.values():
    if (
        contract.hidden_size == config.embed_dim
        and contract.intermediate_size == config.hidden_dim
        and contract.num_attention_heads == config.num_heads
        and contract.num_kv_heads == config.num_kv_heads
        and contract.head_dim == config.head_dim
        and contract.tp_size == tp_size
        and config.vocab_size == model_contracts.VOCAB_SIZE
    ):
      return contract

  block_m = 128
  if config.embed_dim >= 1024 and tp_size <= 4:
    block_n = 256
    block_k = 256
  else:
    block_n = 128
    block_k = 128

  q_width = (config.num_heads * config.head_dim) // tp_size
  kv_width = (config.num_kv_heads * config.head_dim) // tp_size
  mlp_width = config.hidden_dim // tp_size
  hidden = config.embed_dim
  local_vocab = config.vocab_size // tp_size

  k_widths = {hidden, q_width, mlp_width}
  n_widths = {q_width, kv_width, hidden, mlp_width, local_vocab}

  k_pad: dict[int, int] = {}
  for w in k_widths:
    for tile in (block_k, fixed_lm_head.BK):
      if w % tile != 0:
        k_pad[w] = _round_up(w, tile)

  n_pad: dict[int, int] = {}
  for w in n_widths:
    for tile in (block_n, fixed_lm_head.BN):
      if w % tile != 0:
        n_pad[w] = _round_up(w, tile)

  sw_pad: dict[int, int] = {}
  if mlp_width % model_contracts.SWIGLU_BF != 0:
    sw_pad[mlp_width] = _round_up(mlp_width, model_contracts.SWIGLU_BF)

  return model_contracts.ModelContract(
      name=f'qwen3_d{hidden}_f{config.hidden_dim}_tp{tp_size}',
      block_m=block_m,
      block_n=block_n,
      block_k=block_k,
      matmul_k_padding=k_pad,
      matmul_n_padding=n_pad,
      swiglu_feature_padding=sw_pad,
      hidden_size=hidden,
      intermediate_size=config.hidden_dim,
      num_attention_heads=config.num_heads,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
      tp_size=tp_size,
  )


def _active_tp_size_and_mesh(
    tp_axis: str = 'tp',
) -> tuple[int, jax.sharding.Mesh | None]:
  """Returns `(tp_size, mesh)` if a physical mesh with `tp_axis` is active."""
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh is not None and not mesh.empty and tp_axis in mesh.shape:
    return int(mesh.shape[tp_axis]), mesh
  return 1, None


def _fixed_order_tp_sum_with_vjp(
    partial: jax.Array,
    axis_name: str,
    count: int,
    *,
    mode: str,
) -> jax.Array:
  """`fixed_order_tp_sum` forward with the exact VJP of a cross-rank sum.

  Forward: the rank-ordered sum `((p_0 + p_1) + p_2) + ...` of the per-rank
  partials (`fixed_order_reduce`, bitwise identical on every rank).

  Backward: `y = sum_r partial_r`, so `d partial_r = ct` (the full output
  cotangent) on every rank. The learner differentiates OUTSIDE the
  `check_vma=False` shard_map that calls this, and JAX's transpose rule for
  such a map (`_shard_map_transpose`) divides the cotangent of an output whose
  `out_specs` leaves a mesh axis unmentioned by that axis' size and `psum`s the
  input cotangents over the axes *their* specs leave unmentioned. For an axis
  over which the body is merely replicated (e.g. a size-1 or data-parallel
  `fsdp`) the two cancel; for the TP axis the body's cross-rank sum is what
  must transpose into the broadcast `d partial_r = ct`, so a custom VJP that
  returns the cotangent unscaled hands `ct / TP` to the local matmul. The
  previous identity VJP did exactly that and scaled every gradient flowing
  through `o_proj`, `down_proj` and the embedding gather by `1/TP` (measured
  `0.25x` on the CPU TP=4 test, see `test_param_grads_match_stock_tp4`).
  Multiplying by `count` (the TP axis size only, never the full mesh size)
  restores it exactly for power-of-two TP degrees and needs no collective,
  whereas a `psum` of the identical per-rank copies would re-round the bf16
  chain `(c/4 + c/4) + c/4 + c/4` and cost an all-reduce per site.

  Args:
    partial: This rank's `[M, N]` bf16 partial.
    axis_name: TP mesh axis.
    count: TP degree (the size of `axis_name`).
    mode: `fixed_order_reduce.RING` / `GATHER` / `SCATTER` (bitwise equal).

  Returns:
    The replicated `[M, N]` sum.
  """

  # `HARD_ROUNDING_TP_SUM`: every rank-step add rounds through
  # `reduce_precision` (same operands, same association order). The library's
  # default `add` already pins the rounding (`fixed_order_reduce.hard_bf16_add`,
  # bitwise the same arithmetic as `_hard_add` on bf16); the bit off requests
  # the plain `+` chain explicitly so the A/B attribution runs keep their
  # meaning.
  add = (
      _hard_add if _hard(HARD_ROUNDING_TP_SUM) else fixed_order_reduce.plain_add
  )

  @jax.custom_vjp
  def op(p):
    return fixed_order_reduce.fixed_order_tp_sum(
        p, axis_name, count, mode=mode, add=add
    )

  def fwd(p):
    return op(p), None

  def bwd(_, cotangent):
    full = cotangent * jnp.asarray(count, dtype=cotangent.dtype)
    return (full.astype(partial.dtype),)

  op.defvjp(fwd, bwd)
  return op(partial)


def _column_parallel_local(local, *args):
  """Runs a TP-local column-parallel body; replicated-operand grads come from JAX.

  The TP-replicated operands (hidden state `x`, RMSNorm gamma) enter the
  enclosing shard_map with `P(None, ...)`. Their TP-sharded consumer yields one
  partial cotangent per rank, and JAX's transpose of a `check_vma=False`
  shard_map `psum`s the cotangent of every input whose spec leaves the TP axis
  unmentioned -- exactly the column-parallel `dx` all-reduce.
  `fixed_order_reduce.p59_column_parallel` additionally completes those
  cotangents with a fixed-order f32 TP sum *inside* the body; that is correct
  when the gradient is taken inside P59's manual map (no outer transpose), but
  stacked on top of the transpose `psum` it multiplied `dx` / `dgamma` by TP
  (measured `4.0x` on every RMSNorm gamma in `canon_all_enabled` and up to
  `11x` on the embedding in single-switch ablations at TP=4). Not attaching it
  also removes one f32 `[TP, M, D]` `all_gather` plus its barriered sum per
  site from the backward (`-4.TP.M.D` bytes transient HBM, `(TP-1).M.D.4`
  fewer ICI bytes); the remaining `psum` moves `~2(TP-1)/TP.M.D.2` bytes.

  Args:
    local: The TP-local body.
    *args: Its operands (replicated or TP-sharded alike).

  Returns:
    `local(*args)`.
  """
  return local(*args)


def _shard_map_with_local_vjp(
    local_fn,
    *,
    mesh: jax.sharding.Mesh | None,
    in_specs: Tuple[P, ...],
    out_spec: P,
    differentiable: Tuple[bool, ...],
    residual_dtypes: Tuple[jaxtyping.DTypeLike | None, ...] | None = None,
):
  """`jax.shard_map` of a TP-replicated-in/out body with a collective-free VJP.

  `final_norm` (3D RMSNorm) and the canonical log-softmax see identical inputs
  on every TP rank and produce identical outputs; they only run under
  `shard_map` so that their Pallas kernels see the per-device array. JAX's
  transpose rule for a `check_vma=False` shard_map cannot know that: it divides
  the output cotangent by TP and `psum`s the input cotangents over TP, i.e. one
  all-reduce of an already rank-identical `[M, V]` f32 (log-softmax, 32 MB at
  `M=1024, V=8192`) or `[M, D]` bf16 (RMSNorm) tensor per site -- pure ICI
  traffic, a second `[M, V]` transient for the all-reduce output, and a
  re-rounding of the `(c/4 + c/4) + c/4 + c/4` chain.

  This coat keeps the forward shard_map verbatim (bitwise identical primal) and
  defines the VJP explicitly: the residual is the replicated primal inputs
  themselves (no copy; optionally stored in `residual_dtypes` when the caller
  guarantees an exact round trip, e.g. bf16 for logits that were produced in
  bf16), and the backward evaluates `jax.vjp(local_fn)` inside a second
  *forward* shard_map, so no transpose rule is ever applied to the mapped body.
  Reduced-precision residuals are upcast to the primal dtype *before*
  differentiating, so the cotangents keep the primal dtype and value. Cost: the
  body's primal is re-traced in the backward (`canonical_rmsnorm` in XLA; the
  log-softmax Stages 1+2 normalizer, one HBM pass over `[M, V]`), far below the
  all-reduce it replaces. With `mesh=None` (TP=1) the body runs unmapped and
  only the residual policy applies.

  Args:
    local_fn: The per-rank body, returning a single array.
    mesh: The TP mesh, or `None` when not running tensor-parallel.
    in_specs: One replicated spec per operand.
    out_spec: The replicated output spec.
    differentiable: Per operand, whether a cotangent must be produced (False for
      integer operands such as token ids).
    residual_dtypes: Optional per-operand storage dtype of the residual; `None`
      keeps the primal.

  Returns:
    A callable `f(*args)` equal to `shard_map(local_fn)(*args)` in forward.
  """
  n = len(in_specs)
  differentiable = tuple(bool(flag) for flag in differentiable)
  if len(differentiable) != n:
    raise ValueError(
        f'differentiable has {len(differentiable)} flags for {n} operands'
    )
  if residual_dtypes is None:
    residual_dtypes = (None,) * n
  if len(residual_dtypes) != n:
    raise ValueError(
        f'residual_dtypes has {len(residual_dtypes)} entries for {n} operands'
    )
  diff_idx = tuple(i for i in range(n) if differentiable[i])

  def run_fwd(*args):
    if mesh is None:
      return local_fn(*args)
    return jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=tuple(in_specs),
        out_specs=out_spec,
        check_vma=False,
    )(*args)

  def call(*args):
    if len(args) != n:
      raise ValueError(f'expected {n} operands, got {len(args)}')
    primal_dtypes = tuple(a.dtype for a in args)

    def local_vjp(*flat):
      operands = list(flat[:n])
      cotangent = flat[n]
      # Upcast reduced-precision residuals BEFORE differentiating so that the
      # pullback produces cotangents in the primal dtype (an exact round trip
      # by the caller's guarantee), not in the storage dtype.
      for i in diff_idx:
        operands[i] = operands[i].astype(primal_dtypes[i])

      def body(*diff_args):
        full = list(operands)
        for i, value in zip(diff_idx, diff_args):
          full[i] = value
        return local_fn(*full)

      _, pullback = jax.vjp(body, *[operands[i] for i in diff_idx])
      return pullback(cotangent)

    def run_bwd(*flat):
      if mesh is None:
        return local_vjp(*flat)
      return jax.shard_map(
          local_vjp,
          mesh=mesh,
          in_specs=(*in_specs, out_spec),
          out_specs=tuple(in_specs[i] for i in diff_idx),
          check_vma=False,
      )(*flat)

    @jax.custom_vjp
    def op(*a):
      return run_fwd(*a)

    def fwd(*a):
      residual = tuple(
          value if dtype is None else value.astype(dtype)
          for value, dtype in zip(a, residual_dtypes)
      )
      return run_fwd(*a), residual

    def bwd(residual, cotangent):
      cts = run_bwd(*residual, cotangent)
      out: list[jax.Array | None] = [None] * n
      for i, ct in zip(diff_idx, cts):
        out[i] = ct
      return tuple(out)

    op.defvjp(fwd, bwd)
    return op(*args)

  return call


def _canon_matmul_2d(
    x_2d: jax.Array,
    w_2d: jax.Array,
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> jax.Array:
  """Runs 2D bf16 fixed-tile padded matmul with canonical VJP."""
  x_bf16 = x_2d.astype(jnp.bfloat16)
  w_bf16 = w_2d.astype(jnp.bfloat16)
  bm = contract.block_m if block_m is None else block_m
  bn = contract.block_n if block_n is None else block_n
  bk = contract.block_k if block_k is None else block_k

  def forward(a: jax.Array, b: jax.Array) -> jax.Array:
    return padded_matmul.matmul(
        a,
        b,
        contract=contract,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=bm,
        block_n=bn,
        block_k=bk,
    )

  out_bf16 = canonical_vjp.matmul(
      x_bf16, w_bf16, forward=forward, contract=contract
  )
  return out_bf16.astype(out_dtype)


def _canon_fused_matmul_2d(
    x_2d: jax.Array,
    w_parts: Tuple[jax.Array, ...],
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Horizontally fused `x @ concat(w_parts, axis=1)` with a copy-free residual.

  Forward is the exact same program as
  `_canon_matmul_2d(x, jnp.concatenate(w_parts, axis=1))`: one fixed-tile
  `padded_matmul` over the concatenated output columns. Output column tiles
  are independent, so every element keeps the accumulation chain of the
  un-fused per-projection kernel (bitwise identical, `0.000e+00`).

  HBM Optimization (fused-weight residual):
    `canonical_vjp.matmul` stores its `(a, b)` operands as the `custom_vjp`
    residual. With `b = concat([W_q, W_k, W_v])` (or `[W_gate, W_up]`) that
    residual is a *fresh* array - a second copy of ~2/3 of the layer's
    projection weights that stays live from the forward until the backward
    (`+5 MB/layer` on Qwen3-0.6B@TP4, `~63 MB/layer` on Qwen3-8B@TP4; measured
    `+38 MB` Fwd+Bwd Temp HBM at `L=8, Remat=NONE`). This coat keeps the
    original parameter views `w_parts` as the residual (no extra memory: they
    alias the `nnx.Param`s) and re-materialises the concatenation only as a
    short-lived transient inside the backward, right before the same
    `plain_matmul_pullback` / K-block replica pullback that
    `canonical_vjp.matmul` would run on the fused weight.

    Measured on 4x TPU v5p (TP=4, B=4, `Remat=NONE`): Fwd+Bwd Temp HBM
    `L=8, T=256`: 364.61 -> 306.92 MB (-58 MB); `L=8, T=64`: 177.09 -> 115.30
    MB (-62 MB); latency and all Zero-TIM drift checks unchanged
    (`0.000e+00`). The gain only materialised once the backward concatenation
    was made non-CSE-able (see the `optimization_barrier` note in `op_bwd`);
    `Remat=BLOCK` is unaffected because the forward is recomputed anyway.

  Args:
    x_2d: `[M, K]` input rows.
    w_parts: Weights `[K, N_i]` sharing the contracting dim, concatenated along
      the output axis.
    contract: Model contract (tile policy) of the Pallas matmul.
    interpret: Pallas interpret mode.
    out_dtype: Output dtype.

  Returns:
    `[M, sum(N_i)]` matmul output.
  """
  x_bf16 = x_2d.astype(jnp.bfloat16)
  parts = tuple(w.astype(jnp.bfloat16) for w in w_parts)
  split_points = []
  offset = 0
  for w in parts[:-1]:
    offset += int(w.shape[1])
    split_points.append(offset)

  def forward(a: jax.Array, b: jax.Array) -> jax.Array:
    return padded_matmul.matmul(
        a,
        b,
        contract=contract,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=contract.block_m,
        block_n=contract.block_n,
        block_k=contract.block_k,
    )

  @jax.custom_vjp
  def op(a: jax.Array, ws: Tuple[jax.Array, ...]) -> jax.Array:
    return forward(a, jnp.concatenate(ws, axis=1))

  def op_fwd(a: jax.Array, ws: Tuple[jax.Array, ...]):
    # Residual = original (un-concatenated) weight views, never the fused copy.
    return forward(a, jnp.concatenate(ws, axis=1)), (a, ws)

  def op_bwd(residual, cotangent):
    a, ws = residual
    # Re-materialise the fused weight as a *backward-local* transient. Routing
    # the weight views through an `optimization_barrier` tied to `cotangent`
    # (a) gives this concatenation operands distinct from the forward's, so
    # XLA CSE cannot merge the two `concatenate` ops and thereby keep the
    # forward's fused copy alive across the whole forward -> backward span
    # (measured: without the barrier Fwd+Bwd Temp HBM did not drop at all),
    # and (b) prevents the scheduler from hoisting it ahead of the backward.
    ws_late, cotangent = lax.optimization_barrier((ws, cotangent))
    w_fused = jnp.concatenate(ws_late, axis=1)
    if canonical_vjp.plain_matmul_vjp_enabled():
      da, dw = canonical_vjp.plain_matmul_pullback(a, w_fused, cotangent)
    else:
      _, pullback = jax.vjp(
          lambda a_, b_: canonical_vjp.canonical_matmul(
              a_, b_, contract=contract
          ),
          a,
          w_fused,
      )
      da, dw = pullback(cotangent)
    dws = tuple(jnp.split(dw, split_points, axis=1))
    return da, dws

  op.defvjp(op_fwd, op_bwd)
  return op(x_bf16, parts).astype(out_dtype)


def _norm_matmul_admits(
    d_in: int, contract: model_contracts.ModelContract
) -> bool:
  """Whether the fused P56.4.6 kernel can take `[*, d_in]` rows unpadded.

  The fused kernel normalizes over the full contracting width `K` inside the
  kernel (`sumsq / K`), so unlike `padded_matmul` it can never zero-pad `K`:
  that would change the RMS mean. It also needs `K` to divide the matmul
  `block_k` and the RMSNorm `BF` block. Sites that do not qualify keep the
  un-fused two-kernel chain (a static-shape decision).

  Args:
    d_in: Contracting width `K` of the site.
    contract: Model contract (tile policy) of the Pallas kernels.

  Returns:
    Whether the fused norm+matmul kernel admits this site.
  """
  return d_in % contract.block_k == 0 and d_in % pallas_rmsnorm.BF == 0


def _canon_fused_norm_matmul_2d(
    x_2d: jax.Array,
    gamma: jax.Array,
    w_parts: Tuple[jax.Array, ...],
    *,
    norm_eps: float,
    contract: model_contracts.ModelContract,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Fused `rmsnorm(x, gamma) @ concat(w_parts, axis=1)` (P56.4.6) bench coat.

  Forward: ONE `pallas_norm_matmul.norm_matmul` call. Its prologue normalizes
  each 128-row block with the verbatim P22.XH arithmetic of `pallas_rmsnorm`
  (f32 promote, BF=128-blocked left-to-right sum of squares,
  `rsqrt(sumsq / K + eps)`, `x * inv * gamma`, one bf16 cast) into VMEM
  scratch and then runs the verbatim P22.XE tile loop of `pallas_matmul`
  (bf16 tiles, f32 accumulator, the same sequential `block_k` order) on that
  scratch. `block_k` is the only tile that enters an element's contraction
  order and is taken from the model contract exactly as `_canon_matmul_2d`
  does; `block_m` / `block_n` only choose which output tile a program writes.
  Every per-element chain is therefore the one the two-kernel chain
  `_canon_rmsnorm_nd -> _canon_fused_matmul_2d` executes, so the output is
  bitwise identical (`without_use_canon_norm_matmul` must report
  `|LP - Canon| = 0.000e+00` on TPU) and remains batch-invariant (rows are
  independent; `M` is zero-padded to the fixed 128-row block, padded rows
  normalize to exactly `0 * rsqrt(eps) * gamma = 0` and are sliced off).

  Latency: one kernel launch instead of two per site and no `[M, D]` bf16
  round-trip of the normalized rows through HBM (`-2.M.D` bytes of forward
  transient per site).

  HBM Optimization (Fwd+Bwd residual): the un-fused chain stores TWO `[M, D]`
  bf16 residuals per site -- `x` for the RMSNorm VJP and `rmsnorm(x)` for the
  matmul VJP. This coat stores only `(x, gamma, w_parts)` (parameter views, no
  copy) and re-derives `h = canonical_rmsnorm(x, gamma)` -- the declared
  bit-exact semantic of the Pallas norm, i.e. the identical bits the un-fused
  matmul VJP would have read back -- as a backward-local transient, exactly as
  `canonical_vjp.norm_matmul` does. Net `-2.M.D` bytes per site per layer
  (`-4 MB` per site at `M=1024, D=1024`, two sites per layer) in `Remat=NONE`.
  The fused weight is re-materialised behind an `optimization_barrier` tied to
  the cotangent for the same anti-CSE reason documented in
  `_canon_fused_matmul_2d`.

  Args:
    x_2d: `[M, K]` un-normalized rows (residual stream).
    gamma: `[K]` RMSNorm weight.
    w_parts: Weights `[K, N_i]` concatenated along the output axis.
    norm_eps: RMSNorm epsilon.
    contract: Model contract (tile policy) of the Pallas matmul.
    interpret: Pallas interpret mode.
    out_dtype: Output dtype.

  Returns:
    `[M, sum(N_i)]` projection of the normalized rows.
  """
  x_bf16 = x_2d.astype(jnp.bfloat16)
  gamma_bf16 = gamma.astype(jnp.bfloat16)
  parts = tuple(w.astype(jnp.bfloat16) for w in w_parts)
  m, k = map(int, x_bf16.shape)
  if not _norm_matmul_admits(k, contract):
    raise ValueError(
        f'fused norm_matmul needs K={k} to divide block_k={contract.block_k}'
        f' and BF={pallas_rmsnorm.BF}'
    )
  split_points = []
  offset = 0
  for w in parts[:-1]:
    offset += int(w.shape[1])
    split_points.append(offset)
  n_total = offset + int(parts[-1].shape[1])

  # Row block of the fused kernel; the per-element contraction chain only
  # depends on `block_k` (see the docstring), so this choice is numerically
  # neutral and fixed at 128 for every M (padded rows are sliced off).
  block_m = 128
  block_n = contract.block_n
  block_k = contract.block_k
  _, n_padded = padded_matmul.padded_matmul_extents(
      k, n_total, contract=contract, block_k=block_k, block_n=block_n
  )
  m_padded = _round_up(m, block_m)

  def forward(a: jax.Array, g: jax.Array, b: jax.Array) -> jax.Array:
    if m_padded != m:
      a = jnp.pad(a, ((0, m_padded - m), (0, 0)), constant_values=0)
    if n_padded != n_total:
      b = jnp.pad(b, ((0, 0), (0, n_padded - n_total)), constant_values=0)
    out = pallas_norm_matmul.norm_matmul(
        a,
        g,
        b,
        epsilon=norm_eps,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return out[:m, :n_total]

  @jax.custom_vjp
  def op(a: jax.Array, g: jax.Array, ws: Tuple[jax.Array, ...]) -> jax.Array:
    return forward(a, g, jnp.concatenate(ws, axis=1))

  def op_fwd(a: jax.Array, g: jax.Array, ws: Tuple[jax.Array, ...]):
    # Residual = raw rows + gamma + un-concatenated weight views; neither the
    # normalized rows nor the fused weight copy is kept alive.
    return forward(a, g, jnp.concatenate(ws, axis=1)), (a, g, ws)

  def op_bwd(residual, cotangent):
    a, g, ws = residual
    ws_late, cotangent = lax.optimization_barrier((ws, cotangent))
    w_fused = jnp.concatenate(ws_late, axis=1)
    # `canonical_rmsnorm` is the bit-exact JAX statement of the Pallas norm
    # (rows must divide its BM=8 row block; padded rows are sliced off again).
    m8 = _round_up(m, pallas_rmsnorm.BM)
    a8 = jnp.pad(a, ((0, m8 - m), (0, 0)), constant_values=0) if m8 != m else a
    h8, norm_pullback = jax.vjp(
        lambda a_, g_: pallas_rmsnorm.canonical_rmsnorm(
            a_, g_, epsilon=norm_eps
        ),
        a8,
        g,
    )
    h = h8[:m]
    if canonical_vjp.plain_matmul_vjp_enabled():
      dh, dw = canonical_vjp.plain_matmul_pullback(h, w_fused, cotangent)
    else:
      _, pullback = jax.vjp(
          lambda h_, b_: canonical_vjp.canonical_matmul(
              h_, b_, contract=contract
          ),
          h,
          w_fused,
      )
      dh, dw = pullback(cotangent)
    if m8 != m:
      dh = jnp.pad(dh, ((0, m8 - m), (0, 0)), constant_values=0)
    da8, dg = norm_pullback(dh)
    da = da8[:m] if m8 != m else da8
    dws = tuple(jnp.split(dw, split_points, axis=1))
    return da, dg, dws

  op.defvjp(op_fwd, op_bwd)
  return op(x_bf16, gamma_bf16, parts).astype(out_dtype)


def _canon_rmsnorm_nd_local(
    x: jax.Array,
    w: jax.Array,
    *,
    norm_eps: float,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Runs fixed-BF=128 Pallas RMSNorm on a local (per-rank) array."""
  orig_shape = x.shape
  f = int(orig_shape[-1])
  m = int(x.size // f)
  x_2d = x.reshape(m, f).astype(jnp.bfloat16)
  w_1d = w.reshape(f).astype(jnp.bfloat16)

  bm = pallas_rmsnorm.BM
  bf = pallas_rmsnorm.BF
  mp = _round_up(m, bm)

  if f % bf != 0:
    xf = x_2d.astype(jnp.float32)
    wf = w_1d.astype(jnp.float32)
    rms = lax.rsqrt(
        jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(norm_eps)
    )
    return (xf * rms * wf[None, :]).astype(out_dtype).reshape(orig_shape)

  if mp != m:
    x_padded = jnp.pad(x_2d, ((0, mp - m), (0, 0)), constant_values=0)
  else:
    x_padded = x_2d

  def forward(a: jax.Array, weight: jax.Array) -> jax.Array:
    return pallas_rmsnorm.rmsnorm(
        a,
        weight,
        epsilon=norm_eps,
        interpret=interpret,
        shape_invariant_numerics=True,
    )

  out_padded = canonical_vjp.rmsnorm(
      x_padded, w_1d, epsilon=norm_eps, forward=forward
  )
  out_2d = out_padded[:m, :]
  return out_2d.astype(out_dtype).reshape(orig_shape)


def _canon_rmsnorm_nd(
    x: jax.Array,
    w: jax.Array,
    *,
    norm_eps: float,
    interpret: bool,
    out_dtype: jaxtyping.DTypeLike,
) -> jax.Array:
  """Runs fixed-BF=128 Pallas RMSNorm with row padding and TP shard_map support."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')

  def local_nd(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
    return _canon_rmsnorm_nd_local(
        x_loc,
        w_loc,
        norm_eps=norm_eps,
        interpret=interpret,
        out_dtype=out_dtype,
    )

  if mesh is not None and tp_size > 1:
    if x.ndim == 4:
      # q_norm / k_norm: heads are TP-sharded, gamma is replicated. The gamma
      # cotangent is the plain column-parallel case: JAX's transpose of the
      # `P(None)` input `psum`s the per-rank partial `dgamma`
      # (see `_column_parallel_local`).
      return jax.shard_map(
          lambda a, g: _column_parallel_local(local_nd, a, g),
          mesh=mesh,
          in_specs=(P(None, None, 'tp', None), P(None)),
          out_specs=P(None, None, 'tp', None),
          check_vma=False,
      )(x, w)
    if x.ndim == 3:
      # final_norm (and the layer norms when `use_canon_norm_matmul` is off or
      # the site does not admit the fusion): input, gamma AND output are all
      # TP-replicated, so the collective-free VJP coat applies.
      return _shard_map_with_local_vjp(
          local_nd,
          mesh=mesh,
          in_specs=(P(None, None, None), P(None)),
          out_spec=P(None, None, None),
          differentiable=(True, True),
      )(x, w)
  return local_nd(x, w)


def _apply_rope_canonical_local(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Local per-rank body of canonical RoPE (sequence padded to `ROPE_PAD_ROWS`)."""
  l = int(inputs.shape[1])
  lp = _round_up(l, ROPE_PAD_ROWS) if ROPE_PAD_ROWS else l
  if lp != l:
    pad_l = lp - l
    inp_pad = jnp.pad(
        inputs, ((0, 0), (0, pad_l), (0, 0), (0, 0)), constant_values=0
    )
    pos_pad = jnp.pad(positions, ((0, 0), (0, pad_l)), constant_values=0)
  else:
    inp_pad = inputs
    pos_pad = positions

  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = jnp.float32(rope_theta) ** fraction

  sinusoid_inp = (
      pos_pad.astype(jnp.float32)[..., jnp.newaxis]
      / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  # `HARD_ROUNDING_ROPE`: the sin / cos round trip through `inputs.dtype`, the
  # input upcast and the output casts become flag-proof rounding points.
  hard = _hard(HARD_ROUNDING_ROPE)
  sin = _upcast(_cast(jnp.sin(sinusoid_inp), inputs.dtype, hard), hard)
  cos = _upcast(_cast(jnp.cos(sinusoid_inp), inputs.dtype, hard), hard)

  inp_f32 = _upcast(inp_pad, hard)
  first_half, second_half = jnp.split(inp_f32, 2, axis=-1)
  # Separate products via optimization_barrier so XLA cannot fuse
  # `a*cos - b*sin` into `fmsub` in one compilation unit and separate
  # `mul + sub` in another.
  first_cos = lax.optimization_barrier(first_half * cos)
  second_sin = lax.optimization_barrier(second_half * sin)
  second_cos = lax.optimization_barrier(second_half * cos)
  first_sin = lax.optimization_barrier(first_half * sin)

  first_part = _cast(first_cos - second_sin, inputs.dtype, hard)
  second_part = _cast(second_cos + first_sin, inputs.dtype, hard)
  out_pad = jnp.concatenate([first_part, second_part], axis=-1)
  return out_pad[:, :l, :, :]


def apply_rope_canonical(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Applies RoPE with barriered f32 arithmetic and TP shard_map support."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(inputs.shape[2]) % tp_size == 0:
    return jax.shard_map(
        lambda x_loc, p_loc: _apply_rope_canonical_local(
            x_loc, p_loc, head_dim=head_dim, rope_theta=rope_theta
        ),
        mesh=mesh,
        in_specs=(P(None, None, 'tp', None), P(None, None)),
        out_specs=P(None, None, 'tp', None),
        check_vma=False,
    )(inputs, positions)
  return _apply_rope_canonical_local(
      inputs, positions, head_dim=head_dim, rope_theta=rope_theta
  )


def _apply_rope_canonical_qk_local(
    query: jaxtyping.Array,  # [B, L, QH_local, H]
    key: jaxtyping.Array,  # [B, L, KH_local, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Rotates the query and key heads of one rank in a single RoPE body."""
  num_query_heads = int(query.shape[2])
  rotated = _apply_rope_canonical_local(
      jnp.concatenate([query, key], axis=2),
      positions,
      head_dim=head_dim,
      rope_theta=rope_theta,
  )
  return rotated[:, :, :num_query_heads, :], rotated[:, :, num_query_heads:, :]


def apply_rope_canonical_qk(
    query: jaxtyping.Array,  # [B, L, QH, H]
    key: jaxtyping.Array,  # [B, L, KH, H]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Fused `apply_rope_canonical` of `query` and `key` (`ROPE_FUSE_QK`).

  RoPE is element-wise per head, so stacking the query heads in front of the
  key heads and rotating once is bitwise the two separate calls
  (`qwen3_model_canon_test.test_apply_rope_canonical_qk_matches_separate_calls`).
  With `ROPE_FUSE_QK` off, or when the two tensors cannot share one `shard_map`
  (different dtypes, or a head count that is not a multiple of the TP size),
  it falls back to the two separate calls.

  Args:
    query: Queries `[B, L, QH, H]`.
    key: Keys `[B, L, KH, H]`.
    positions: Token positions `[B, L]`.
    head_dim: Head dimension `H`.
    rope_theta: RoPE base frequency.

  Returns:
    The rotated `(query, key)`, same shapes and dtypes as the inputs.
  """
  if not ROPE_FUSE_QK or query.dtype != key.dtype:
    return (
        apply_rope_canonical(
            query, positions, head_dim=head_dim, rope_theta=rope_theta
        ),
        apply_rope_canonical(
            key, positions, head_dim=head_dim, rope_theta=rope_theta
        ),
    )
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if (
      mesh is not None
      and tp_size > 1
      and int(query.shape[2]) % tp_size == 0
      and int(key.shape[2]) % tp_size == 0
  ):
    return jax.shard_map(
        lambda q_loc, k_loc, p_loc: _apply_rope_canonical_qk_local(
            q_loc, k_loc, p_loc, head_dim=head_dim, rope_theta=rope_theta
        ),
        mesh=mesh,
        in_specs=(
            P(None, None, 'tp', None),
            P(None, None, 'tp', None),
            P(None, None),
        ),
        out_specs=(P(None, None, 'tp', None), P(None, None, 'tp', None)),
        check_vma=False,
    )(query, key, positions)
  if mesh is not None and tp_size > 1:
    # Only one of the two tensors is TP-divisible: let each call decide.
    return (
        apply_rope_canonical(
            query, positions, head_dim=head_dim, rope_theta=rope_theta
        ),
        apply_rope_canonical(
            key, positions, head_dim=head_dim, rope_theta=rope_theta
        ),
    )
  return _apply_rope_canonical_qk_local(
      query, key, positions, head_dim=head_dim, rope_theta=rope_theta
  )


def _canonical_online_attention_local(
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, S, KH, D]
    value_proj: jax.Array,  # [B, S, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,  # [B, T, S]
    segment_ids: jax.Array | None = None,  # [B, T]
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    interpret: bool = False,
) -> jax.Array:
  """Local per-rank body of the fixed-tile online softmax attention.

  Builds the GQA-grouped operands and the merged boolean mask, then runs
  `canon_attention.attention_core`: the forward selected by
  `canon_attention.ATTENTION_FWD` (A4, the Pallas fixed-tile exp2
  online-softmax kernel = the Zero-TIM forward) with the backward selected by
  `canon_attention.ATTENTION_BWD` (B4). `interpret` runs both Pallas kernels
  in interpret mode (CPU).

  Args:
    query_proj: Queries `[B, T, QH, D]`.
    key_proj: Keys `[B, S, KH, D]`.
    value_proj: Values `[B, S, KH, D]`.
    scale: Softmax temperature (`1 / sqrt(D)`).
    attn_mask: Optional attention mask `[B, T, S]`; `None` attends everywhere.
    segment_ids: Optional segment ids `[B, T]`, merged into the mask for
      self-attention (`T == S`).
    kv_block_size: Fixed key block (`bkv_csz`).
    interpret: Pallas interpret mode of the forward and backward kernels (CPU
      tests).

  Returns:
    Attention output `[B, T, QH, D]` in the query dtype.
  """
  b, t, qh, d = query_proj.shape
  _, s, kh, _ = key_proj.shape
  g = qh // kh

  q_grouped = query_proj.reshape(b, t, kh, g, d).transpose(0, 2, 3, 1, 4)
  k_grouped = key_proj.transpose(0, 2, 1, 3)  # [B, KH, S, D]
  v_grouped = value_proj.transpose(0, 2, 1, 3)  # [B, KH, S, D]

  if attn_mask is not None:
    mask_bts = attn_mask.astype(jnp.bool_)
  else:
    mask_bts = jnp.ones((b, t, s), dtype=jnp.bool_)

  if segment_ids is not None:
    if t != s:
      raise ValueError(
          'segment_ids are only supported for self-attention (T == S), got'
          f' T={t}, S={s}'
      )
    seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    mask_bts = jnp.logical_and(mask_bts, seg_mask)

  out = canon_attention.attention_core(
      q_grouped,
      k_grouped,
      v_grouped,
      mask_bts,
      scale=scale,
      kv_block_size=kv_block_size,
      interpret=interpret,
      hard_rounding=_hard(HARD_ROUNDING_ATTENTION),
  )  # [B, KH, G, T, D]
  return out.transpose(0, 3, 1, 2, 4).reshape(b, t, qh, d)


def canonical_online_attention(
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, S, KH, D]
    value_proj: jax.Array,  # [B, S, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,  # [B, T, S]
    segment_ids: jax.Array | None = None,  # [B, T]
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    interpret: bool = False,
) -> jax.Array:
  """Fixed-tile online softmax attention (`bkv_csz=512`; Pallas forward, A4)."""
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(key_proj.shape[2]) % tp_size == 0:
    mask_spec = P(None, None, None) if attn_mask is not None else None
    seg_spec = P(None, None) if segment_ids is not None else None

    def local_attn(q_loc, k_loc, v_loc, m_loc, s_loc):
      return _canonical_online_attention_local(
          q_loc,
          k_loc,
          v_loc,
          scale=scale,
          attn_mask=m_loc,
          segment_ids=s_loc,
          kv_block_size=kv_block_size,
          interpret=interpret,
      )

    return jax.shard_map(
        local_attn,
        mesh=mesh,
        in_specs=(
            P(None, None, 'tp', None),
            P(None, None, 'tp', None),
            P(None, None, 'tp', None),
            mask_spec,
            seg_spec,
        ),
        out_specs=P(None, None, 'tp', None),
        check_vma=False,
    )(query_proj, key_proj, value_proj, attn_mask, segment_ids)
  return _canonical_online_attention_local(
      query_proj,
      key_proj,
      value_proj,
      scale=scale,
      attn_mask=attn_mask,
      segment_ids=segment_ids,
      kv_block_size=kv_block_size,
      interpret=interpret,
  )


def rpa_attention_selected() -> bool:
  """Whether `canon_attention.ATTENTION_FWD` routes attention to the RPA kernel."""
  return canon_attention.ATTENTION_FWD == canon_attention.RPA_FWD


def _check_rpa_contract(
    attn_mask: jax.Array | None, segment_ids: jax.Array | None
) -> None:
  """The RPA backend only runs the bench's prefix-causal programs."""
  if attn_mask is None:
    raise ValueError(
        f'{canon_attention.RPA_FWD!r} implements prefix-causal attention;'
        ' attn_mask=None (unmasked attention) is not supported'
    )
  if segment_ids is not None:
    raise ValueError(
        f'{canon_attention.RPA_FWD!r} does not support segment_ids (packed'
        ' sequences)'
    )


def rpa_attention_cacheless(
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, T, KH, D]
    value_proj: jax.Array,  # [B, T, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Cacheless causal attention on the engine's RPA kernel (`RPA_FWD`).

  `attn_mask` must be the causal mask implied by the positions (the kernel
  masks causally on absolute positions and the mask is not re-validated).

  Args:
    query_proj: Queries `[B, T, QH, D]`.
    key_proj: Keys `[B, T, KH, D]`.
    value_proj: Values `[B, T, KH, D]`.
    scale: Softmax scale.
    attn_mask: The bench's `[B, T, T]` causal mask (required).
    segment_ids: Unsupported (raises).

  Returns:
    `out[B, T, QH, D]`; see `rpa_backend.attention_cacheless`.
  """
  _check_rpa_contract(attn_mask, segment_ids)
  _, mesh = _active_tp_size_and_mesh('tp')
  return rpa_backend.attention_cacheless(
      query_proj, key_proj, value_proj, scale=scale, mesh=mesh, tp_axis='tp'
  )


def rpa_attention_cached(
    cache: LayerCache,
    query_proj: jax.Array,  # [B, T, QH, D]
    key_proj: jax.Array,  # [B, T, KH, D]
    value_proj: jax.Array,  # [B, T, KH, D]
    *,
    scale: float,
    attn_mask: jax.Array | None,
    segment_ids: jax.Array | None = None,
) -> tuple[LayerCache, jax.Array]:
  """KV-cached attention on the engine's RPA kernel (`RPA_FWD`).

  The new rows are written at `[end_index, end_index + T)` of every
  sequence's page table (`attn_mask` must be the prefix-causal mask of those
  positions); the dense `init_cache` layout is converted to the paged layout
  on the first call, see `rpa_backend.attention_cached`.

  Args:
    cache: The layer cache (`'k'`, `'v'`, `'end_index'`).
    query_proj: Queries `[B, T, QH, D]`.
    key_proj: New keys `[B, T, KH, D]`.
    value_proj: New values `[B, T, KH, D]`.
    scale: Softmax scale.
    attn_mask: The bench's `[B, T, S]` mask (required).
    segment_ids: Unsupported (raises).

  Returns:
    `(new_cache, out[B, T, QH, D])`.
  """
  _check_rpa_contract(attn_mask, segment_ids)
  _, mesh = _active_tp_size_and_mesh('tp')
  return rpa_backend.attention_cached(
      cache,
      query_proj,
      key_proj,
      value_proj,
      scale=scale,
      mesh=mesh,
      tp_axis='tp',
  )


def _fixed_m_lm_head_2d_local(
    x_bf16: jax.Array,  # [M, D]
    w_bf16: jax.Array,  # [D, V_local]
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
) -> jax.Array:
  """Local per-rank P38 `FIXED_M=256` padded/chunked LM head."""
  m, hidden = map(int, x_bf16.shape)
  _, vocab = map(int, w_bf16.shape)
  local_matmul = fixed_lm_head.canonical_local_matmul(
      contract, interpret=interpret
  )
  fixed_m = fixed_lm_head.FIXED_M
  bm = fixed_lm_head.BM
  bn = fixed_lm_head.BN if vocab >= fixed_lm_head.BN else contract.block_n
  bk = fixed_lm_head.BK if hidden >= 1024 else contract.block_k

  def run_fixed_chunk(a_chunk: jax.Array, weight: jax.Array) -> jax.Array:
    return local_matmul(
        a_chunk,
        weight,
        block_m=bm,
        block_n=bn,
        block_k=bk,
        shape_invariant_numerics=True,
    )

  if m < fixed_m:
    a_padded = jnp.pad(x_bf16, ((0, fixed_m - m), (0, 0)), constant_values=0)
    return run_fixed_chunk(a_padded, w_bf16)[:m, :]
  if m == fixed_m:
    return run_fixed_chunk(x_bf16, w_bf16)

  mp = _round_up(m, fixed_m)
  if mp != m:
    a_all = jnp.pad(x_bf16, ((0, mp - m), (0, 0)), constant_values=0)
  else:
    a_all = x_bf16

  def chunked_forward(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    return lax.map(
        lambda chunk: run_fixed_chunk(chunk, weight), a_chunks
    ).reshape((mp, vocab))

  @jax.custom_vjp
  def chunked_vjp(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    return chunked_forward(a_learner, weight)

  def chunked_fwd(a_learner: jax.Array, weight: jax.Array):
    return chunked_forward(a_learner, weight), (a_learner, weight)

  def chunked_bwd(residual, cotangent):
    a_learner, weight = residual
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    cot_chunks = cotangent.reshape((-1, fixed_m, vocab))

    def accumulate(w_cot, values):
      a_chunk, out_cot = values
      # Latency Optimization: when plain_matmul_vjp is active (default), call
      # plain_matmul_pullback directly instead of jax.vjp(run_fixed_chunk, ...),
      # avoiding re-launching the forward Pallas LM-head kernel on every chunk
      # during backward.
      if canonical_vjp.plain_matmul_vjp_enabled():
        a_cot, chunk_w_cot = canonical_vjp.plain_matmul_pullback(
            a_chunk, weight, out_cot
        )
      else:
        _, pullback = jax.vjp(run_fixed_chunk, a_chunk, weight)
        a_cot, chunk_w_cot = pullback(out_cot)
      return w_cot + chunk_w_cot, a_cot

    w_cot, a_cot_chunks = lax.scan(
        accumulate,
        jnp.zeros_like(weight),
        (a_chunks, cot_chunks),
    )
    return a_cot_chunks.reshape(a_learner.shape), w_cot

  chunked_vjp.defvjp(chunked_fwd, chunked_bwd)
  return chunked_vjp(a_all, w_bf16)[:m, :]


def _fixed_m_lm_head_2d(
    x_2d: jax.Array,  # [M, D]
    w_dv: jax.Array,  # [D, V]
    *,
    contract: model_contracts.ModelContract,
    interpret: bool,
    endpoint: str,
) -> jax.Array:
  """Runs the P38 `FIXED_M=256` padded/chunked LM head on `[M, D] @ [D, V]`."""
  del endpoint
  x_bf16 = x_2d.astype(jnp.bfloat16)
  w_bf16 = w_dv.astype(jnp.bfloat16)
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  if mesh is not None and tp_size > 1 and int(w_bf16.shape[1]) % tp_size == 0:

    def local_head(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
      # Vocab-parallel head: `x` replicated, `W` column-sharded. `dx` is the
      # column-parallel psum supplied by JAX's transpose (see
      # `_column_parallel_local`).
      return _column_parallel_local(
          lambda a, weight: _fixed_m_lm_head_2d_local(
              a, weight, contract=contract, interpret=interpret
          ),
          x_loc,
          w_loc,
      )

    return jax.shard_map(
        local_head,
        mesh=mesh,
        in_specs=(P(None, None), P(None, 'tp')),
        out_specs=P(None, 'tp'),
        check_vma=False,
    )(x_bf16, w_bf16)

  return _fixed_m_lm_head_2d_local(
      x_bf16, w_bf16, contract=contract, interpret=interpret
  )


class CanonEinsum(nnx.Module):
  """Einsum module supporting both stock `jnp.einsum` and Zero-TIM matmul."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.w = nnx.Param(
        nnx.initializers.glorot_uniform()(
            rngs.params(), shape, dtype=param_dtype
        ),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_canon: bool = False,
      use_fixed_order_reduce: bool = False,
      contract: model_contracts.ModelContract | None = None,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    x_arr = jnp.astype(x, self.dtype)
    w_arr = jnp.astype(self.w.value, self.dtype)

    if self.einsum_str == 'BTNH,NHD->BTD':
      b, t, n_heads, head_dim = x_arr.shape
      _, _, d = w_arr.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')

      def local_mm(a2: jax.Array, w2: jax.Array) -> jax.Array:
        if use_canon and contract is not None:
          return _canon_matmul_2d(
              a2,
              w2,
              contract=contract,
              interpret=interpret,
              out_dtype=jnp.bfloat16,
          )
        return jnp.dot(a2.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)).astype(
            jnp.bfloat16
        )

      if not use_canon and not use_fixed_order_reduce:
        return jnp.einsum(self.einsum_str, x_arr, w_arr)

      if mesh is not None and tp_size > 1:

        def local_o_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          a2 = x_loc.reshape(b * t, -1).astype(jnp.bfloat16)
          w2 = w_loc.reshape(a2.shape[1], d).astype(jnp.bfloat16)
          partial = local_mm(a2, w2)
          if use_fixed_order_reduce:
            # Contract-parallel site: the bf16 per-rank partials are summed in
            # rank order (bitwise identical to every other reducer mode); the
            # VJP broadcasts the full cotangent to every rank.
            acc = _fixed_order_tp_sum_with_vjp(
                partial,
                'tp',
                tp_size,
                mode=_forward_tp_sum_mode(b * t),
            )
          else:
            acc = lax.psum(partial, 'tp')
          return acc.astype(self.dtype).reshape(b, t, d)

        return jax.shard_map(
            local_o_proj,
            mesh=mesh,
            in_specs=(P(None, None, 'tp', None), P('tp', None, None)),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x_arr, w_arr)

      if use_fixed_order_reduce and n_heads >= 2 and n_heads % 2 == 0:
        tp_shards = 2
        h_per_tp = n_heads // tp_shards
        x_tp = x_arr.reshape(b * t, tp_shards, h_per_tp * head_dim)
        w_tp = w_arr.reshape(tp_shards, h_per_tp * head_dim, d)
        acc = jnp.zeros((b * t, d), dtype=jnp.float32)
        for r in range(tp_shards):
          part = local_mm(x_tp[:, r, :], w_tp[r, :, :])
          acc = acc + part.astype(jnp.float32)
        return acc.astype(self.dtype).reshape(b, t, d)

      if not use_canon or contract is None:
        return jnp.einsum(self.einsum_str, x_arr, w_arr)

      x_2d = x_arr.reshape(b * t, n_heads * head_dim)
      w_2d = w_arr.reshape(n_heads * head_dim, d)
      out_2d = _canon_matmul_2d(
          x_2d,
          w_2d,
          contract=contract,
          interpret=interpret,
          out_dtype=self.dtype,
      )
      return out_2d.reshape(b, t, d)

    if not use_canon or contract is None:
      return jnp.einsum(self.einsum_str, x_arr, w_arr)

    if self.einsum_str in ('BTD,DNH->BTNH', 'BSD,DKH->BSKH'):
      b, t, d = x_arr.shape
      _, n_heads, head_dim = w_arr.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')
      if mesh is not None and tp_size > 1 and n_heads % tp_size == 0:

        def local_qkv(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          n_h_loc = int(w_loc.shape[1])

          def _mm(a_3d: jax.Array, weight_3d: jax.Array) -> jax.Array:
            out_2d = _canon_matmul_2d(
                a_3d.reshape(b * t, d),
                weight_3d.reshape(d, n_h_loc * head_dim),
                contract=contract,
                interpret=interpret,
                out_dtype=self.dtype,
            )
            return out_2d.reshape(b, t, n_h_loc, head_dim)

          return _column_parallel_local(_mm, x_loc, w_loc)

        return jax.shard_map(
            local_qkv,
            mesh=mesh,
            in_specs=(P(None, None, None), P(None, 'tp', None)),
            out_specs=P(None, None, 'tp', None),
            check_vma=False,
        )(x_arr, w_arr)

      x_2d = x_arr.reshape(b * t, d)
      w_2d = w_arr.reshape(d, n_heads * head_dim)
      out_2d = _canon_matmul_2d(
          x_2d,
          w_2d,
          contract=contract,
          interpret=interpret,
          out_dtype=self.dtype,
      )
      return out_2d.reshape(b, t, n_heads, head_dim)

    if self.einsum_str == 'BTD,DV->BTV':
      b, t, d = x_arr.shape
      _, vocab = w_arr.shape
      x_2d = x_arr.reshape(b * t, d)
      out_2d = _fixed_m_lm_head_2d(
          x_2d,
          w_arr,
          contract=contract,
          interpret=interpret,
          endpoint='untied_lm_head',
      )
      return out_2d.astype(self.dtype).reshape(b, t, vocab)

    return jnp.einsum(self.einsum_str, x_arr, w_arr)


class CanonEmbedder(nnx.Module):
  """Embedder module supporting stock decode and P38 fixed_lm_head decode."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype

  @jax.named_scope('embedder_encode')
  def encode(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_fixed_order_reduce: bool = False,
  ) -> jaxtyping.Array:
    x_ids = jnp.asarray(x, dtype=jnp.int32)
    tp_size, mesh = _active_tp_size_and_mesh('tp')
    if (
        use_fixed_order_reduce
        and mesh is not None
        and tp_size > 1
        and int(self.input_embedding.value.shape[0]) % tp_size == 0
    ):
      orig_shape = x_ids.shape
      flat_ids = x_ids.reshape(-1)
      table = jnp.astype(self.input_embedding.value, self.dtype)

      def local_embed(
          ids_local: jax.Array, table_local: jax.Array
      ) -> jax.Array:
        j = lax.axis_index('tp')
        vloc = table_local.shape[0]
        li = ids_local - j * vloc
        ok = (li >= 0) & (li < vloc)
        part = jnp.where(
            ok[:, None],
            jnp.take(table_local, jnp.clip(li, 0, vloc - 1), axis=0),
            jnp.zeros((), table_local.dtype),
        )
        # Vocab-parallel gather: exactly one rank holds each id, the others
        # contribute zeros, so the rank-ordered sum is the row itself (bitwise)
        # and the VJP must hand the full cotangent to the owning rank.
        return _fixed_order_tp_sum_with_vjp(
            part, 'tp', tp_size, mode=fixed_order_reduce.RING
        )

      out_2d = jax.shard_map(
          local_embed,
          mesh=mesh,
          in_specs=(P(None), P('tp', None)),
          out_specs=P(None, None),
          check_vma=False,
      )(flat_ids, table)
      return out_2d.astype(self.dtype).reshape(
          *orig_shape, self.input_embedding.value.shape[1]
      )
    out = self.input_embedding[(x_ids,)]
    out = jnp.astype(out, self.dtype)
    out = shard(out, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return out

  @jax.named_scope('embedder_decode')
  def decode(
      self,
      x: jaxtyping.ArrayLike,
      *,
      use_fixed_lm_head: bool = False,
      contract: model_contracts.ModelContract | None = None,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    x_arr = jnp.astype(x, self.dtype)
    w_arr = jnp.astype(self.input_embedding.value, self.dtype)
    if not use_fixed_lm_head or contract is None:
      return jnp.dot(x_arr, w_arr.T)

    b, t, d = x_arr.shape
    vocab = w_arr.shape[0]
    x_2d = x_arr.reshape(b * t, d)
    out_2d = _fixed_m_lm_head_2d(
        x_2d,
        w_arr.T,
        contract=contract,
        interpret=interpret,
        endpoint='tied_embed',
    )
    return out_2d.astype(self.dtype).reshape(b, t, vocab)


class CanonRMSNorm(nnx.Module):
  """RMSNorm layer supporting both stock JAX and P22.XH Pallas RMSNorm."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(
            rngs.params(), dim, param_dtype  # pyrefly: ignore[bad-argument-type]
        ),  # pyrefly: ignore[bad-argument-type]
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(
      self,
      x: jaxtyping.Array,
      *,
      use_canon: bool = False,
      interpret: bool = False,
  ) -> jaxtyping.Array:
    if not use_canon:
      x_f32 = jnp.astype(x, jnp.float32)
      rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps)
      return jnp.astype(
          jnp.astype(self.w.value, jnp.float32) * (x_f32 / rms), self.dtype
      )
    return _canon_rmsnorm_nd(
        x,
        self.w.value,
        norm_eps=self.norm_eps,
        interpret=interpret,
        out_dtype=self.dtype,
    )


class CanonAttention(nnx.Module):
  """Attention module with Zero-TIM kernel switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.canon_config = canon_config
    self.contract = contract
    self.q_proj = CanonEinsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = CanonEinsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = CanonEinsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = CanonEinsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_norm = CanonRMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_norm = CanonRMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def _qkv_proj(
      self,
      x: jaxtyping.Array,
      *,
      use_canon: bool,
      interpret: bool,
      norm_weight: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes `(q, k, v)` projections, fusing into 1 Pallas call when canonical.

    Latency & HBM Optimization (Horizontal QKV Fusion):
      - Forward Latency: `q_proj`, `k_proj`, and `v_proj` share the exact same
        input `x` `[B, T, D]` and contracting dimension `D`. Concatenating
        `[W_q, W_k, W_v]` along the output column axis on each TP shard and
        invoking one fixed-tile matmul eliminates 2 Pallas kernel launches
        and 2 redundant M-dimension paddings per layer in every Decode step
        and Prefill pass, while keeping output tiles 100% bitwise identical.
      - HBM Residuals: the fused coat saves the input activation `x_2d` only
        ONCE per layer instead of 3 times (-66% QKV activation memory), and
        `_canon_fused_matmul_2d` keeps the un-fused `W_q/W_k/W_v` parameter
        views (not the concatenated copy) as its backward residual, so the
        fusion adds no persistent weight memory in `Remat=NONE` training.
      - Backward: `dx` is one fused Pallas matmul (`dy_qkv @ W_qkv.T`) whose
        TP partials JAX reduces with the single column-parallel `psum` of the
        replicated `x` (see `_column_parallel_local`).

    Latency & HBM Optimization (RMSNorm prologue fusion,
    `use_canon_norm_matmul`):
      With `norm_weight` (the `input_layernorm` gamma) given, `x` is the
      UN-normalized residual stream and the norm is applied inside the
      projection kernel by `_canon_fused_norm_matmul_2d` (P56.4.6): the
      normalized rows never leave VMEM, saving one kernel launch and one
      `[M, D]` bf16 HBM round trip per layer in the forward, and the matmul's
      `[M, D]` normalized-row residual in `Remat=NONE` training (the norm is
      re-derived bit-exactly in the backward). Bitwise identical to the
      un-fused `input_layernorm -> _qkv_proj` chain (see the helper).

    Args:
      x: `[B, T, D]` hidden states (un-normalized iff `norm_weight` is given).
      use_canon: Use the fixed-tile Pallas matmul.
      interpret: Pallas interpret mode.
      norm_weight: Optional `[D]` RMSNorm gamma to fuse as a prologue (requires
        `use_canon`; the caller checks `_norm_matmul_admits`).

    Returns:
      `(q, k, v)` as `[B, T, N_q, H]`, `[B, T, N_kv, H]`, `[B, T, N_kv, H]`.
    """
    if not use_canon:
      if norm_weight is not None:
        raise ValueError('norm fusion requires the canonical QKV projection')
      return (
          self.q_proj(
              x, use_canon=False, contract=self.contract, interpret=interpret
          ),
          self.k_proj(
              x, use_canon=False, contract=self.contract, interpret=interpret
          ),
          self.v_proj(
              x, use_canon=False, contract=self.contract, interpret=interpret
          ),
      )

    x_arr = jnp.astype(x, self.config.dtype)
    wq = jnp.astype(self.q_proj.w.value, self.config.dtype)
    wk = jnp.astype(self.k_proj.w.value, self.config.dtype)
    wv = jnp.astype(self.v_proj.w.value, self.config.dtype)
    gamma = (
        None
        if norm_weight is None
        else jnp.astype(norm_weight, self.config.dtype)
    )
    b, t, d = x_arr.shape
    _, q_heads, head_dim = wq.shape
    _, kv_heads, _ = wk.shape
    tp_size, mesh = _active_tp_size_and_mesh('tp')

    def fused_2d(
        a_2d: jax.Array, g: jax.Array | None, ws: Tuple[jax.Array, ...]
    ) -> jax.Array:
      if g is None:
        return _canon_fused_matmul_2d(
            a_2d,
            ws,
            contract=self.contract,
            interpret=interpret,
            out_dtype=self.config.dtype,
        )
      # `norm_eps` is a static Python float shared by every norm of the model
      # (never a traced operand, so the fusion is safe under `jax.checkpoint`).
      return _canon_fused_norm_matmul_2d(
          a_2d,
          g,
          ws,
          norm_eps=self.config.norm_eps,
          contract=self.contract,
          interpret=interpret,
          out_dtype=self.config.dtype,
      )

    if (
        mesh is not None
        and tp_size > 1
        and q_heads % tp_size == 0
        and kv_heads % tp_size == 0
    ):

      def local_qkv_fused(
          x_loc: jax.Array,
          wq_loc: jax.Array,
          wk_loc: jax.Array,
          wv_loc: jax.Array,
          g_loc: jax.Array | None,
      ) -> tuple[jax.Array, jax.Array, jax.Array]:
        qh_loc = int(wq_loc.shape[1])
        kh_loc = int(wk_loc.shape[1])
        q_dim = qh_loc * head_dim
        kv_dim = kh_loc * head_dim

        def _mm_qkv(
            a_3d: jax.Array,
            wq_3d: jax.Array,
            wk_3d: jax.Array,
            wv_3d: jax.Array,
            g_1d: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
          # The fused coats concatenate `[W_q | W_k | W_v]` inside the forward
          # only; their VJP residuals keep the un-fused parameter views so the
          # fusion adds no persistent weight copy (see the helpers).
          out_qkv_2d = fused_2d(
              a_3d.reshape(b * t, d),
              g_1d,
              (
                  wq_3d.reshape(d, q_dim),
                  wk_3d.reshape(d, kv_dim),
                  wv_3d.reshape(d, kv_dim),
              ),
          )
          q_2d, k_2d, v_2d = jnp.split(
              out_qkv_2d, [q_dim, q_dim + kv_dim], axis=1
          )
          return (
              q_2d.reshape(b, t, qh_loc, head_dim),
              k_2d.reshape(b, t, kh_loc, head_dim),
              v_2d.reshape(b, t, kh_loc, head_dim),
          )

        return _column_parallel_local(
            _mm_qkv, x_loc, wq_loc, wk_loc, wv_loc, g_loc
        )

      return jax.shard_map(
          local_qkv_fused,
          mesh=mesh,
          in_specs=(
              P(None, None, None),
              P(None, 'tp', None),
              P(None, 'tp', None),
              P(None, 'tp', None),
              P(None) if gamma is not None else None,
          ),
          out_specs=(
              P(None, None, 'tp', None),
              P(None, None, 'tp', None),
              P(None, None, 'tp', None),
          ),
          check_vma=False,
      )(x_arr, wq, wk, wv, gamma)

    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    out_qkv_2d = fused_2d(
        x_arr.reshape(b * t, d),
        gamma,
        (
            wq.reshape(d, q_dim),
            wk.reshape(d, kv_dim),
            wv.reshape(d, kv_dim),
        ),
    )
    q_2d, k_2d, v_2d = jnp.split(out_qkv_2d, [q_dim, q_dim + kv_dim], axis=1)
    return (
        q_2d.reshape(b, t, q_heads, head_dim),
        k_2d.reshape(b, t, kv_heads, head_dim),
        v_2d.reshape(b, t, kv_heads, head_dim),
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
      norm_weight: jax.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]
    cc = self.canon_config
    interpret = cc.resolve_interpret()

    q_raw, k_raw, value_proj = self._qkv_proj(
        x,
        use_canon=cc.use_canon_qkv_proj,
        interpret=interpret,
        norm_weight=norm_weight,
    )

    def _run_qk_norm(
        inp: jax.Array,
        w_val: jax.Array,
        *,
        norm_eps: float,
        dtype: jnp.dtype,
    ) -> jax.Array:
      if not cc.use_canon_rmsnorm:
        inp_f32 = jnp.astype(inp, jnp.float32)
        rms = jnp.sqrt(jnp.mean(inp_f32**2, axis=-1, keepdims=True) + norm_eps)
        return jnp.astype(
            jnp.astype(w_val, jnp.float32) * (inp_f32 / rms), dtype
        )
      return _canon_rmsnorm_nd(
          inp,
          w_val,
          norm_eps=norm_eps,
          interpret=interpret,
          out_dtype=dtype,
      )

    def _norm_rope_and_attend_cacheless(
        q_in: jax.Array,
        k_in: jax.Array,
        v_in: jax.Array,
        q_w: jax.Array,
        k_w: jax.Array,
        seg_pos: jax.Array,
        mask: jax.Array | None,
        seg_ids: jax.Array | None,
    ) -> jax.Array:
      query_proj = _run_qk_norm(
          q_in,
          q_w,
          norm_eps=self.q_norm.norm_eps,
          dtype=self.q_norm.dtype,
      )
      key_proj = _run_qk_norm(
          k_in,
          k_w,
          norm_eps=self.k_norm.norm_eps,
          dtype=self.k_norm.dtype,
      )
      query_proj = shard(
          query_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )
      key_proj = shard(
          key_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )
      v_sharded = shard(
          v_in, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )
      if cc.use_canon_rope:
        query_proj, key_proj = apply_rope_canonical_qk(
            query_proj,
            key_proj,
            seg_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )
      else:
        query_proj = qwen3_stock.apply_rope(
            query_proj,
            seg_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )
        key_proj = qwen3_stock.apply_rope(
            key_proj,
            seg_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )
      b, t, qh, d = query_proj.shape
      _, _, kh, _ = key_proj.shape
      if cc.use_canon_attention:
        if rpa_attention_selected():
          return rpa_attention_cacheless(
              query_proj,
              key_proj,
              v_sharded,
              scale=self.scale,
              attn_mask=mask,
              segment_ids=seg_ids,
          )
        return canonical_online_attention(
            query_proj,
            key_proj,
            v_sharded,
            scale=self.scale,
            attn_mask=mask,
            segment_ids=seg_ids,
            kv_block_size=cc.attention_kv_block_size,
            interpret=interpret,
        )
      query_grouped = query_proj.reshape((b, t, kh, qh // kh, d))
      attn = (
          jnp.einsum('BTHGD,BSHD->BHGTS', query_grouped, key_proj) * self.scale
      )
      if mask is not None:
        attn = jnp.where(mask[:, None, None, :, :], attn, K_MASK)
      if seg_ids is not None:
        seg_mask = seg_ids[:, :, None] == seg_ids[:, None, :]
        attn = jnp.where(seg_mask[:, None, None, :, :], attn, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )
      qkv_out = jnp.einsum('BHGTS,BSHD->BTHGD', attn, v_sharded)
      return qkv_out.reshape((b, t, qh, d))

    is_block_remat = _remat_is(self.config, RematConfig.BLOCK)
    use_any_canon_attn = (
        cc.use_canon_rmsnorm or cc.use_canon_rope or cc.use_canon_attention
    )
    if (
        cache is None
        and CHECKPOINT_ATTN_CORE
        and use_any_canon_attn
        and not is_block_remat
    ):
      qkv = jax.checkpoint(_norm_rope_and_attend_cacheless)(
          q_raw,
          k_raw,
          value_proj,
          self.q_norm.w.value,
          self.k_norm.w.value,
          segment_pos,
          attn_mask,
          segment_ids,
      )
      new_cache = None
    elif cache is None:
      qkv = _norm_rope_and_attend_cacheless(
          q_raw,
          k_raw,
          value_proj,
          self.q_norm.w.value,
          self.k_norm.w.value,
          segment_pos,
          attn_mask,
          segment_ids,
      )
      new_cache = None
    else:
      query_proj = self.q_norm(
          q_raw, use_canon=cc.use_canon_rmsnorm, interpret=interpret
      )
      key_proj = self.k_norm(
          k_raw, use_canon=cc.use_canon_rmsnorm, interpret=interpret
      )

      query_proj = shard(
          query_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )
      key_proj = shard(
          key_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )
      value_proj = shard(
          value_proj, self.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
      )

      if cc.use_canon_rope:
        query_proj, key_proj = apply_rope_canonical_qk(
            query_proj,
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )
      else:
        query_proj = qwen3_stock.apply_rope(
            query_proj,
            segment_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )
        key_proj = qwen3_stock.apply_rope(
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
            rope_theta=self.config.rope_theta,
        )

      if cc.use_canon_attention and rpa_attention_selected():
        # The engine kernel writes the new rows into its paged cache itself
        # (the dense `dynamic_update_slice` cache below is not materialized).
        new_cache, qkv = rpa_attention_cached(
            cache,
            query_proj,
            key_proj,
            value_proj,
            scale=self.scale,
            attn_mask=attn_mask,
            segment_ids=segment_ids,
        )
      else:
        end_index = cache['end_index'][0]
        slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache['v'],
            value_proj,
            slice_indices,
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache['k'], key_proj, slice_indices
        )
        cache_value_proj = value_proj
        cache_key_proj = key_proj

        b, t, qh, d = query_proj.shape
        _, _, kh, _ = key_proj.shape

        if cc.use_canon_attention:
          qkv = canonical_online_attention(
              query_proj,
              key_proj,
              value_proj,
              scale=self.scale,
              attn_mask=attn_mask,
              segment_ids=segment_ids,
              kv_block_size=cc.attention_kv_block_size,
              interpret=interpret,
          )
        else:
          query_grouped = query_proj.reshape((b, t, kh, qh // kh, d))
          attn = (
              jnp.einsum('BTHGD,BSHD->BHGTS', query_grouped, key_proj)
              * self.scale
          )
          if attn_mask is not None:
            attn = jnp.where(attn_mask[:, None, None, :, :], attn, K_MASK)
          if segment_ids is not None:
            seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
            attn = jnp.where(seg_mask[:, None, None, :, :], attn, K_MASK)
          attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
              key_proj.dtype
          )
          qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
          qkv = qkv.reshape((b, t, qh, d))

        new_cache = {
            'v': cache_value_proj,
            'k': cache_key_proj,
            'end_index': cache['end_index'] + seq_len,
        }

    outputs = self.o_proj(
        qkv,
        use_canon=cc.use_canon_o_proj,
        use_fixed_order_reduce=cc.use_fixed_order_reduce,
        contract=self.contract,
        interpret=interpret,
    )
    outputs = shard(
        outputs, self.shd_config.act_btd  # pyrefly: ignore[bad-argument-type]
    )

    return new_cache, outputs

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
      norm_weight: jax.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if _remat_is(self.config, RematConfig.BLOCK):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state,
          x,
          segment_pos,
          cache,
          attn_mask,
          segment_ids=segment_ids,
          norm_weight=norm_weight,
      )
    return self.block(
        x,
        segment_pos,
        cache,
        attn_mask,
        segment_ids=segment_ids,
        norm_weight=norm_weight,
    )

  @property
  def head_dim(self):
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    return self.k_proj.shape[1]


class CanonMLP(nnx.Module):
  """MLP module with Zero-TIM `padded_matmul` and `padded_swiglu` switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.canon_config = canon_config
    self.contract = contract
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_fd,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def _linear_proj(
      self,
      x: jax.Array,
      linear_mod: nnx.Linear,
      *,
      use_canon: bool,
      use_fixed_order_reduce: bool = False,
      contract_parallel: bool = False,
      interpret: bool,
  ) -> jax.Array:
    b, t, d_in = x.shape
    w = jnp.astype(linear_mod.kernel.value, self.config.dtype)
    d_out = w.shape[1]
    tp_size, mesh = _active_tp_size_and_mesh('tp')

    def local_mm(a2: jax.Array, w2: jax.Array) -> jax.Array:
      if use_canon:
        return _canon_matmul_2d(
            a2,
            w2,
            contract=self.contract,
            interpret=interpret,
            out_dtype=jnp.bfloat16,
        )
      return jnp.dot(a2.astype(jnp.bfloat16), w2.astype(jnp.bfloat16)).astype(
          jnp.bfloat16
      )

    if contract_parallel:
      if not use_canon and not use_fixed_order_reduce:
        return linear_mod(x)
      if mesh is not None and tp_size > 1:

        def local_down_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
          a2 = x_loc.reshape(b * t, -1).astype(jnp.bfloat16)
          w2 = w_loc.astype(jnp.bfloat16)
          partial = local_mm(a2, w2)
          if use_fixed_order_reduce:
            # Contract-parallel site (see `o_proj`): rank-ordered sum of the
            # bf16 partials, full-cotangent broadcast in the VJP.
            acc = _fixed_order_tp_sum_with_vjp(
                partial,
                'tp',
                tp_size,
                mode=_forward_tp_sum_mode(b * t),
            )
          else:
            acc = lax.psum(partial, 'tp')
          return acc.astype(self.config.dtype).reshape(b, t, d_out)

        return jax.shard_map(
            local_down_proj,
            mesh=mesh,
            in_specs=(P(None, None, 'tp'), P('tp', None)),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x, w)
      if use_fixed_order_reduce and d_in >= 256 and d_in % 2 == 0:
        tp_shards = 2
        k_per_tp = d_in // tp_shards
        x_tp = x.reshape(b * t, tp_shards, k_per_tp)
        w_tp = w.reshape(tp_shards, k_per_tp, d_out)
        acc = jnp.zeros((b * t, d_out), dtype=jnp.float32)
        for r in range(tp_shards):
          part = local_mm(x_tp[:, r, :], w_tp[r, :, :])
          acc = acc + part.astype(jnp.float32)
        return acc.astype(self.config.dtype).reshape(b, t, d_out)

    if not use_canon:
      return linear_mod(x)

    if mesh is not None and tp_size > 1 and d_out % tp_size == 0:

      def local_col_proj(x_loc: jax.Array, w_loc: jax.Array) -> jax.Array:
        d_out_loc = int(w_loc.shape[1])

        def _mm(a_3d: jax.Array, weight_2d: jax.Array) -> jax.Array:
          out_2d = _canon_matmul_2d(
              a_3d.reshape(b * t, d_in),
              weight_2d,
              contract=self.contract,
              interpret=interpret,
              out_dtype=self.config.dtype,
          )
          return out_2d.reshape(b, t, d_out_loc)

        return _column_parallel_local(_mm, x_loc, w_loc)

      return jax.shard_map(
          local_col_proj,
          mesh=mesh,
          in_specs=(P(None, None, None), P(None, 'tp')),
          out_specs=P(None, None, 'tp'),
          check_vma=False,
      )(x, w)

    out_2d = _canon_matmul_2d(
        x.reshape(b * t, d_in),
        w,
        contract=self.contract,
        interpret=interpret,
        out_dtype=self.config.dtype,
    )
    return out_2d.reshape(b, t, d_out)

  def _gate_up_proj(
      self,
      x: jax.Array,
      *,
      use_canon: bool,
      interpret: bool,
      norm_weight: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Computes `(gate, up)` projections, fusing into 1 Pallas call when canonical.

    Latency & HBM Optimization (Horizontal Gate+Up Fusion):
      - Forward Latency: `gate_proj` and `up_proj` share the exact same input
        `x` `[B, T, D]` and contracting dimension `D`. Concatenating
        `[W_gate, W_up]` along the output column axis on each TP shard and
        invoking one fixed-tile matmul eliminates 1 Pallas kernel launch and
        1 redundant M-dimension padding per layer, while keeping output tiles
        100% bitwise identical.
      - HBM Residuals: the fused coat saves the input activation `x_2d` only
        ONCE per layer instead of twice (-50% MLP input residual memory), and
        `_canon_fused_matmul_2d` keeps the un-fused `W_gate/W_up` parameter
        views (not the concatenated copy) as its backward residual.
      - Backward: `dx` is one fused Pallas matmul (`dy_gu @ W_gu.T`) whose TP
        partials JAX reduces with the single column-parallel `psum` of the
        replicated `x` (see `_column_parallel_local`).

    Latency & HBM Optimization (RMSNorm prologue fusion,
    `use_canon_norm_matmul`):
      With `norm_weight` (the `post_attention_layernorm` gamma) given, `x` is
      the UN-normalized residual stream and the norm runs inside the
      projection kernel (`_canon_fused_norm_matmul_2d`, P56.4.6); see
      `CanonAttention._qkv_proj` for the launch / HBM accounting. Bitwise
      identical to the un-fused `post_attention_layernorm -> _gate_up_proj`
      chain.

    Args:
      x: `[B, T, D]` hidden states (un-normalized iff `norm_weight` is given).
      use_canon: Use the fixed-tile Pallas matmul.
      interpret: Pallas interpret mode.
      norm_weight: Optional `[D]` RMSNorm gamma to fuse as a prologue (requires
        `use_canon`; the caller checks `_norm_matmul_admits`).

    Returns:
      `(gate, up)`, each `[B, T, F]`.
    """
    if not use_canon:
      if norm_weight is not None:
        raise ValueError(
            'norm fusion requires the canonical Gate/Up projection'
        )
      return self.gate_proj(x), self.up_proj(x)

    b, t, d_in = x.shape
    w_gate = jnp.astype(self.gate_proj.kernel.value, self.config.dtype)
    w_up = jnp.astype(self.up_proj.kernel.value, self.config.dtype)
    gamma = (
        None
        if norm_weight is None
        else jnp.astype(norm_weight, self.config.dtype)
    )
    d_out = w_gate.shape[1]
    tp_size, mesh = _active_tp_size_and_mesh('tp')

    def fused_2d(
        a_2d: jax.Array, g: jax.Array | None, ws: Tuple[jax.Array, ...]
    ) -> jax.Array:
      if g is None:
        return _canon_fused_matmul_2d(
            a_2d,
            ws,
            contract=self.contract,
            interpret=interpret,
            out_dtype=self.config.dtype,
        )
      return _canon_fused_norm_matmul_2d(
          a_2d,
          g,
          ws,
          norm_eps=self.config.norm_eps,
          contract=self.contract,
          interpret=interpret,
          out_dtype=self.config.dtype,
      )

    if mesh is not None and tp_size > 1 and d_out % tp_size == 0:

      def local_gate_up_fused(
          x_loc: jax.Array,
          wg_loc: jax.Array,
          wu_loc: jax.Array,
          g_loc: jax.Array | None,
      ) -> tuple[jax.Array, jax.Array]:
        d_out_loc = int(wg_loc.shape[1])

        def _mm_gu(
            a_3d: jax.Array,
            wg_2d: jax.Array,
            wu_2d: jax.Array,
            g_1d: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array]:
          # Copy-free fused `[W_gate | W_up]` matmul (residual = param views).
          out_gu_2d = fused_2d(a_3d.reshape(b * t, d_in), g_1d, (wg_2d, wu_2d))
          g_2d, u_2d = jnp.split(out_gu_2d, [d_out_loc], axis=1)
          return (
              g_2d.reshape(b, t, d_out_loc),
              u_2d.reshape(b, t, d_out_loc),
          )

        return _column_parallel_local(_mm_gu, x_loc, wg_loc, wu_loc, g_loc)

      return jax.shard_map(
          local_gate_up_fused,
          mesh=mesh,
          in_specs=(
              P(None, None, None),
              P(None, 'tp'),
              P(None, 'tp'),
              P(None) if gamma is not None else None,
          ),
          out_specs=(P(None, None, 'tp'), P(None, None, 'tp')),
          check_vma=False,
      )(x, w_gate, w_up, gamma)

    out_gu_2d = fused_2d(x.reshape(b * t, d_in), gamma, (w_gate, w_up))
    g_2d, u_2d = jnp.split(out_gu_2d, [d_out], axis=1)
    return g_2d.reshape(b, t, d_out), u_2d.reshape(b, t, d_out)

  def block(
      self,
      x: jaxtyping.Array,
      norm_weight: jax.Array | None = None,
  ) -> jaxtyping.Array:
    cc = self.canon_config
    interpret = cc.resolve_interpret()

    gate, up = self._gate_up_proj(
        x,
        use_canon=cc.use_canon_mlp_proj,
        interpret=interpret,
        norm_weight=norm_weight,
    )

    if cc.use_canon_swiglu:
      b, t, f = gate.shape
      tp_size, mesh = _active_tp_size_and_mesh('tp')

      def swiglu_fwd(g: jax.Array, u: jax.Array) -> jax.Array:
        return padded_swiglu.swiglu(
            g,
            u,
            contract=self.contract,
            interpret=interpret,
            shape_invariant_numerics=True,
        )

      if mesh is not None and tp_size > 1 and f % tp_size == 0:

        def local_swiglu(g_loc: jax.Array, u_loc: jax.Array) -> jax.Array:
          f_loc = int(g_loc.shape[-1])
          g_2d = g_loc.reshape(b * t, f_loc).astype(jnp.bfloat16)
          u_2d = u_loc.reshape(b * t, f_loc).astype(jnp.bfloat16)
          act_2d = canonical_vjp.swiglu(g_2d, u_2d, forward=swiglu_fwd)
          return act_2d.astype(self.config.dtype).reshape(b, t, f_loc)

        activations = jax.shard_map(
            local_swiglu,
            mesh=mesh,
            in_specs=(P(None, None, 'tp'), P(None, None, 'tp')),
            out_specs=P(None, None, 'tp'),
            check_vma=False,
        )(gate, up)
      else:
        gate_2d = gate.reshape(b * t, f).astype(jnp.bfloat16)
        up_2d = up.reshape(b * t, f).astype(jnp.bfloat16)
        act_2d = canonical_vjp.swiglu(gate_2d, up_2d, forward=swiglu_fwd)
        activations = act_2d.astype(self.config.dtype).reshape(b, t, f)
    else:
      activations = nnx.silu(gate) * up

    activations = shard(
        activations, self.shd_config.act_btf  # pyrefly: ignore[bad-argument-type]
    )
    outputs = self._linear_proj(
        activations,
        self.down_proj,
        use_canon=cc.use_canon_mlp_proj,
        use_fixed_order_reduce=cc.use_fixed_order_reduce,
        contract_parallel=True,
        interpret=interpret,
    )
    return outputs

  @jax.named_scope('feed_forward')
  def __call__(
      self,
      x: jaxtyping.ArrayLike,
      norm_weight: jax.Array | None = None,
  ) -> jaxtyping.Array:
    if _remat_is(self.config, RematConfig.BLOCK):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, norm_weight=norm_weight
      )
    return self.block(x, norm_weight=norm_weight)  # pyrefly: ignore[bad-argument-type]


class CanonDecoderLayer(nnx.Module):
  """DecoderLayer with Zero-TIM kernel switches."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig,
      contract: model_contracts.ModelContract,
  ):
    self.config = config
    self.canon_config = canon_config
    self.contract = contract
    self.input_layernorm = CanonRMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.attn = CanonAttention(
        config=config,
        rngs=rngs,
        canon_config=canon_config,
        contract=contract,
    )
    self.post_attention_layernorm = CanonRMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if config.num_experts is None:
      self.mlp = CanonMLP(
          config=config,
          rngs=rngs,
          canon_config=canon_config,
          contract=contract,
      )
    else:
      self.mlp = qwen3_stock.MoELayer(
          config=config,
          rngs=rngs,
      )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    cc = self.canon_config
    interpret = cc.resolve_interpret()
    # P56.4.6 RMSNorm-prologue fusion (`use_canon_norm_matmul`): a static,
    # per-site decision. It only replaces a chain that is *already* canonical
    # at both ends (Pallas norm -> fixed-tile projection) and whose contracting
    # width the fused kernel admits unpadded (`_norm_matmul_admits`); every
    # other combination keeps the two-kernel chain, so flipping this switch
    # alone never changes which arithmetic runs -- only where the normalized
    # rows live (VMEM scratch vs an HBM tensor + residual).
    fuse_norm = (
        cc.use_canon_norm_matmul
        and cc.use_canon_rmsnorm
        and _norm_matmul_admits(int(x.shape[-1]), self.contract)
    )
    fuse_attn_norm = fuse_norm and cc.use_canon_qkv_proj
    mlp = self.mlp
    fuse_mlp_norm = fuse_norm and cc.use_canon_mlp_proj

    if fuse_attn_norm:
      inputs_normalized = x
      attn_norm_weight = self.input_layernorm.w.value
    else:
      inputs_normalized = self.input_layernorm(
          x, use_canon=cc.use_canon_rmsnorm, interpret=interpret
      )
      attn_norm_weight = None
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
        segment_ids=segment_ids,
        norm_weight=attn_norm_weight,
    )
    # `HARD_ROUNDING_RESIDUAL`: the two residual adds round through
    # `reduce_precision`; `canon_all_disabled` stays the stock program.
    hard_residual = _hard(HARD_ROUNDING_RESIDUAL) and cc.any_enabled()
    attn_output = _residual_add(attn_output, x, hard_residual)
    residual = attn_output
    if isinstance(mlp, CanonMLP) and fuse_mlp_norm:
      outputs = mlp(
          attn_output, norm_weight=self.post_attention_layernorm.w.value
      )
    else:
      attn_output = self.post_attention_layernorm(
          attn_output, use_canon=cc.use_canon_rmsnorm, interpret=interpret
      )
      outputs = mlp(attn_output)
    outputs = _residual_add(residual, outputs, hard_residual)
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if _remat_is(self.config, RematConfig.DECODER):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, segment_pos, cache, attn_mask, segment_ids=segment_ids
      )
    return self.block(x, segment_pos, cache, attn_mask, segment_ids=segment_ids)


class Qwen3Canon(BackendMappingMixin, nnx.Module):
  """Canonical Qwen3 model with configurable Zero-TIM kernel switches."""

  BACKEND_PACKAGE_PATH = qwen3_stock.__name__

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      canon_config: CanonKernelConfig = CanonKernelConfig.all_enabled(),
      contract: model_contracts.ModelContract | None = None,
  ):
    self.config = config
    self.canon_config = canon_config
    tp_size, _ = _active_tp_size_and_mesh('tp')
    self.contract = (
        resolve_or_build_contract(config, tp_size=tp_size)
        if contract is None
        else contract
    )
    self.embedder = CanonEmbedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        CanonDecoderLayer(
            config=config,
            rngs=rngs,
            canon_config=canon_config,
            contract=self.contract,
        )
        for _ in range(config.num_layers)
    ])
    self.final_norm = CanonRMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = CanonEinsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=self.config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def set_canon_config(self, canon_config: CanonKernelConfig) -> None:
    """Updates `canon_config` across the model and all decoder layers in-place."""
    self.canon_config = canon_config
    for layer in self.layers:
      layer.canon_config = canon_config
      layer.attn.canon_config = canon_config
      if isinstance(layer.mlp, CanonMLP):
        layer.mlp.canon_config = canon_config

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initializes the KV cache for the model."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    return {
        f'layer_{i}': {
            'k': jnp.zeros(shape, dtype=config.dtype),
            'v': jnp.zeros(shape, dtype=config.dtype),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
        for i in range(config.num_layers)
    }

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
      segment_ids: jaxtyping.Array | None = None,  # [B, L]
      skip_lm_head: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    new_cache: Cache | None = None if cache is None else {}
    cc = self.canon_config
    x = self.embedder.encode(
        input_tokens, use_fixed_order_reduce=cc.use_fixed_order_reduce
    )

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          segment_ids=segment_ids,
      )
      if new_cache is not None and layer_cache is not None:
        new_cache[layer_name] = layer_cache

    interpret = cc.resolve_interpret()
    x = self.final_norm(x, use_canon=cc.use_canon_rmsnorm, interpret=interpret)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)

    if skip_lm_head:
      return x, new_cache

    logits = self.compute_final_logits(x)
    return logits, new_cache

  def compute_final_logits(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes final logits in the LM head's dtype (`config.dtype`, bf16).

    Stock `Qwen3` returns `astype(bf16_head_output, f32)`. The values are the
    same (the upcast is exact) and every consumer of this model upcasts exactly
    where it needs f32: `compute_token_logprobs` / `compute_log_softmax` do so
    in-kernel on the vocab-parallel path and before the library call
    otherwise, and `jnp.argmax` is dtype-agnostic on exact values.

    HBM & Latency Optimization (A1): on the stock model XLA fuses that convert
    into the `log_softmax` reductions, but on the canonical model the f32
    `[B, T, V/TP]` tensor is an operand of a Pallas custom call and therefore
    a materialized copy (8 MB per device at `M=1024, V/TP=2048`), which before
    A1 was re-copied by the `[:, :-1, :]` scoring slice and the row padding
    downstream (both removed by `next_token_logprobs`). Returning the native
    dtype removes the copy chain and halves the bytes the logprob kernels,
    their residual and their cotangent move.

    Args:
      x: Final-norm output `[B, T, D]`.

    Returns:
      Logits `[B, T, V]` in `config.dtype`.
    """
    cc = self.canon_config
    interpret = cc.resolve_interpret()
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(
          x,
          use_fixed_lm_head=cc.use_fixed_lm_head,
          contract=self.contract,
          interpret=interpret,
      )
    else:
      logits = self.lm_head(
          x,
          use_canon=cc.use_fixed_lm_head,
          contract=self.contract,
          interpret=interpret,
      )
    return jnp.astype(logits, self.config.dtype)

  def compute_log_softmax(
      self,
      logits: jaxtyping.Array,
      *,
      exact_residual_dtype: jaxtyping.DTypeLike | None = None,
  ) -> jaxtyping.Array:
    """Computes log-probabilities over the vocabulary.

    Args:
      logits: `[..., V]` float logits.
      exact_residual_dtype: See `compute_log_softmax`; pass `self.config.dtype`
        only for logits produced by this model's `compute_final_logits`.

    Returns:
      `float32` log-probabilities of the same shape.
    """
    return compute_log_softmax(
        logits,
        use_canon_logsoftmax=self.canon_config.use_canon_logsoftmax,
        interpret=self.canon_config.resolve_interpret(),
        exact_residual_dtype=exact_residual_dtype,
    )

  def compute_token_logprobs(
      self,
      logits: jaxtyping.Array,
      token_ids: jaxtyping.Array,
      *,
      exact_residual_dtype: jaxtyping.DTypeLike | None = None,
  ) -> jaxtyping.Array:
    """Computes gathered log-probabilities for specific `token_ids`.

    Args:
      logits: `[..., V]` float logits.
      token_ids: `[...]` integer targets.
      exact_residual_dtype: See `compute_token_logprobs`; pass
        `self.config.dtype` only for logits produced by this model's
        `compute_final_logits`.

    Returns:
      `float32[...]` per-token log-probabilities.
    """
    return compute_token_logprobs(
        logits,
        token_ids,
        use_canon_logsoftmax=self.canon_config.use_canon_logsoftmax,
        interpret=self.canon_config.resolve_interpret(),
        exact_residual_dtype=exact_residual_dtype,
    )

  def get_model_input(self):
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }


def _logsoftmax_vjp_coat(
    local_fn,
    *,
    mesh: jax.sharding.Mesh | None,
    in_specs: Tuple[P, ...],
    out_spec: P,
    differentiable: Tuple[bool, ...],
    exact_residual_dtype: jaxtyping.DTypeLike | None,
):
  """Wraps a canonical log-softmax body for TP and/or a reduced residual.

  HBM & Latency Optimization (Fwd+Bwd, `use_canon_logsoftmax`):
    Under TP the log-softmax runs in a `check_vma=False` shard_map with
    replicated `[M, V]` logits on every rank. Differentiating *through* that
    map makes JAX divide the (rank-identical) cotangent by TP and `psum` the
    `[M, V]` f32 logits cotangent back over TP: one all-reduce of `4.M.V` bytes
    per rank (`32 MB` at `M=1024, V=8192`, the TPU bench; `~600 MB` for
    Qwen3's `V=151936` at `M=1024`), a second `[M, V]` transient for its
    output, and a re-rounded `(c/4 + c/4) + c/4 + c/4` chain.
    `_shard_map_with_local_vjp` keeps the forward program verbatim and
    evaluates the body's own `custom_vjp` inside a second forward map: no
    collective, exact cotangent.

    `exact_residual_dtype` additionally lets the caller store the logits
    residual in the dtype the LM head produced them in (`model.config.dtype`,
    bf16): `compute_final_logits` returns `astype(bf16_logits, f32)` for both
    the stock and the canonical head, so the bf16 round trip is bit-exact and
    the residual shrinks from `4.M.V` to `2.M.V` bytes while the cotangent
    stays f32 (the coat upcasts before differentiating). It also replaces the
    kernel's own `(logits f32, normalizer)` residual, which would otherwise be
    kept by the inner `custom_vjp`. The cost is one re-run of the Stages 1+2
    normalizer pass over `[M, V]` in the backward, i.e. the same HBM traffic
    the removed all-reduce had to perform anyway. The coat is only attached
    when it buys one of the two (TP > 1 or a reduced residual dtype); at TP=1
    with the primal dtype the body runs bare, exactly as before.

  Args:
    local_fn: The per-rank body (`log_softmax` or `token_logprobs`).
    mesh: Active TP mesh, or `None`.
    in_specs: Replicated per-operand specs.
    out_spec: Replicated output spec.
    differentiable: Per-operand cotangent flags (False for token ids).
    exact_residual_dtype: Optional storage dtype of the logits residual (operand
      0); `None` keeps f32.

  Returns:
    A callable with the forward semantics of `shard_map(local_fn)` (or
    `local_fn` when `mesh` is None).
  """
  if mesh is None and exact_residual_dtype is None:
    return local_fn
  residual_dtypes = (exact_residual_dtype,) + (None,) * (len(in_specs) - 1)
  return _shard_map_with_local_vjp(
      local_fn,
      mesh=mesh,
      in_specs=in_specs,
      out_spec=out_spec,
      differentiable=differentiable,
      residual_dtypes=residual_dtypes,
  )


def vocab_parallel_logsoftmax_admitted(vocab: int, tp_size: int) -> bool:
  """Whether `[M, V]` logits sharded `V/TP` per rank admit the vocab-parallel path.

  `_vocab_parallel_normalizer` reproduces the library's tile grouping only when
  every rank's `V/TP` columns are whole `VOCAB_TILE=1024` tiles: rank `r`'s
  tile summaries are then exactly the library's global tiles
  `[r * tiles_local, (r + 1) * tiles_local)` and Stage 2 combines them in the
  library's left-to-right order. A non-aligned shard would have to pad *inside*
  the global tile sequence, which defines a different (self-consistent, but not
  engine-compatible) reduction, so such shapes fall back to the gathered
  full-vocabulary library path. Qwen3's `V=151936` is not admitted at
  TP=2/4/8 (`V/TP % 1024` = 192/96/560); the bench's `V=8192, TP=4`
  (`v_local=2048`) and the CPU tests' `V=4096, TP=4` are.

  Args:
    vocab: Full vocabulary size `V`.
    tp_size: Tensor-parallel degree the logits are sharded over.

  Returns:
    True when the vocab-parallel normalizer is bitwise the library normalizer.
  """
  return (
      tp_size > 1
      and vocab % tp_size == 0
      and (vocab // tp_size) % canonical_logsoftmax.VOCAB_TILE == 0
  )


def _vocab_parallel_normalizer(
    logits_local: jax.Array,
    *,
    tp_axis: str,
    tp_size: int,
    interpret: bool,
) -> jax.Array:
  """Stages 1+2 normalizer on TP-sharded `logits_local[m, v_local]` (any `m`).

  Must be invoked inside `jax.shard_map` over `tp_axis` with
  `v_local % VOCAB_TILE == 0` (see `vocab_parallel_logsoftmax_admitted`).
  - Stage 1 (`partial_kernel`) runs locally on each TP rank over its `v_local`
    slice (`tiles_local = v_local / 1024` tiles), avoiding the `[M, V]`
    `all_gather` and cutting Stage 1 Pallas work by `tp_size`x. The grid walks
    `(row block, tile)` over the 2-D operand directly and the operand may be
    bf16 (upcast in-kernel, exactly as the library's `partial_kernel` does), so
    neither an f32 copy nor a tile-splitting reshape (a relayout copy on TPU)
    of the logits is ever materialized (A1 / A1b).
  - Bitwise contract: a lane reduction is NOT shape-invariant under XLA (CPU
    interpret mode reduces a `(1, 1024)` or `(1, 1, 2, 1024)` block in a
    different order than the library's `(1, 1, 8, 1024)` block: 72% of the
    tile sums differ by an ulp, ~5% of the normalizers). Every Stage 1 grid
    step therefore consumes `TILES_PER_GROUP * block_rows` rows and reshapes
    them in-kernel to the library's exact value shape
    `[block_rows, 1, TILES_PER_GROUP, VOCAB_TILE]` -- the rows stand in for a
    group's tiles; each `(row, tile)` reduction is over the same 1024 values
    in the same program -- before running the library's `partial_kernel`
    arithmetic verbatim.
  - Only the per-tile `(local_max, local_sum)` summaries (one
    `[2, mp, tiles_local]` f32 array, `16 KB` at `mp=1024, tiles_local=2`) are
    all-gathered across `tp_axis`, in a single collective.
  - Stage 2 (`combine_kernel`) combines all `tp_size * tiles_local` tile
    summaries in exact left-to-right tile order (library `TILES_PER_GROUP`
    padding included) on the library's in-kernel VALUE shape
    `(block_rows, vocab_groups * TILES_PER_GROUP)`, read from the compact
    `[mp, padded_tiles]` summaries and written as a lane-dense `[mp, 128]`
    column (no 128-lane broadcast of the operands, no `[mp, 1, 8, 128]`
    output), producing a `log_normalizer[m]` that is bitwise identical to the
    library's unsharded `canonical_logsoftmax._pallas_normalizer`; pinned by
    `qwen3_model_canon_test.test_vocab_parallel_logsoftmax_matches_library_bitwise`.
  Every stage is row-independent, so `m` is only padded to the Stage 1 row
  step (`64` on TPU, `8` in interpret mode) and never chunked: one program for
  `m=1024` instead of four 256-row `lax.map` iterations, each with its own two
  launches and two collectives.

  Args:
    logits_local: This rank's `[m, v_local]` logits shard (bf16 or f32).
    tp_axis: Mesh axis name of the vocabulary sharding.
    tp_size: Tensor-parallel degree.
    interpret: Pallas interpret mode (CPU tests).

  Returns:
    The f32 `log_normalizer[m]` (`logsumexp` of each full row).
  """
  m, v_local = map(int, logits_local.shape)
  block_rows = 1 if interpret else canonical_logsoftmax.ROW_BUCKET_ALIGN
  vocab_tile = canonical_logsoftmax.VOCAB_TILE
  tiles_per_group = canonical_logsoftmax.TILES_PER_GROUP
  summary_align = canonical_logsoftmax.SUMMARY_ALIGN
  if v_local % vocab_tile:
    raise ValueError(
        f'vocab-parallel normalizer needs v_local % {vocab_tile} == 0, got'
        f' v_local={v_local}; gate on vocab_parallel_logsoftmax_admitted()'
    )
  tiles_local = v_local // vocab_tile
  rows_per_step = tiles_per_group * block_rows

  mp = _round_up(m, rows_per_step)
  if mp != m:
    # Padded rows are independent of the real rows and dropped below.
    logits_local = jnp.pad(
        logits_local,
        ((0, mp - m), (0, 0)),
        constant_values=jnp.zeros((), dtype=logits_local.dtype),
    )

  def partial_kernel(x_ref, max_ref, sum_ref):
    # `[rows_per_step, VOCAB_TILE]` -> the library's
    # `[block_rows, 1, TILES_PER_GROUP, VOCAB_TILE]` (a vreg re-indexing on
    # TPU: both second-minor extents are sublane multiples, the lane extent is
    # unchanged), then the library's `partial_kernel` body verbatim.
    x = x_ref[...].astype(jnp.float32)
    x = x.reshape(block_rows, 1, tiles_per_group, vocab_tile)
    tile_max = jnp.max(x, axis=-1)
    tile_sum = jnp.sum(
        jnp.exp(x - tile_max[..., None]), axis=-1, dtype=jnp.float32
    )
    max_ref[...] = jnp.broadcast_to(tile_max[..., None], max_ref.shape)
    sum_ref[...] = jnp.broadcast_to(tile_sum[..., None], sum_ref.shape)

  # Summaries keep the library's `[.., 1, TILES_PER_GROUP, SUMMARY_ALIGN]`
  # block so the stores match too; `(mp // 8, 1, 8, T * 128)` is the row-major
  # `[mp, T * 128]` with row `8 * i + t` at `[i, 0, t]`.
  summary_shape = jax.ShapeDtypeStruct(
      (mp // tiles_per_group, 1, tiles_per_group, tiles_local * summary_align),
      jnp.float32,
  )
  partial_max, partial_sum = pl.pallas_call(
      partial_kernel,
      out_shape=(summary_shape, summary_shape),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((rows_per_step, vocab_tile), lambda i, j: (i, j)),
          ],
          out_specs=[
              pl.BlockSpec(
                  (block_rows, 1, tiles_per_group, summary_align),
                  lambda i, j: (i, 0, 0, j),
              ),
              pl.BlockSpec(
                  (block_rows, 1, tiles_per_group, summary_align),
                  lambda i, j: (i, 0, 0, j),
              ),
          ],
          grid=(mp // rows_per_step, tiles_local),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'parallel'),
          allow_input_fusion=(False,),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f'canon_logsoftmax_vp_partial_m{mp}_vl{v_local}',
  )(logits_local)

  # Lane 0 of each 128-lane summary block holds the tile value; gather both
  # summaries with one collective, rank-ordered along the tile axis.
  local_summary = jnp.stack([
      partial_max.reshape(mp, tiles_local, summary_align)[:, :, 0],
      partial_sum.reshape(mp, tiles_local, summary_align)[:, :, 0],
  ])  # [2, mp, tiles_local]
  all_summary = lax.all_gather(local_summary, tp_axis, axis=2, tiled=True)
  all_max = all_summary[0]  # [mp, tp_size * tiles_local]
  all_sum = all_summary[1]

  total_tiles = tp_size * tiles_local
  vocab_groups = (total_tiles + tiles_per_group - 1) // tiles_per_group
  padded_tiles = vocab_groups * tiles_per_group
  if padded_tiles != total_tiles:
    # A fully padded library tile (all `finfo.min` columns) summarizes to
    # `(finfo.min, VOCAB_TILE)`; its Stage 2 contribution vanishes exactly.
    all_max = jnp.pad(
        all_max,
        ((0, 0), (0, padded_tiles - total_tiles)),
        constant_values=jnp.finfo(jnp.float32).min,
    )
    all_sum = jnp.pad(
        all_sum,
        ((0, 0), (0, padded_tiles - total_tiles)),
        constant_values=jnp.float32(vocab_tile),
    )

  # Stage 2 consumes the compact `[mp, padded_tiles]` summaries directly. The
  # library's `combine_kernel` loads lane 0 of a lane-padded
  # `(block_rows, vocab_groups, TILES_PER_GROUP, SUMMARY_ALIGN)` block and
  # reshapes it to `(block_rows, vocab_groups * TILES_PER_GROUP)` before its
  # two reductions; a `(block_rows, padded_tiles)` block IS that value (same
  # shape, same lane positions), so the reductions are the library's, while
  # the lane-padded operands (`2 * mp * SUMMARY_ALIGN * 4` bytes, 8 MB at
  # `mp=1024`) and the lane-padded `[mp, 1, 8, 128]` output (4 MB) are never
  # materialized: Step 8 measured them as a `+2.7 MB` Prefill / `+9.6 MB`
  # Fwd+Bwd peak regression once the 256-row chunking was removed (A1b).
  padded_tiles = vocab_groups * tiles_per_group

  def combine_kernel(max_ref, sum_ref, norm_ref):
    l_max = max_ref[...].astype(jnp.float32)
    l_sum = sum_ref[...].astype(jnp.float32)
    global_max = jnp.max(l_max, axis=-1)
    global_sum = jnp.sum(
        l_sum * jnp.exp(l_max - global_max[:, None]),
        axis=-1,
        dtype=jnp.float32,
    )
    normalizer = global_max + jnp.log(global_sum)
    norm_ref[...] = jnp.broadcast_to(normalizer[:, None], norm_ref.shape)

  log_normalizer = pl.pallas_call(
      combine_kernel,
      out_shape=jax.ShapeDtypeStruct((mp, summary_align), jnp.float32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_rows, padded_tiles), lambda row: (row, 0)),
              pl.BlockSpec((block_rows, padded_tiles), lambda row: (row, 0)),
          ],
          out_specs=pl.BlockSpec(
              (block_rows, summary_align), lambda row: (row, 0)
          ),
          grid=(mp // block_rows,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel',),
          allow_input_fusion=(False, False),
          shape_invariant_numerics=True,
      ),
      interpret=interpret,
      name=f'canon_logsoftmax_vp_combine_m{mp}_t{total_tiles}',
  )(all_max, all_sum)
  return log_normalizer[:m, 0]


def _vocab_parallel_log_softmax(
    flat_logits: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    tp_axis: str,
    tp_size: int,
    interpret: bool,
    exact_residual_dtype: jaxtyping.DTypeLike | None,
) -> jax.Array:
  """Vocab-parallel `log_softmax` of the `P(None, 'tp')` `flat_logits[m, V]`.

  `flat_logits` may be bf16 (the LM head's native dtype): the normalizer kernel
  upcasts in-kernel and the output is f32, so the result is bitwise the result
  for `flat_logits.astype(f32)` without materializing that copy. The cotangent
  is returned in the primal dtype; it is formed in f32 and rounded once, which
  is exactly what differentiating through an outer `astype(f32)` did before.

  Args:
    flat_logits: `[m, V]` logits, `P(None, tp_axis)` sharded.
    mesh: The TP mesh.
    tp_axis: Mesh axis name of the vocabulary sharding.
    tp_size: Tensor-parallel degree.
    interpret: Pallas interpret mode (CPU tests).
    exact_residual_dtype: Optional storage dtype of the logits residual; `None`
      keeps the primal dtype.

  Returns:
    The f32 `log_softmax` `[m, V]`, sharded like `flat_logits`.
  """

  def _fwd_local(logits_local: jax.Array):
    normalizer = _vocab_parallel_normalizer(
        logits_local,
        tp_axis=tp_axis,
        tp_size=tp_size,
        interpret=interpret,
    )
    out_local = logits_local.astype(jnp.float32) - normalizer[:, None]
    logits_res = (
        logits_local.astype(exact_residual_dtype)
        if exact_residual_dtype is not None
        else logits_local
    )
    return out_local, (logits_res, normalizer)

  def _bwd_local(res_local, cot_local: jax.Array):
    logits_res, normalizer = res_local
    logits_f32_local = logits_res.astype(jnp.float32)
    cot_f32_local = cot_local.astype(jnp.float32)
    prob_local = jnp.exp(logits_f32_local - normalizer[:, None])
    cot_sum = lax.psum(jnp.sum(cot_f32_local, axis=-1, keepdims=True), tp_axis)
    grad_f32 = cot_f32_local - prob_local * cot_sum
    return grad_f32.astype(flat_logits.dtype)

  fwd_shmap = jax.shard_map(
      _fwd_local,
      mesh=mesh,
      in_specs=(P(None, tp_axis),),
      out_specs=(P(None, tp_axis), (P(None, tp_axis), P(None))),
      check_vma=False,
  )
  bwd_shmap = jax.shard_map(
      _bwd_local,
      mesh=mesh,
      in_specs=((P(None, tp_axis), P(None)), P(None, tp_axis)),
      out_specs=P(None, tp_axis),
      check_vma=False,
  )

  @jax.custom_vjp
  def op(logits_in: jax.Array) -> jax.Array:
    out, _ = fwd_shmap(logits_in)
    return out

  def _fwd(logits_in: jax.Array):
    return fwd_shmap(logits_in)

  def _bwd(residual, cotangent: jax.Array):
    return (bwd_shmap(residual, cotangent),)

  op.defvjp(_fwd, _bwd)
  return op(flat_logits)


def _vocab_parallel_token_logprobs(
    flat_logits: jax.Array,
    flat_tokens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    tp_axis: str,
    tp_size: int,
    interpret: bool,
    exact_residual_dtype: jaxtyping.DTypeLike | None,
) -> jax.Array:
  """Vocab-parallel `token_logprobs` of the `P(None, 'tp')` `flat_logits[m, V]`.

  `flat_logits` may be bf16 (the LM head's native dtype, see
  `Qwen3Canon.compute_final_logits`): the normalizer kernel upcasts in-kernel
  and the gathered target column is upcast after the gather (both exact), so
  the result is bitwise the result for `flat_logits.astype(f32)` while the
  `[m, V/TP]` f32 copy, its residual and its f32 cotangent are never
  materialized. The cotangent is formed in f32 (`-p * ct`, plus `ct` at the
  target column) and rounded once to the primal dtype -- the rounding point an
  outer `astype(f32)` transposed to before.

  Args:
    flat_logits: `[m, V]` logits, `P(None, tp_axis)` sharded.
    flat_tokens: `[m]` int32 target token ids (replicated).
    mesh: The TP mesh.
    tp_axis: Mesh axis name of the vocabulary sharding.
    tp_size: Tensor-parallel degree.
    interpret: Pallas interpret mode (CPU tests).
    exact_residual_dtype: Optional storage dtype of the logits residual; `None`
      keeps the primal dtype.

  Returns:
    The f32 per-row target log-probabilities `[m]` (replicated).
  """
  m, vocab = map(int, flat_logits.shape)
  v_local = vocab // tp_size

  def _fwd_local(logits_local: jax.Array, tok_1d: jax.Array):
    normalizer = _vocab_parallel_normalizer(
        logits_local,
        tp_axis=tp_axis,
        tp_size=tp_size,
        interpret=interpret,
    )
    axis_idx = lax.axis_index(tp_axis)
    v_start = axis_idx * v_local
    local_tok = tok_1d - v_start
    in_shard = (local_tok >= 0) & (local_tok < v_local)
    safe_tok = jnp.where(in_shard, local_tok, 0)
    gathered = jnp.take_along_axis(logits_local, safe_tok[:, None], axis=-1)[
        :, 0
    ].astype(jnp.float32)
    token_col = lax.psum(
        jnp.where(in_shard, gathered, jnp.float32(0.0)), tp_axis
    )
    token_logprob = token_col - normalizer
    logits_res = (
        logits_local.astype(exact_residual_dtype)
        if exact_residual_dtype is not None
        else logits_local
    )
    return token_logprob, (logits_res, normalizer, tok_1d)

  def _bwd_local(res_local, cot_1d: jax.Array):
    logits_res, normalizer, tok_1d = res_local
    logits_f32_local = logits_res.astype(jnp.float32)
    cot_f32 = cot_1d.astype(jnp.float32)
    prob_local = jnp.exp(logits_f32_local - normalizer[:, None])
    grad_local = -prob_local * cot_f32[:, None]
    axis_idx = lax.axis_index(tp_axis)
    v_start = axis_idx * v_local
    local_tok = tok_1d - v_start
    in_shard = (local_tok >= 0) & (local_tok < v_local)
    safe_tok = jnp.where(in_shard, local_tok, 0)
    rows = jnp.arange(m, dtype=jnp.int32)
    grad_local = grad_local.at[rows, safe_tok].add(
        jnp.where(in_shard, cot_f32, jnp.float32(0.0))
    )
    return grad_local.astype(flat_logits.dtype)

  fwd_shmap = jax.shard_map(
      _fwd_local,
      mesh=mesh,
      in_specs=(P(None, tp_axis), P(None)),
      out_specs=(P(None), (P(None, tp_axis), P(None), P(None))),
      check_vma=False,
  )
  bwd_shmap = jax.shard_map(
      _bwd_local,
      mesh=mesh,
      in_specs=((P(None, tp_axis), P(None), P(None)), P(None)),
      out_specs=P(None, tp_axis),
      check_vma=False,
  )

  @jax.custom_vjp
  def op(logits_in: jax.Array, tok_in: jax.Array) -> jax.Array:
    out, _ = fwd_shmap(logits_in, tok_in)
    return out

  def _fwd(logits_in: jax.Array, tok_in: jax.Array):
    return fwd_shmap(logits_in, tok_in)

  def _bwd(residual, cotangent: jax.Array):
    return bwd_shmap(residual, cotangent), None

  op.defvjp(_fwd, _bwd)
  return op(flat_logits, flat_tokens)


def compute_log_softmax(
    logits: jaxtyping.Array,
    *,
    use_canon_logsoftmax: bool = True,
    interpret: bool | None = None,
    exact_residual_dtype: jaxtyping.DTypeLike | None = None,
) -> jax.Array:
  """Computes full-vocabulary log-softmax (stock or Zero-TIM canonical).

  Handles arbitrary batch/sequence dimensions by flattening to 2D `[M, V]`,
  padding `M` to a valid `ROW_BUCKET_ALIGN=8` bucket (or chunking by
  `PRODUCTION_M=256`), and invoking `canonical_logsoftmax.log_softmax`. Under a
  TP mesh whose vocabulary shards are whole 1024-column tiles
  (`vocab_parallel_logsoftmax_admitted`) the vocab-parallel program runs
  instead; it consumes `logits` in their native dtype (the kernel upcasts) and
  is bitwise the library program.

  Args:
    logits: Float array `[..., V]`.
    use_canon_logsoftmax: Whether to use Zero-TIM `canonical_logsoftmax`.
    interpret: Optional Pallas interpret mode override.
    exact_residual_dtype: Optional dtype in which `logits` are known to be
      exactly representable (e.g. `model.config.dtype` for logits returned by
      `compute_final_logits`); the backward residual is stored in it. See
      `_logsoftmax_vjp_coat`. Canonical path only.

  Returns:
    Log-probabilities of the same shape as `logits` in `float32`.
  """
  if not use_canon_logsoftmax:
    return jax.nn.log_softmax(jnp.astype(logits, jnp.float32), axis=-1)

  if interpret is None:
    interpret = jax.default_backend() == 'cpu'
  vocab = int(logits.shape[-1])

  os.environ.setdefault(canonical_logsoftmax.ENV, '1')
  orig_shape = logits.shape
  m = int(logits.size // vocab)

  row_align = 1 if interpret else canonical_logsoftmax.ROW_BUCKET_ALIGN
  prod_m = canonical_logsoftmax.PRODUCTION_M
  tp_size, mesh = _active_tp_size_and_mesh('tp')

  if (
      VOCAB_PARALLEL_LOGSOFTMAX
      and mesh is not None
      and vocab_parallel_logsoftmax_admitted(vocab, tp_size)
  ):
    out_2d = _vocab_parallel_log_softmax(
        logits.reshape(m, vocab),
        mesh=mesh,
        tp_axis='tp',
        tp_size=tp_size,
        interpret=interpret,
        exact_residual_dtype=exact_residual_dtype,
    )
    return out_2d.reshape(orig_shape)

  logits_f32 = jnp.astype(logits, jnp.float32)
  flat = logits_f32.reshape(m, vocab)

  def _run_2d(mat_2d: jax.Array) -> jax.Array:
    return canonical_logsoftmax.log_softmax(
        mat_2d, interpret=interpret, strict_vocab=False
    )

  run_2d = _logsoftmax_vjp_coat(
      _run_2d,
      mesh=mesh if (mesh is not None and tp_size > 1) else None,
      in_specs=(P(None, None),),
      out_spec=P(None, None),
      differentiable=(True,),
      exact_residual_dtype=exact_residual_dtype,
  )

  if m <= prod_m:
    mp = _round_up(m, row_align)
    if mp != m:
      padded = jnp.pad(
          flat, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
      )
    else:
      padded = flat
    out_2d = run_2d(padded)[:m, :]
    return out_2d.reshape(orig_shape)

  mp = _round_up(m, prod_m)
  if mp != m:
    padded = jnp.pad(
        flat, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
    )
  else:
    padded = flat
  chunks = padded.reshape(-1, prod_m, vocab)
  out_chunks = lax.map(run_2d, chunks)
  return out_chunks.reshape(mp, vocab)[:m, :].reshape(orig_shape)


def compute_token_logprobs(
    logits: jaxtyping.Array,
    token_ids: jaxtyping.Array,
    *,
    use_canon_logsoftmax: bool = True,
    interpret: bool | None = None,
    exact_residual_dtype: jaxtyping.DTypeLike | None = None,
) -> jax.Array:
  """Computes per-token log-probabilities for `token_ids` (shape `[...]`).

  HBM & Latency Optimization:
  When `use_canon_logsoftmax=True`, routes directly to
  `canonical_logsoftmax.token_logprobs` (or `_vocab_parallel_token_logprobs`
  under a TP mesh when `VOCAB_PARALLEL_LOGSOFTMAX=True` and
  `vocab_parallel_logsoftmax_admitted(V, TP)`) instead of materializing the
  full `[M, V]` `float32` `compute_log_softmax(logits)` tensor and slicing with
  `jnp.take_along_axis`:
  - Forward skips Stage 3 (`normalize_kernel`) and subtracts the per-row
    Stages 1+2 `log_normalizer[M]` directly from the gathered target column
    `logits[row, token_ids[row]]`, eliminating ~24 MB (`M=64`) to ~152 MB
    (`M=256`) of temporary HBM in Prefill while remaining 100% bitwise
    identical to `take_along_axis(compute_log_softmax(logits), token_ids)`.
  - Under TP > 1 (`VOCAB_PARALLEL_LOGSOFTMAX=True`),
    `_vocab_parallel_token_logprobs` consumes `logits` directly in its native
    `P(None, 'tp')` vocabulary sharding `[M, V/TP]` AND in its native dtype
    (bf16 from `compute_final_logits`; the Stage 1 kernel upcasts in-kernel)
    without all-gathering the full `[M, V]` tensor or materializing an f32
    copy of the shard: Stage 1 runs on `1/TP` columns per rank, only the tiny
    `[2, M, tiles_local]` tile summaries (`16 KB` at `M=1024, tiles_local=2`)
    are all-gathered for Stage 2, the saved backward residual is the
    `[M, V/TP]` bf16 shard itself plus `normalizer[M]` (`4 KB`), and backward
    computes `dL/dlogits` locally in `[M, V/TP]` without re-running Stages 1+2
    or executing any collectives. Shards that are not whole 1024-column tiles
    (e.g. `V=151936` at TP=2/4/8) fall back to the gathered library program,
    which is the engine's.

  Args:
    logits: Logits tensor `[..., V]` (any float dtype; bf16 is consumed without
      an upcast copy on the vocab-parallel path).
    token_ids: Integer token IDs `[...]`.
    use_canon_logsoftmax: Whether to use Zero-TIM `canonical_logsoftmax`.
    interpret: Optional Pallas interpret mode override.
    exact_residual_dtype: Optional dtype in which `logits` are known to be
      exactly representable (`model.config.dtype` for `compute_final_logits`
      outputs); see `_logsoftmax_vjp_coat`. Canonical path only.

  Returns:
    Per-token log-probabilities `float32[...]` matching `token_ids.shape`.
  """
  if not use_canon_logsoftmax:
    log_probs = compute_log_softmax(
        logits,
        use_canon_logsoftmax=False,
        interpret=interpret,
    )
    return jnp.take_along_axis(
        log_probs, token_ids.astype(jnp.int32)[..., None], axis=-1
    )[..., 0]

  if interpret is None:
    interpret = jax.default_backend() == 'cpu'
  vocab = int(logits.shape[-1])

  os.environ.setdefault(canonical_logsoftmax.ENV, '1')
  orig_token_shape = token_ids.shape
  m = int(logits.size // vocab)
  flat_tokens = token_ids.astype(jnp.int32).reshape(m)

  row_align = 1 if interpret else canonical_logsoftmax.ROW_BUCKET_ALIGN
  prod_m = canonical_logsoftmax.PRODUCTION_M
  tp_size, mesh = _active_tp_size_and_mesh('tp')

  if (
      VOCAB_PARALLEL_LOGSOFTMAX
      and mesh is not None
      and vocab_parallel_logsoftmax_admitted(vocab, tp_size)
  ):
    out_1d = _vocab_parallel_token_logprobs(
        logits.reshape(m, vocab),
        flat_tokens,
        mesh=mesh,
        tp_axis='tp',
        tp_size=tp_size,
        interpret=interpret,
        exact_residual_dtype=exact_residual_dtype,
    )
    return out_1d.reshape(orig_token_shape)

  flat_logits = jnp.astype(logits, jnp.float32).reshape(m, vocab)

  def _run_2d(mat_2d: jax.Array, tok_1d: jax.Array) -> jax.Array:
    return canonical_logsoftmax.token_logprobs(
        mat_2d, tok_1d, interpret=interpret, strict_vocab=False
    )

  run_2d = _logsoftmax_vjp_coat(
      _run_2d,
      mesh=mesh if (mesh is not None and tp_size > 1) else None,
      in_specs=(P(None, None), P(None)),
      out_spec=P(None),
      differentiable=(True, False),
      exact_residual_dtype=exact_residual_dtype,
  )

  if m <= prod_m:
    mp = _round_up(m, row_align)
    if mp != m:
      padded_logits = jnp.pad(
          flat_logits, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
      )
      padded_tokens = jnp.pad(
          flat_tokens, ((0, mp - m),), constant_values=jnp.int32(0)
      )
    else:
      padded_logits = flat_logits
      padded_tokens = flat_tokens
    out_1d = run_2d(padded_logits, padded_tokens)[:m]
    return out_1d.reshape(orig_token_shape)

  mp = _round_up(m, prod_m)
  if mp != m:
    padded_logits = jnp.pad(
        flat_logits, ((0, mp - m), (0, 0)), constant_values=jnp.float32(0.0)
    )
    padded_tokens = jnp.pad(
        flat_tokens, ((0, mp - m),), constant_values=jnp.int32(0)
    )
  else:
    padded_logits = flat_logits
    padded_tokens = flat_tokens
  logit_chunks = padded_logits.reshape(-1, prod_m, vocab)
  token_chunks = padded_tokens.reshape(-1, prod_m)
  out_chunks = lax.map(
      lambda xs: run_2d(xs[0], xs[1]), (logit_chunks, token_chunks)
  )
  return out_chunks.reshape(mp)[:m].reshape(orig_token_shape)


def get_tp_sharding_config(tp_axis: str = 'tp') -> ShardingConfig:
  """Returns TP-sharded weights and activations with replicated residual stream."""
  return ShardingConfig(
      emb_vd=P(tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      emb_dv=P(None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      q_weight_dnh=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      kv_weight_dnh=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      o_weight_nhd=P(tp_axis, None, None),  # pyrefly: ignore[bad-argument-type]
      ffw_weight_df=P(None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      ffw_weight_fd=P(tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      rms_norm_weight=P(None),  # pyrefly: ignore[bad-argument-type]
      act_btd=P(None, None, None),  # pyrefly: ignore[bad-argument-type]
      act_btf=P(None, None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      act_btnh=P(None, None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
      score_weight_d1=P(None, None),  # pyrefly: ignore[bad-argument-type]
      exp_weight_edf=P(None, None, tp_axis),  # pyrefly: ignore[bad-argument-type]
      exp_weight_efd=P(None, tp_axis, None),  # pyrefly: ignore[bad-argument-type]
  )


def init_non_zero_mlp_weights(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    *,
    seed: int = 42,
) -> None:
  """Initializes synthetic model weights with realistic fan-in scaling.

  Stock `Qwen3` initializes MLP linear kernels to zeros (expecting checkpoint
  restore) and 3D `Einsum` weights with default `glorot_uniform` (which treats
  the leading dimension as a spatial receptive field, yielding `stddev ~ 0.007`
  relative to `embedder` `stddev = 1.0`). Populating balanced fan-in scaled
  weights ensures that every sub-layer (`qkv_proj`, `o_proj`, `mlp_proj`,
  `swiglu`, `lm_head`) contributes at realistic magnitude to the residual
  stream in `bfloat16`.

  Args:
    model: A `Qwen3` or `Qwen3Canon` instance to initialize in-place.
    seed: PRNG seed.
  """
  tp_size, mesh = _active_tp_size_and_mesh('tp')
  shd_cfg = model.config.shd_config

  def _place(arr: jax.Array, spec) -> jax.Array:
    if mesh is None or tp_size <= 1:
      return arr
    pspec = spec if isinstance(spec, P) else P(*spec)
    return jax.device_put(arr, jax.sharding.NamedSharding(mesh, pspec))

  key = jax.random.PRNGKey(seed)
  emb_key, key = jax.random.split(key)
  emb_w = model.embedder.input_embedding.value
  emb_scale = (float(emb_w.shape[1]) ** -0.5) * 0.8
  model.embedder.input_embedding.value = _place(
      (
          jax.random.normal(emb_key, emb_w.shape, dtype=jnp.float32) * emb_scale
      ).astype(emb_w.dtype),
      shd_cfg.emb_vd,
  )

  if hasattr(model, 'lm_head'):
    lm_key, key = jax.random.split(key)
    lm_w = model.lm_head.w.value
    lm_scale = float(lm_w.shape[0]) ** -0.5
    model.lm_head.w.value = _place(
        (
            jax.random.normal(lm_key, lm_w.shape, dtype=jnp.float32) * lm_scale
        ).astype(lm_w.dtype),
        shd_cfg.emb_dv,
    )

  model.final_norm.w.value = _place(model.final_norm.w.value, P(None))

  for layer in model.layers:
    layer.input_layernorm.w.value = _place(
        layer.input_layernorm.w.value, P(None)
    )
    layer.post_attention_layernorm.w.value = _place(
        layer.post_attention_layernorm.w.value, P(None)
    )
    layer.attn.q_norm.w.value = _place(layer.attn.q_norm.w.value, P(None))
    layer.attn.k_norm.w.value = _place(layer.attn.k_norm.w.value, P(None))

    k_q, k_k, k_v, k_o, key = jax.random.split(key, 5)
    for proj_mod, k_proj, fan_in, w_spec in (
        (
            layer.attn.q_proj,
            k_q,
            layer.attn.q_proj.w.value.shape[0],
            shd_cfg.q_weight_dnh,
        ),
        (
            layer.attn.k_proj,
            k_k,
            layer.attn.k_proj.w.value.shape[0],
            shd_cfg.kv_weight_dnh,
        ),
        (
            layer.attn.v_proj,
            k_v,
            layer.attn.v_proj.w.value.shape[0],
            shd_cfg.kv_weight_dnh,
        ),
        (
            layer.attn.o_proj,
            k_o,
            layer.attn.o_proj.w.value.shape[0]
            * layer.attn.o_proj.w.value.shape[1],
            shd_cfg.o_weight_nhd,
        ),
    ):
      w_val = proj_mod.w.value
      scale = (float(fan_in) ** -0.5) * 0.8
      proj_mod.w.value = _place(
          (
              jax.random.normal(k_proj, w_val.shape, dtype=jnp.float32) * scale
          ).astype(w_val.dtype),
          w_spec,
      )

    if hasattr(layer.mlp, 'gate_proj') and isinstance(
        layer.mlp.gate_proj, nnx.Linear
    ):
      k1, k2, k3, key = jax.random.split(key, 4)
      gate_w = layer.mlp.gate_proj.kernel.value
      up_w = layer.mlp.up_proj.kernel.value
      down_w = layer.mlp.down_proj.kernel.value
      scale_in = (float(gate_w.shape[0]) ** -0.5) * 0.8
      scale_down = (float(down_w.shape[0]) ** -0.5) * 0.8
      layer.mlp.gate_proj.kernel.value = _place(
          (
              jax.random.normal(k1, gate_w.shape, dtype=jnp.float32) * scale_in
          ).astype(gate_w.dtype),
          shd_cfg.ffw_weight_df,
      )
      layer.mlp.up_proj.kernel.value = _place(
          (
              jax.random.normal(k2, up_w.shape, dtype=jnp.float32) * scale_in
          ).astype(up_w.dtype),
          shd_cfg.ffw_weight_df,
      )
      layer.mlp.down_proj.kernel.value = _place(
          (
              jax.random.normal(k3, down_w.shape, dtype=jnp.float32)
              * scale_down
          ).astype(down_w.dtype),
          shd_cfg.ffw_weight_fd,
      )


def copy_weights(
    src_model: qwen3_stock.Qwen3 | Qwen3Canon,
    dst_model: qwen3_stock.Qwen3 | Qwen3Canon,
) -> None:
  """Copies all `nnx.Param` weights from `src_model` into `dst_model`."""
  _, src_params = nnx.split(src_model, nnx.Param)
  nnx.update(dst_model, src_params)


def next_token_logprobs(
    logits: jax.Array,  # [B, T, V]
    tokens: jax.Array,  # [B, T]
    *,
    use_canon_logsoftmax: bool,
    exact_residual_dtype: jaxtyping.DTypeLike | None = None,
    last_targets: jax.Array | None = None,  # [B, 1]
) -> tuple[jax.Array, jax.Array]:
  """Scores `tokens[:, 1:]` against rows `[0, T-1)` and `last_targets` against row `T-1`.

  HBM & Latency Optimization (A1): all `B*T` rows are scored in ONE
  `compute_token_logprobs` call instead of slicing `logits[:, :-1, :]` first.
  Every stage of the canonical log-softmax is row-independent, so each row's
  logprob is bitwise the value the sliced call produced, while
  - the `[B, T-1, V]` slice -- a full copy when its consumer is a Pallas
    custom call rather than a fusible XLA op (`4 MB` per device at `M=1024,
    V/TP=2048` bf16, `8 MB` as f32 before A1) -- is gone,
  - `B*T` rows (`256` / `1024` in the bench) already are the kernel's padded
    row count, so the `B*(T-1) -> B*T` row padding copy is gone as well, and
  - the sampler's first generated token is scored by the same call (its logit
    row is the last prompt row), saving a second normalizer launch and its
    collectives.
  The last row is a real row for the kernel either way; without
  `last_targets` it is scored against token 0 and discarded (one row of work,
  zero cotangent in the backward).

  Args:
    logits: `[B, T, V]` logits (any float dtype).
    tokens: `[B, T]` token ids; `tokens[:, t + 1]` is row `t`'s target.
    use_canon_logsoftmax: Whether to use Zero-TIM `canonical_logsoftmax`.
    exact_residual_dtype: See `compute_token_logprobs`.
    last_targets: Optional `[B, 1]` targets for the last row.

  Returns:
    `(next_token_logprobs[B, T-1], last_row_logprobs[B, 1])`, both f32.
  """
  b, t = tokens.shape
  if last_targets is None:
    last_targets = jnp.zeros((b, 1), dtype=jnp.int32)
  targets = jnp.concatenate(
      [tokens[:, 1:].astype(jnp.int32), last_targets.astype(jnp.int32)],
      axis=1,
  )
  if targets.shape != (b, t):
    raise ValueError(f'targets {targets.shape} must match tokens {(b, t)}')
  lps = compute_token_logprobs(
      logits,
      targets,
      use_canon_logsoftmax=use_canon_logsoftmax,
      exact_residual_dtype=exact_residual_dtype,
  )
  return lps[:, :-1], lps[:, -1:]


def _forward_sample(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    prompt_tokens: jax.Array,  # [B, L_prompt]
    *,
    max_new_tokens: int,
    cache_size: int,
    use_canon_logsoftmax: bool,
    return_logits: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
  """KV-cached prefill + greedy decode shared by the two sampler entry points.

  The scan body is the same in both modes; `return_logits` only adds the
  stacked per-step logits as an extra scan output.

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    prompt_tokens: Input prompt tokens `[B, L_prompt]`.
    max_new_tokens: Number of tokens to generate autoregressively.
    cache_size: Total KV cache length (`>= L_prompt + max_new_tokens`).
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax` for scoring.
    return_logits: Also return the per-step logits.

  Returns:
    `(full_tokens, decode_step_logprobs, prefill_prompt_logprobs, step_logits)`
    as documented on `forward_sample_with_logprobs`; `step_logits` is `[B,
    max_new_tokens, V]` in the model's logits dtype when `return_logits` is set
    (row 0 is the last prompt row of the prefill program, the rest are the
    decode steps) and `None` otherwise.
  """
  b, l_prompt = prompt_tokens.shape
  cache = model.init_cache(b, cache_size, model.config.dtype)
  for layer_cache in cache.values():
    layer_cache['k'] = shard(
        layer_cache['k'], model.config.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )
    layer_cache['v'] = shard(
        layer_cache['v'], model.config.shd_config.act_btnh  # pyrefly: ignore[bad-argument-type]
    )

  # 1. Prefill over prompt_tokens [B, L_prompt] with the KV cache
  #    (attention mask [B, L_prompt, cache_size]).
  prompt_pos = jnp.broadcast_to(
      jnp.arange(l_prompt, dtype=jnp.int32)[None, :], (b, l_prompt)
  )
  cache_positions = jnp.arange(cache_size, dtype=jnp.int32)[None, None, :]
  prefill_mask = jnp.logical_and(
      cache_positions <= prompt_pos[:, :, None],
      cache_positions < l_prompt,
  )
  prefill_logits, cache = model(prompt_tokens, prompt_pos, cache, prefill_mask)

  # First generated token comes from the last prompt position's logits.
  first_next_token = jnp.argmax(prefill_logits[:, -1:, :], axis=-1).astype(
      jnp.int32
  )  # [B, 1]
  # One row-complete scoring call: rows `[0, L-1)` against the prompt tokens
  # `[1, L)` and the last prompt row against `first_next_token` (A1).
  prefill_prompt_logprobs, first_logprob = next_token_logprobs(
      prefill_logits,
      prompt_tokens,
      use_canon_logsoftmax=use_canon_logsoftmax,
      exact_residual_dtype=model.config.dtype,
      last_targets=first_next_token,
  )  # [B, L_prompt - 1], [B, 1]
  first_logits = prefill_logits[:, -1:, :]  # [B, 1, V]

  if max_new_tokens == 1:
    full_tokens = jnp.concatenate([prompt_tokens, first_next_token], axis=1)
    return (
        full_tokens,
        first_logprob,
        prefill_prompt_logprobs,
        first_logits if return_logits else None,
    )

  # 2. Autoregressive Decode loop for steps 1 .. max_new_tokens - 1
  def decode_step(carry, step_idx):
    curr_cache, prev_token = carry
    cur_pos = jnp.full((b, 1), l_prompt + step_idx, dtype=jnp.int32)
    step_mask = cache_positions <= cur_pos[:, :, None]
    step_logits, next_cache = model(prev_token, cur_pos, curr_cache, step_mask)
    next_token = jnp.argmax(step_logits, axis=-1).astype(jnp.int32)
    step_logprob = compute_token_logprobs(
        step_logits,
        next_token,
        use_canon_logsoftmax=use_canon_logsoftmax,
        exact_residual_dtype=model.config.dtype,
    )
    outputs = (next_token[:, 0], step_logprob[:, 0])
    if return_logits:
      outputs += (step_logits[:, 0, :],)
    return (next_cache, next_token), outputs

  _, scan_outputs = lax.scan(
      decode_step,
      (cache, first_next_token),
      jnp.arange(max_new_tokens - 1, dtype=jnp.int32),
  )
  rest_tokens, rest_logprobs = scan_outputs[0], scan_outputs[1]
  # rest_tokens / rest_logprobs are [max_new_tokens - 1, B]; transpose them
  # to [B, max_new_tokens - 1].
  gen_tokens = jnp.concatenate([first_next_token, rest_tokens.T], axis=1)
  decode_logprobs = jnp.concatenate([first_logprob, rest_logprobs.T], axis=1)
  full_tokens = jnp.concatenate([prompt_tokens, gen_tokens], axis=1)
  if not return_logits:
    return full_tokens, decode_logprobs, prefill_prompt_logprobs, None
  # [max_new_tokens - 1, B, V] -> [B, max_new_tokens - 1, V].
  rest_logits = jnp.swapaxes(scan_outputs[2], 0, 1)
  step_logits = jnp.concatenate([first_logits, rest_logits], axis=1)
  return full_tokens, decode_logprobs, prefill_prompt_logprobs, step_logits


@functools.partial(
    jax.jit,
    static_argnames=('max_new_tokens', 'cache_size', 'use_canon_logsoftmax'),
)
def forward_sample_with_logprobs(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    prompt_tokens: jax.Array,  # [B, L_prompt]
    *,
    max_new_tokens: int,
    cache_size: int,
    use_canon_logsoftmax: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs KV-cached prefill + greedy decode sampling and returns logprobs.

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    prompt_tokens: Input prompt tokens `[B, L_prompt]`.
    max_new_tokens: Number of tokens to generate autoregressively.
    cache_size: Total KV cache length (`>= L_prompt + max_new_tokens`).
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax` for scoring.

  Returns:
    Tuple of `(full_tokens, decode_step_logprobs, prefill_prompt_logprobs)`:
      - `full_tokens`: `[B, L_prompt + max_new_tokens]` int32 token IDs
      - `decode_step_logprobs`: `[B, max_new_tokens]` float32 logprobs computed
        at each step during KV-cached decoding (Path A: Rollout / Sampler)
      - `prefill_prompt_logprobs`: `[B, L_prompt - 1]` float32 logprobs of
        prompt tokens `[1..L_prompt-1]` computed during prefill
  """
  full_tokens, decode_logprobs, prefill_prompt_logprobs, _ = _forward_sample(
      model,
      prompt_tokens,
      max_new_tokens=max_new_tokens,
      cache_size=cache_size,
      use_canon_logsoftmax=use_canon_logsoftmax,
      return_logits=False,
  )
  return full_tokens, decode_logprobs, prefill_prompt_logprobs


@functools.partial(
    jax.jit,
    static_argnames=('max_new_tokens', 'cache_size', 'use_canon_logsoftmax'),
)
def forward_sample_with_logits(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    prompt_tokens: jax.Array,  # [B, L_prompt]
    *,
    max_new_tokens: int,
    cache_size: int,
    use_canon_logsoftmax: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """`forward_sample_with_logprobs` that also returns the sampler's logits.

  Attribution helper: the prefill and the decode scan are the same program
  body as `forward_sample_with_logprobs` (same model calls, same scoring), with
  the per-step logits stacked as an additional scan output, so a
  decode-vs-prefill logprob drift can be split into "the sampler's logits
  already differ from the prefill's" and "the scoring differs".

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    prompt_tokens: Input prompt tokens `[B, L_prompt]`.
    max_new_tokens: Number of tokens to generate autoregressively.
    cache_size: Total KV cache length (`>= L_prompt + max_new_tokens`).
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax` for scoring.

  Returns:
    The `(full_tokens, decode_step_logprobs, prefill_prompt_logprobs)` of
    `forward_sample_with_logprobs` followed by `step_logits`: `[B,
    max_new_tokens, V]` logits in the model's logits dtype; row 0 is the last
    prompt row of the prefill program (the first generated token's logits),
    rows `1..` are the decode steps.
  """
  full_tokens, decode_logprobs, prefill_prompt_logprobs, step_logits = (
      _forward_sample(
          model,
          prompt_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=use_canon_logsoftmax,
          return_logits=True,
      )
  )
  assert step_logits is not None
  return full_tokens, decode_logprobs, prefill_prompt_logprobs, step_logits


@functools.partial(jax.jit, static_argnames=('use_canon_logsoftmax',))
def score_sequence_prefill(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    full_tokens: jax.Array,  # [B, L_total]
    *,
    use_canon_logsoftmax: bool = False,
) -> jax.Array:
  """Computes next-token logprobs over `full_tokens` via cacheless prefill (Path B)."""
  b, l_total = full_tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )
  logits, _ = model(full_tokens, positions, None, causal_mask)
  # Row-complete scoring (A1): no `logits[:, :-1, :]` slice copy.
  token_lps, _ = next_token_logprobs(
      logits,
      full_tokens,
      use_canon_logsoftmax=use_canon_logsoftmax,
      exact_residual_dtype=model.config.dtype,
  )
  return token_lps


@functools.partial(jax.jit, static_argnames=('use_canon_logsoftmax',))
def score_sequence_fwd_bwd(
    model: qwen3_stock.Qwen3 | Qwen3Canon,
    full_tokens: jax.Array,  # [B, L_total]
    *,
    use_canon_logsoftmax: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Computes next-token logprobs inside a `nnx.value_and_grad` trace (Path C).

  In RL training (GRPO/PPO), the learner computes `old_logprobs` /
  `current_logprobs` inside a forward+backward autodiff compilation unit.
  Without Zero-TIM custom VJPs and canonical kernels, XLA re-fuses the primal
  forward graph differently when paired with a backward pass.

  Args:
    model: `Qwen3` or `Qwen3Canon` model.
    full_tokens: Sequence tokens `[B, L_total]`.
    use_canon_logsoftmax: Whether to use `canonical_logsoftmax`.

  Returns:
    Tuple `(token_logprobs, grad_norm)`:
      - `token_logprobs`: `[B, L_total - 1]` float32 next-token logprobs
      - `grad_norm`: scalar float32 L2 norm of parameter gradients
  """
  b, l_total = full_tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )

  graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

  def loss_fn(param_state):
    m = nnx.merge(graphdef, param_state, other_state)
    logits, _ = m(full_tokens, positions, None, causal_mask)
    # Logits hold exact bf16 values for both models (canon: native bf16 head
    # output, stock: bf16 head output cast to f32), so the bf16 residual of
    # the canonical log-softmax VJP is an exact round trip (see
    # `_logsoftmax_vjp_coat`). Row-complete scoring (A1): no slice copy, and
    # the discarded last row receives a zero cotangent.
    token_lps, _ = next_token_logprobs(
        logits,
        full_tokens,
        use_canon_logsoftmax=use_canon_logsoftmax,
        exact_residual_dtype=model.config.dtype,
    )
    loss = -jnp.mean(token_lps)
    return loss, token_lps

  (_, token_logprobs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
  grad_leaves = jax.tree_util.tree_leaves(grads)
  grad_sq = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in grad_leaves)
  return token_logprobs, jnp.sqrt(grad_sq)
