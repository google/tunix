# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The rollout/trainer contract for MoE router replay.

The sampler records which experts each token was routed to; the trainer replays
those choices instead of running its own router. Both sides have to agree on
one question: what does a slot that carries no expert id mean?

There are two answers, and conflating them is a wrong-answer bug:

  ============  =========================================================
  MISSING_ROUTE A real token sits here, but no route was captured for it.
  (-1)          The trainer runs its own router. Zeroing the MoE output
                instead would silently delete the block for a live token.
  PADDING_ROUTE No token sits here at all -- left/right padding, or unused
  (-2)          packing capacity. The trainer must not route it: doing so
                pollutes load-balance statistics and burns expert capacity
                that real tokens need.
  ============  =========================================================

Short enough to keep in your head: **-2 means there is no token; -1 means there
is a token but no route.** When in doubt use -1. It is always safe, only
wasteful -- see the corollary to Theorem S in the design doc.

Rows are atomic. A top-k row is either a complete real selection, all -1, or
all -2. A half-filled row has no meaning: the trainer cannot dispatch a token
to "two and a half experts", so it falls back for the whole row.
"""

from absl import logging
import numpy as np

# See the module docstring. These are Tunix's invention -- the sampler emits no
# sentinels of its own, and `0` is a real expert id there.
MISSING_ROUTE = -1
PADDING_ROUTE = -2

# Routing is the largest tensor this feature adds: `[tokens, layers, top_k]`
# per sequence, carried host->device every step. For a 40-layer top-8 model
# that is 1.25 KB/token at int32, so the width is worth caring about.
#
# int16 spans [-32768, 32767], which holds both sentinels and any expert id we
# will plausibly see (the largest deployed MoE routers are in the low
# thousands). `assert_expert_range` enforces the bound rather than trusting it.
ROUTE_DTYPE = np.int16
_MAX_EXPERT_ID = np.iinfo(ROUTE_DTYPE).max


def assert_expert_range(num_experts: int) -> None:
  """Fails if `num_experts` cannot be represented in `ROUTE_DTYPE`.

  Silent overflow here would wrap real expert ids onto other experts, or onto
  the sentinels, which reads as "no route" -- a wrong-answer bug with no crash.
  """
  if num_experts - 1 > _MAX_EXPERT_ID:
    raise ValueError(
        f"num_experts={num_experts} exceeds what {ROUTE_DTYPE.__name__} can"
        f" hold (max id {_MAX_EXPERT_ID}). Widen ROUTE_DTYPE."
    )


def full_sequence(
    prompt_routes: np.ndarray | None,
    completion_routes: np.ndarray | None,
    prompt_len: int,
    completion_len: int,
) -> np.ndarray | None:
  """Assembles one generation's routes in input-token order.

  A forward pass records the token it *consumed*, so a `P`-token prompt plus `G`
  generated tokens yields `P + G - 1` routed positions: the last generated token
  is never fed back in and so has no route of its own. That final row stays
  `MISSING_ROUTE`, which is correct -- its logits are not used by the loss, and a
  native route there is harmless.

  Prompt routes matter even though prompt positions carry no loss. Their MoE
  output feeds the keys and values every completion position attends to, so
  replaying the completion while re-routing the prompt reproduces neither
  policy.

  WHAT THE SAMPLER ACTUALLY HANDS US. vLLM accumulates one chunk per engine
  step and concatenates them when the request finishes:

      # vllm/v1/engine/output_processor.py
      # Routed experts accumulation (prompt + sample chunks)
      routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)
      # vllm/outputs.py
      routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]

  The prefill chunk is in that list, so `CompletionOutput.routed_experts` spans
  the *whole sequence*, and `RequestOutput` has no separate prompt array at all.
  `SamplingParams.routed_experts_prompt_start` can drop a leading slice of the
  prompt, and a prefix-cache hit skips the cached span -- so whatever is absent
  is absent from the FRONT.

  Combined with the trailing gap above, the capture is a contiguous span that
  ENDS at position `P + G - 1` (exclusive) and starts `n` rows before it:

      start = max(0, (P + G - 1) - n)

  One rule, no branching, and it is forced by the data rather than guessed.
  A healthy full capture is `P + G - 1` rows and lands at `start = 0`; a
  decode-only capture is `G - 1` rows and lands at `start = P`; a prefix-cache
  hit of `C` tokens lands at `start = C`.

  Both edge alignments are wrong on real data, which is worth stating because
  each looks plausible in isolation. For the observed failure -- `P = 118`,
  `G = 512`, `n = 629` -- left-aligning at `P` overflows the array, and
  right-aligning at `P + G` puts the span at `[1, 630)`, shifting every route
  one token late. A one-token shift is not a crash; it is a silently wrong
  policy.

  Args:
    prompt_routes: `[captured_prompt_len, num_layers, top_k]`, or None. No vLLM
      build we ship against populates this; it is accepted for backends that
      report prefill separately.
    completion_routes: `[n, num_layers, top_k]`, or None. See the table.
    prompt_len: Number of prompt tokens actually fed to the sampler.
    completion_len: Number of tokens generated.

  Returns:
    `[prompt_len + completion_len, num_layers, top_k]` in `ROUTE_DTYPE`, or
    None if the sampler captured nothing at all.

  Raises:
    ValueError: If a captured tensor cannot be placed unambiguously. Guessing
      an alignment here would silently attach experts to the wrong tokens.
  """
  prompt_routes = _as_routes(prompt_routes, "prompt_routed_experts")
  completion_routes = _as_routes(completion_routes, "routed_experts")
  if prompt_routes is None and completion_routes is None:
    return None

  trailing = _agreeing_trailing_shape(prompt_routes, completion_routes)
  total = prompt_len + completion_len
  full = np.full((total,) + trailing, MISSING_ROUTE, ROUTE_DTYPE)

  # Lowest index we hold a real route for. Everything below it stays
  # MISSING_ROUTE and gets re-routed by the trainer.
  covered_from = total

  if prompt_routes is not None:
    if prompt_routes.shape[0] > prompt_len:
      raise ValueError(
          f"sampler captured {prompt_routes.shape[0]} prompt routes for only"
          f" {prompt_len} prompt tokens; the capture does not belong to this"
          " request."
      )
    # A cache hit is always a *prefix*, so a short capture is unambiguously the
    # prompt's tail. Align right.
    full[prompt_len - prompt_routes.shape[0] : prompt_len] = prompt_routes
    covered_from = prompt_len - prompt_routes.shape[0]

  if completion_routes is not None:
    n = completion_routes.shape[0]
    if n > total:
      raise ValueError(
          f"sampler captured {n} routes for a sequence of only {total} tokens"
          f" ({prompt_len} prompt + {completion_len} generated); the capture"
          " does not belong to this request."
      )
    # ANCHOR AT THE END, NOT AT EITHER EDGE.
    #
    # Routes exist for exactly the positions that were fed into a forward.
    # Prefill consumes 0..P-1 and emits token P; decode step i consumes P+i-1
    # and emits P+i. So the last *consumed* position is total-2, and the final
    # generated token has no route of its own -- which is why a healthy capture
    # is P+G-1 rows, not P+G. Whatever is absent is absent from the FRONT: a
    # prefix-cache hit skips a prompt prefix, and
    # SamplingParams.routed_experts_prompt_start drops one deliberately.
    #
    # Hence the span ends at total-1 and starts n rows before it. Both naive
    # alternatives are off by one on the real data: left-aligning at prompt_len
    # overflows, and right-aligning at total shifts every route one token late.
    start = max(0, total - 1 - n)
    full[start : start + n] = completion_routes
    covered_from = min(covered_from, start)

  if covered_from > 0:
    # Not fatal -- the trainer re-routes those positions itself -- but the
    # resulting policy is neither the rollout's nor the trainer's, because
    # prompt MoE output feeds the keys and values every completion position
    # attends to. Anyone paying for replay needs to know they only got part.
    logging.log_first_n(
        logging.WARNING,
        "Sampler routes cover positions [%d, %d) of a %d-token sequence"
        " (%d prompt + %d generated). The trainer re-routes the first %d"
        " prompt positions with its own router. Expected when prefix caching"
        " is on or routed_experts_prompt_start > 0; otherwise the rollout"
        " backend is not capturing prefill routes.",
        1,
        covered_from,
        total,
        total,
        prompt_len,
        completion_len,
        covered_from,
    )

  return full


def require_maxtext_support() -> None:
  """Fails now if the installed MaxText would misread our sentinels.

  MaxText learned the two-sentinel protocol late. Before that, `get_topk` did
  `top_k_indices = forced_routed_experts` unconditionally and masked out every
  id outside `[0, num_experts)` as an unused slot. Against that version
  `MISSING_ROUTE` does not mean "re-route this token", it means "route this
  token nowhere": the MoE block is zeroed for a live token and the run trains a
  quietly mutilated model. Nothing downstream notices -- the loss stays finite
  and gradients keep flowing -- so this has to be checked up front.

  Raises:
    RuntimeError: If MaxText is absent, predates the protocol, or disagrees
      with us about the sentinel values.
  """
  try:
    from maxtext.layers import moe  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise RuntimeError(
        "router replay requires MaxText as the trainer, but importing"
        f" maxtext.layers.moe failed: {e}"
    ) from e

  for name, ours in (
      ("ROUTER_REPLAY_MISSING", MISSING_ROUTE),
      ("ROUTER_REPLAY_PADDING", PADDING_ROUTE),
  ):
    theirs = getattr(moe, name, None)
    if theirs is None:
      raise RuntimeError(
          f"the installed MaxText defines no {name}, so it predates router"
          f" replay's two-sentinel protocol and would read {MISSING_ROUTE} as"
          " 'route this token nowhere', silently zeroing the MoE block for"
          " real tokens. Upgrade MaxText, or leave return_routed_experts off."
      )
    if theirs != ours:
      raise RuntimeError(
          f"MaxText's {name} is {theirs} but tunix emits {ours}; the two sides"
          " disagree on the router-replay wire format."
      )


def validate(
    routed_experts: np.ndarray, num_experts: int | None = None
) -> None:
  """Checks that every top-k row is atomic and internally consistent.

  This scans every value, so it is `O(tokens x layers x top_k)` and is meant for
  tests and for debugging a new producer -- not for every rollout step. The
  trainer falls back atomically on malformed rows, so this is a diagnostic
  rather than the thing that keeps training safe.

  Args:
    routed_experts: Any array whose last axis is the top-k axis.
    num_experts: Upper bound to check expert ids against, if known.

  Raises:
    ValueError: On a non-integer dtype, a missing top-k axis, a row that mixes
      sentinels with real ids, a row that repeats an expert, or an id outside
      `[0, num_experts)`.
  """
  routed = np.asarray(routed_experts)
  if not np.issubdtype(routed.dtype, np.integer):
    raise ValueError(
        f"routed_experts must hold integer expert ids; got {routed.dtype}."
    )
  if routed.ndim < 2 or routed.shape[-1] == 0:
    raise ValueError(
        "routed_experts needs a non-empty top-k axis; got shape"
        f" {routed.shape}."
    )

  rows = routed.reshape(-1, routed.shape[-1])
  first_elements = rows[..., 0]
  is_missing = first_elements == MISSING_ROUTE
  is_padding = first_elements == PADDING_ROUTE
  is_real = first_elements >= 0

  # Rows are atomic, so the first element decides what the rest must be.
  valid = (
      (is_missing & np.all(rows == MISSING_ROUTE, axis=-1))
      | (is_padding & np.all(rows == PADDING_ROUTE, axis=-1))
      | (is_real & np.all(rows >= 0, axis=-1))
  )
  if not valid.all():
    raise ValueError(
        "every routed_experts row must be all real, all -1, or all -2; got"
        f" {rows[np.argmin(valid)].tolist()}."
    )

  real_rows = rows[is_real]
  if real_rows.size and real_rows.shape[-1] > 1:
    # A repeated id dispatches one token to the same expert twice, which
    # double-counts its contribution rather than failing. Sorting is N log N,
    # which is nothing when top_k is 2 to 8.
    repeats = np.any(
        np.diff(np.sort(real_rows, axis=-1), axis=-1) == 0, axis=-1
    )
    if repeats.any():
      raise ValueError(
          "routed_experts row repeats an expert:"
          f" {real_rows[np.argmax(repeats)].tolist()}."
      )
  if num_experts is not None:
    # Catch the representability problem too: an expert count that overflows
    # ROUTE_DTYPE wraps real ids onto other experts, or onto the sentinels,
    # where they read as "no route" -- wrong answers with no crash.
    assert_expert_range(num_experts)
    if real_rows.size and real_rows.max() >= num_experts:
      raise ValueError(
          f"expert id {real_rows.max()} is outside [0, {num_experts})."
      )


def _as_routes(value, name: str) -> np.ndarray | None:
  """Normalizes one captured tensor to `[tokens, num_layers, top_k]`.

  Screens the producer's rows and demotes any that are provably corrupt to
  `MISSING_ROUTE`, so the trainer re-routes those tokens with its own router
  instead of dispatching them to experts the sampler never chose. See
  `_screen_producer_values`.
  """
  if value is None:
    return None
  raw = np.asarray(value)
  if not np.issubdtype(raw.dtype, np.integer):
    raise ValueError(
        f"{name} must hold integer expert ids; got {raw.dtype}."
    )
  corrupt = _screen_producer_values(raw, name)
  routes = raw.astype(ROUTE_DTYPE)
  if routes.ndim != 3:
    raise ValueError(
        f"{name} must be [tokens, num_layers, top_k]; got shape {routes.shape}."
    )
  if corrupt is not None and corrupt.any():
    # Demote AFTER widening: the sentinel is negative and cannot be stored in
    # the producer's unsigned buffer. Whole rows only -- routing is atomic, so
    # a partially-trusted row has no meaning (see the module docstring).
    flat = routes.reshape(-1, routes.shape[-1])
    flat[corrupt] = MISSING_ROUTE
    routes = flat.reshape(routes.shape)
  return routes



def _screen_producer_values(raw: np.ndarray, name: str) -> np.ndarray | None:
  """Finds capture rows that cannot have come from a top-k.

  WHY A VALUE CHECK CANNOT WORK HERE. The obvious screen -- reject ids outside
  `[0, num_experts)` -- is useless for this model. Qwen3.5-35B-A3B has
  `num_experts: 256` and the sampler's arena is `uint8`
  (`RoutedExpertsManager CPU buffer: slots=643840, layers=40, top_k=8,
  dtype=uint8`). A uint8 spans exactly 0..255, so the expert space saturates the
  dtype with ZERO headroom: every one of the 256 byte values is a legal expert
  id. There is no spare encoding for "nothing was written here", and no value a
  corrupt slot could hold that a healthy slot could not.

  So an unwritten or stale slot does not fail loudly. A zero-initialized arena
  hands us expert `0`, which is real, in range, and silently wrong -- the token
  is dispatched to a plausible expert that the sampler never chose. That
  produces a finite loss, no warning, and a small-but-real logprob error, which
  is the failure signature we are chasing.

  WHAT STILL WORKS IS STRUCTURAL. `jax.lax.top_k` selects k DISTINCT experts, so
  a healthy row can never repeat an id. A constant fill does nothing but repeat:
  an unwritten row is `[0]*k`, which is k-fold duplicated and therefore provably
  not a top-k output. That check is dtype-agnostic, needs no knowledge of
  `num_experts`, and cannot false-positive on healthy data.

  Cost is a sort over the length-8 top-k axis, a few milliseconds per sequence,
  which is why this runs every step while `validate_routes` (full atomicity plus
  sentinel-consistency checks) stays a debugging tool.

  Args:
    raw: The producer's array `[tokens, num_layers, top_k]`, before widening.
    name: Field name, for the log message.

  Returns:
    A boolean mask over the flattened `[tokens * num_layers]` rows, True where
    the row is provably not top-k output, or None when nothing was checked.
    The caller demotes those rows to `MISSING_ROUTE`.
  """
  if not raw.size or raw.ndim < 2 or raw.shape[-1] < 2:
    return None

  lo = int(raw.min())
  hi = int(raw.max())
  rows = raw.reshape(-1, raw.shape[-1])

  # A repeat is impossible from top_k, so this counts provably-corrupt rows.
  repeated = np.any(np.diff(np.sort(rows, axis=-1), axis=-1) == 0, axis=-1)
  n_repeated = int(np.count_nonzero(repeated))
  # The constant-row subset is the fill-value signature specifically.
  n_constant = int(np.count_nonzero(rows.min(axis=-1) == rows.max(axis=-1)))

  # Unconditional for the first few calls: this is the measurement that says
  # whether the capture is clean, and it costs nothing to emit.
  logging.log_first_n(
      logging.INFO,
      "router_replay capture %s: dtype=%s shape=%s min=%d max=%d"
      " rows=%d repeated_rows=%d constant_rows=%d",
      3,
      name,
      raw.dtype,
      raw.shape,
      lo,
      hi,
      rows.shape[0],
      n_repeated,
      n_constant,
  )

  if n_repeated:
    logging.log_first_n(
        logging.ERROR,
        "router_replay capture %s: %d of %d rows (%.4f%%) repeat an expert id"
        " and so cannot be top-k output; %d of them are constant, the"
        " signature of an unwritten or stale slot. dtype=%s min=%d max=%d."
        " Example row: %s. These ids are IN RANGE, so nothing downstream would"
        " have rejected them; they are being demoted to MISSING_ROUTE so the"
        " trainer re-routes those tokens instead of dispatching them to experts"
        " the sampler never chose.",
        5,
        name,
        n_repeated,
        rows.shape[0],
        100.0 * n_repeated / rows.shape[0],
        n_constant,
        raw.dtype,
        lo,
        hi,
        rows[int(np.argmax(repeated))].tolist(),
    )

  return repeated



def _agreeing_trailing_shape(
    prompt_routes, completion_routes
) -> tuple[int, ...]:
  """Returns the shared `(num_layers, top_k)`, rejecting a mismatch."""
  if prompt_routes is None:
    return completion_routes.shape[1:]
  if completion_routes is None:
    return prompt_routes.shape[1:]
  if prompt_routes.shape[1:] != completion_routes.shape[1:]:
    raise ValueError(
        "prompt and completion routes disagree on (num_layers, top_k):"
        f" {prompt_routes.shape[1:]} vs {completion_routes.shape[1:]}."
    )
  return prompt_routes.shape[1:]
