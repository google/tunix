"""CANON_P32_KEEP_TAPE=stream: the two-tape window runs the same programs.

Drives ``segmented_dp_grpo_value_and_grad`` on the DP16 stub harness of
canonical_qwen3_adapter_test (forward and reverse groups replaced by recording
stubs; the GRPO loss and its pullback are the real ones) once in batch mode
(``1``) and once in stream mode (``stream``), and checks that the stream
schedule issues forward g+1 before reverse g, never keeps more than two
tapes alive, hands every reverse a bit-identical cotangent, and returns
bit-identical gradients and logprobs.  A row-mixing loss is the negative
control: the end-of-update batch pullback must refuse the streamed
cotangents.  Runs on sixty-four forced CPU devices (a 16x4 data/model mesh).
"""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=64"
  ).strip()

import contextlib  # pylint: disable=g-import-not-at-top
import importlib.util
import io
from pathlib import Path
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import algo_core
from tunix.rl import canonical_qwen3_adapter
from tunix.rl import common
from tunix.rl import dp_training

_HARNESS_PATH = Path(__file__).with_name("canonical_qwen3_adapter_test.py")
_spec = importlib.util.spec_from_file_location("p32_dp16_harness", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

FME = canonical_qwen3_adapter.FunctionalMappingError
GROUPS = 16

_ENV = {
    "CANON_P32_DP16_SEGMENTED": "1",
    "CANON_P32_TRAIN_ADMITTED": "1",
    "CANON_P32_DP_REDUCTION_ADMITTED": "1",
    "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
    "CANON_P33_RUN_STAGE": "full",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_WANDB_ONLINE_REQUIRED": "1",
    "CANON_P31_MONOTONIC_METRICS": "1",
    "CANON_WANDB_PROJECT": "zero-tim-frozenlake-dp16-tp4",
    "CANON_WANDB_GROUP": "qwen3-8b-dp16-tp4",
    "CANON_WANDB_RUN_NAME": "p32-stream-tape-test",
    "WANDB_MODE": "online",
    "WANDB_API_KEY": "test-key-not-a-credential",
    "CANON_P28_SEGMENTED_TRAIN": "1",
    "CANON_P32_WORKLOAD": "frozenlake",
    "CANON_DP_SIZE": "16",
    "CANON_TP_SIZE": "4",
    "CANON_ENGINE_DP_SIZE": "16",
    "CANON_QWEN3_TP_SIZE": "4",
    "CANON_TOTAL_DEVICES": "64",
    "CANON_GLOBAL_PROMPTS": "32",
    "CANON_LOCAL_PROMPTS": "2",
    "CANON_NUM_GENERATIONS": "8",
    "CANON_LOCAL_TRAJECTORIES": "16",
    "CANON_GLOBAL_TRAJECTORIES": "256",
    "CANON_LOGPROB_M": "256",
    "CANON_TARGET_M": "256",
    "MIN_TOKEN_BUCKET": "4096",
    "CANON_FIXED_AR": "1",
    "CANON_FIXED_AR_EMBED": "1",
    "CANON_RPA_VJP2": "1",
    "CANON_VJP2_MAX_SEQS": "1",
    "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
    "CANON_PALLAS_LOGSOFTMAX": "1",
    "CANON_P28_SEGMENTED_FORWARD": "1",
    "CANON_P28_G6_UPDATE": "1",
    "CANON_P29_FULL_TRAIN": "1",
    "CANON_ALIGNMENT_GATE": "1",
    "CANON_ALIGNMENT_GATE_ONLY": "0",
    "CANON_ALIGNMENT_UPDATE_CANARY": "0",
    "CANON_ALIGNMENT_TRAIN": "1",
    "CANON_PRE_ALIGN_GATE": "1",
    "CANON_P33_SHORT_ALIGNMENT": "0",
    "CANON_OPT_STATE_RESIDENT": "0",
    "CANON_P30_OPT_STATE_OFFLOAD": "1",
    "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
    "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
    "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
    "CANON_P30_RELEASE_CAPTURED_STATE": "1",
    "CANON_P30_RESHARD_ACCUMULATOR": "1",
    "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
    "FL_SHARED_MESH": "16,4",
}


def test_selector_parses_stream_fail_closed():
  mode = canonical_qwen3_adapter._p32_keep_tape_mode  # pylint: disable=protected-access
  flag = canonical_qwen3_adapter._p32_keep_tape  # pylint: disable=protected-access
  for value, expected in (("", ""), ("0", ""), ("1", "batch"), ("stream", "stream")):
    with mock.patch.dict(os.environ, {"CANON_P32_KEEP_TAPE": value}, clear=False):
      assert mode() == expected
      assert flag() is bool(expected)
  for junk in ("streaming", "2", "batch", "STREAM"):
    with mock.patch.dict(os.environ, {"CANON_P32_KEEP_TAPE": junk}, clear=False):
      with pytest.raises(FME, match="unset, 0, 1, or stream"):
        mode()


class _FakeReducer:

  def __init__(self, template, *, dp_size, dp_axis, require_distinct_fingerprints):
    del template, require_distinct_fingerprints
    self.dp_size = dp_size
    self.dp_axis = dp_axis
    self.values = []

  def stage_diagnostics(self, staged):
    signatures, finite = jax.vmap(dp_training._gradient_diagnostics)(staged)  # pylint: disable=protected-access
    nonzero = jax.vmap(dp_training._gradient_nonzero_counts)(staged)  # pylint: disable=protected-access
    return signatures, finite, nonzero

  def finalize_staged(self, staged):
    self.values = [
        jax.tree.map(lambda value: value[rank], staged)
        for rank in range(self.dp_size)
    ]
    reduced = dp_training.fixed_dp_sum(self.values)
    fingerprints = tuple(f"rank-{rank}" for rank in range(self.dp_size))
    return reduced, {
        "dp_size": self.dp_size,
        "dp_axis": self.dp_axis,
        "rank_contributions": self.dp_size,
        "rank_local_fingerprints": fingerprints,
        "rank_local_fingerprint_unique_count": self.dp_size,
        "rank_local_fingerprint_duplicate_count": 0,
        "rank_local_fingerprints_distinct": True,
        "reduction_transactions": 1,
        "reduction_rounds": 8,
        "reduction_collectives": 8,
        "replica_check_flags": self.dp_size,
        "post_reduction_replicas_exact": True,
        "shard_map_check_vma": True,
    }


def _train_example():
  prompt_ids = jnp.zeros((256, 4096), jnp.int32).at[:, :2].set(
      jnp.asarray([1, 2], jnp.int32)
  )
  completion_ids = jnp.zeros((256, 2048), jnp.int32).at[:, :3].set(
      jnp.asarray([2, 3, 1], jnp.int32)
  )
  completion_mask = completion_ids != 0
  old = -0.5 * jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (256, 2048)))
  # A real (pytree) TrainExample: the stream cotangent program takes it as a
  # traced argument.
  return common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_ids != 0,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      old_per_token_logps=old.astype(jnp.float32),
      ref_per_token_logps=None,
      advantages=jnp.linspace(-1.0, 1.0, 256, dtype=jnp.float32),
      sampler_is_weights=None,
      segment_ids=None,
  )


_ALGO = types.SimpleNamespace(
    beta=0.0,
    epsilon=0.2,
    epsilon_high=0.2,
    epsilon_c=None,
    loss_algo="grpo",
    loss_agg_mode="sequence-mean-token-mean",
    temperature=1.0,
    kl_loss_mode="k1",
    kl_clamp_value=None,
)


def _run(
    keep_tape,
    *,
    deterministic_repeat=False,
    loss_fn=None,
    adapter=None,
    reduce_once="",
    sink=None,
):
  """Runs one stubbed DP16 update; returns (result, events, live, received)."""
  case = harness.CanonicalQwen3AdapterTest(
      "test_p32_dp16_rejects_the_legacy_data1_segmented_reverse"
  )
  if adapter is None:
    adapter, _ = case._make_p32_group_adapter(sequence_bucket=256)  # pylint: disable=protected-access
  adapter._max_model_len = 6144  # pylint: disable=protected-access
  adapter._p32_d3b_segmented_engine = object()  # pylint: disable=protected-access
  adapter._p59_report_dp_axis = "data"  # pylint: disable=protected-access
  events, live, received = [], [], []

  def forward_group(self, segmented, leaves, spec, *, keep_cache_inputs, keep_tape=False):
    del self, segmented, leaves
    index = sum(1 for kind, _ in events if kind == "fwd")
    reversed_so_far = sum(1 for kind, _ in events if kind == "rev")
    # Tapes alive once this forward exists: the ones issued and not yet
    # reversed, plus this one, plus -- in stream mode -- the tape of the
    # group whose reverse was just dispatched (released only after this
    # lookahead).
    live.append(index - reversed_so_far + 1 + (1 if reversed_so_far else 0))
    events.append(("fwd", index))
    shape = spec["completion_valid"].shape
    key = jax.random.PRNGKey(100 + index)
    return {
        "logps": -jnp.abs(jax.random.normal(key, shape, jnp.float32)),
        "entropy": jnp.abs(
            jax.random.normal(jax.random.fold_in(key, 1), shape, jnp.float32)
        ),
        "counts": {"forward": 1},
        "cache_inputs": ((jnp.ones((2,)),),) if keep_cache_inputs else (),
        "final_caches": (),
        "hidden_inputs": ((jnp.ones((2,)),),) if keep_tape else (),
        "final_hiddens": (jnp.ones((2,)),) if keep_tape else (),
    }

  def reverse_group(self, segmented, leaves, spec, dlogps, dentropy, replay=None):
    del self, segmented, leaves
    index = sum(1 for kind, _ in events if kind == "rev")
    events.append(("rev", index))
    assert replay is not None and replay["hidden_inputs"], "reverse needs a kept tape"
    received.append((np.asarray(dlogps).tobytes(), np.asarray(dentropy).tobytes()))
    signal = jnp.sum(dlogps, axis=1).astype(jnp.float32)[:, None]
    return {
        "engine_gradients": (signal,),
        "initial_cache_cotangents": (jnp.ones((16,), jnp.float32),),
        "counts": {"parallel_reverse": 1},
        "replay_logps": replay["logps"],
        "replay_entropy": replay["entropy"],
    }

  adapter._p32_forward_group = types.MethodType(forward_group, adapter)  # pylint: disable=protected-access
  adapter._p32_reverse_group = types.MethodType(reverse_group, adapter)  # pylint: disable=protected-access
  staged_tables = []

  def report_adjoint(self, state, cotangents):
    del self, state
    staged = tuple(value + jnp.asarray(0, value.dtype) for value in cotangents)
    staged_tables.append(tuple(np.asarray(value).copy() for value in staged))
    return staged

  adapter._p59_rank_parallel_report_adjoint = types.MethodType(  # pylint: disable=protected-access
      report_adjoint, adapter
  )
  adapter._staged_tables_seen = staged_tables
  adapter._p59_reducer_template = types.MethodType(  # pylint: disable=protected-access
      lambda self, state, staged: state, adapter
  )
  adapter._p33_gradient_reducer_factory = _FakeReducer  # pylint: disable=protected-access
  adapter.map_engine_cotangents_to_trainer_state = types.MethodType(
      lambda self, state, cotangents: tuple(cotangents), adapter
  )
  devices = np.asarray(jax.devices()[:64]).reshape(16, 4)
  mesh = jax.sharding.Mesh(devices, ("data", "model"))
  trainer_state = (
      jax.device_put(
          jnp.asarray([1.0]),
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
      ),
  )
  mapped = canonical_qwen3_adapter.FunctionalEngineLeaves(
      paths=(), leaves=(jnp.asarray([1.0]),), source_to_target=()
  )
  env = dict(_ENV)
  env["CANON_P32_KEEP_TAPE"] = keep_tape
  env["CANON_DP_REDUCE_ONCE"] = reduce_once
  env["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
  ).strip()
  stdout = io.StringIO()
  patches = [
      mock.patch.dict(os.environ, env, clear=False),
      mock.patch.object(
          canonical_qwen3_adapter,
          "map_trainer_state_to_engine_leaves",
          return_value=mapped,
      ),
      mock.patch.object(
          canonical_qwen3_adapter,
          "build_p28_segmented_engine_forward",
          return_value=object(),
      ),
      contextlib.redirect_stdout(stdout),
  ]
  if loss_fn is not None:
    patches.append(
        mock.patch.object(algo_core, "grpo_loss_from_precomputed_logps", loss_fn)
    )
  with contextlib.ExitStack() as stack:
    for patch in patches:
      stack.enter_context(patch)
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    result = adapter.segmented_dp_grpo_value_and_grad(
        trainer_state=trainer_state,
        train_example=_train_example(),
        algo_config=_ALGO,
        pad_id=0,
        eos_id=2,
        gradient_microbatch_sink=sink,
        deterministic_repeat=deterministic_repeat,
    )
  result["_adapter"] = adapter
  return result, events, live, received, stdout.getvalue()


def _leaf_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


def test_stream_schedules_a_two_tape_window_and_matches_batch_bitwise():
  batch, batch_events, batch_live, batch_received, batch_out = _run("1")
  stream, stream_events, stream_live, stream_received, stream_out = _run("stream")

  # Batch: every forward, then every reverse; all sixteen tapes alive when
  # the last forward is issued.  Stream: forward g+1 is issued right after
  # reverse g was dispatched and at most two tapes are ever alive.
  assert batch_events == [("fwd", g) for g in range(GROUPS)] + [
      ("rev", g) for g in range(GROUPS)
  ]
  assert max(batch_live) == GROUPS
  expected = [("fwd", 0)]
  for g in range(GROUPS):
    expected.append(("rev", g))
    if g + 1 < GROUPS:
      expected.append(("fwd", g + 1))
  assert stream_events == expected
  assert max(stream_live) == 2

  # Every reverse saw the same cotangent bits, and the update returned the
  # same gradient, logprob and loss bits.
  assert stream_received == batch_received
  assert _leaf_bytes(stream["gradients"]) == _leaf_bytes(batch["gradients"])
  assert _leaf_bytes(stream["per_token_logps"]) == _leaf_bytes(
      batch["per_token_logps"]
  )
  assert _leaf_bytes(stream["token_entropy"]) == _leaf_bytes(
      batch["token_entropy"]
  )
  assert _leaf_bytes(stream["loss"]) == _leaf_bytes(batch["loss"])
  assert len(stream["forward_counts"]) == GROUPS
  assert "[PERF] stage=p32_vag_forward" in batch_out
  assert "[PERF] stage=p32_vag_forward" in stream_out
  assert "forward_group_done" in batch_out and "forward_group_issued" not in batch_out
  assert "forward_group_issued" in stream_out and "forward_group_done" not in stream_out


def test_stream_cotangent_program_compiles_once_across_updates():
  first, *_ = _run("stream")
  adapter = first["_adapter"]
  assert adapter._p32_stream_cotangent_traces == 1  # pylint: disable=protected-access
  second, *_ = _run("stream", adapter=adapter)
  # Same adapter, same algo config, same shapes: no retrace, no recompile,
  # and the second update returns the same gradient bits.
  assert adapter._p32_stream_cotangent_traces == 1  # pylint: disable=protected-access
  assert _leaf_bytes(second["gradients"]) == _leaf_bytes(first["gradients"])


def test_stream_refuses_a_row_mixing_loss_before_returning_a_gradient():
  original = algo_core.grpo_loss_from_precomputed_logps

  def mixing(logps, entropy, train_example, algo_config):
    out = original(logps, entropy, train_example, algo_config)
    coupled = out.primary_loss.unreduced_sum + jnp.square(jnp.sum(logps)) * 1e-6
    return out.replace(primary_loss=out.primary_loss.replace(unreduced_sum=coupled))

  # Batch mode has nothing to compare against and runs; stream mode's
  # end-of-update batch pullback sees cotangents that depend on other rows and
  # refuses.
  _run("1", loss_fn=mixing)
  with pytest.raises(FME, match="stream loss cotangent of group 0 differs"):
    _run("stream", loss_fn=mixing)


def test_keep_tape_refuses_deterministic_repeat():
  with pytest.raises(FME, match="cannot reverse a group twice"):
    _run("stream", deterministic_repeat=True)
  with pytest.raises(FME, match="cannot reverse a group twice"):
    _run("1", deterministic_repeat=True)


def _reference_reduce_once(staged_tables, scale):
  """fixed_dp_sum over ranks of the group-summed staged table, times scale."""
  total = None
  for table in staged_tables:
    table = tuple(jnp.asarray(value) for value in table)
    total = table if total is None else tuple(
        a + b for a, b in zip(total, table, strict=True)
    )
  rows = [
      tuple(value[rank] for value in total) for rank in range(total[0].shape[0])
  ]
  reduced = dp_training.fixed_dp_sum(rows)
  return tuple(value * scale for value in reduced)


def test_reduce_once_reduces_once_and_matches_the_fixed_tree_of_the_group_sum():
  os.environ.pop("CANON_DP_REDUCE_ONCE", None)
  per_group, *_ = _run("stream")
  once, *_ = _run("stream", reduce_once="1")
  adapter = once["_adapter"]
  assert once["dp_reduction_visibility"] == "EXPLICIT_FIXED_TREE_REDUCE_ONCE"
  assert once["dp_reduction_transactions"] == 1
  assert once["dp_staged_accumulations"] == GROUPS
  assert len(once["staged_group_norms"]) == GROUPS
  assert all(norm > 0.0 for norm in once["staged_group_norms"])
  assert all(
      report["gradient_finite"] is True and report["gradient_nonzero"] > 0
      for report in once["reports"]
  )
  # The committed gradient is exactly the fixed DP tree applied to the
  # group-summed staged table (times the loss scale): the same reduce program
  # on a deliberately reassociated sum.
  scale = per_group["loss_output"].primary_loss.compute_scale()
  expected = _reference_reduce_once(adapter._staged_tables_seen, scale)  # pylint: disable=protected-access
  assert _leaf_bytes(once["gradients"]) == _leaf_bytes(expected)
  # Deterministic: a second update produces the same bits.
  again, *_ = _run("stream", reduce_once="1")
  assert _leaf_bytes(again["gradients"]) == _leaf_bytes(once["gradients"])
  # And it is a reassociation of the per-group stream, not a different
  # gradient: the two agree to float32 rounding.
  for left, right in zip(
      jax.tree.leaves(once["gradients"]),
      jax.tree.leaves(per_group["gradients"]),
      strict=True,
  ):
    np.testing.assert_allclose(
        np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-6
    )


def test_reduce_once_g5_capture_rejects_duplicate_rank_signatures():
  # This fixture intentionally has legitimate duplicate rank gradients.  They
  # stay admitted for ordinary training, but the frozen G5 certification arm
  # requires a batch whose rank signatures are all distinct and must reject
  # this one.
  with mock.patch.dict(
      os.environ,
      {"CANON_P61_BACKWARD_NUMERICAL_DIR": "/tmp/g5-negative-control"},
      clear=False,
  ), mock.patch.object(
      canonical_qwen3_adapter.dp_workloads, "validate_environment"
  ):
    with pytest.raises(FME, match="G5 carrier lost a distinct DP rank"):
      _run("stream", reduce_once="1")


def test_reduce_once_streams_one_contribution_standing_for_every_group():
  calls = []

  def sink(index, gradient, multiplier, microbatches=1):
    calls.append((
        index,
        tuple(np.asarray(value).copy() for value in jax.tree.leaves(gradient)),
        float(np.asarray(multiplier)),
        microbatches,
    ))

  result, *_ = _run("stream", reduce_once="1", sink=sink)
  scale = float(np.asarray(result["loss_output"].primary_loss.compute_scale()))
  assert [(call[0], call[3]) for call in calls] == [(0, GROUPS)]
  assert calls[0][2] == scale * GROUPS
  assert result["gradient_microbatches"] == GROUPS
  assert result["gradients"] is None


def test_reduce_once_refuses_diagnostic_modes():
  # Flag-off tape so the reduce-once admission is the first refusal reached.
  with pytest.raises(FME, match="deterministic_repeat are not admitted"):
    _run("", reduce_once="1", deterministic_repeat=True)
  with mock.patch.dict(os.environ, {"CANON_DP_REDUCE_ONCE": "yes"}, clear=False):
    with pytest.raises(ValueError, match="CANON_DP_REDUCE_ONCE must be"):
      dp_training.dp_reduce_once_mode()


def test_staged_accumulator_keeps_the_reducer_exact_layout():
  """The jitted staged add must hand the real reducer its exact shardings.

  The one-host r1 run of CANON_DP_REDUCE_ONCE=1 died here: jit canonicalized
  P('dp', None) to P('dp',) and finalize_staged refused the accumulated table.
  """
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=g-import-not-at-top

  mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("data", "model"))
  template = (
      jax.device_put(jnp.zeros((8,), jnp.float32), NamedSharding(mesh, P("model"))),
      jax.device_put(jnp.zeros((4, 8), jnp.float32), NamedSharding(mesh, P(None, "model"))),
  )
  staged_shardings = (
      NamedSharding(mesh, P("data", "model")),
      NamedSharding(mesh, P("data", None, "model")),
  )

  def table(offset):
    return (
        jax.device_put(
            jnp.stack((jnp.arange(8.0) + offset, jnp.arange(8.0) + 10 + offset)),
            staged_shardings[0],
        ),
        jax.device_put(
            jnp.stack((
                jnp.arange(32.0).reshape(4, 8) + offset,
                jnp.arange(32.0).reshape(4, 8) + 100 + offset,
            )),
            staged_shardings[1],
        ),
    )

  adapter = object.__new__(canonical_qwen3_adapter.Qwen3EngineForwardAdapter)
  reducer = dp_training.FixedDPRankGradientReducer(
      template, dp_size=2, dp_axis="data", require_distinct_fingerprints=False
  )
  # Two updates: the second starts from fresh tables after the first
  # finalize consumed the accumulator (the one-host r2 run died on the
  # second update's first add).
  for update in range(2):
    total = adapter._p59_staged_tree_add(table(1.0), table(2.0))  # pylint: disable=protected-access
    total = adapter._p59_staged_tree_add(total, table(3.0))  # pylint: disable=protected-access
    for leaf, sharding in zip(total, staged_shardings, strict=True):
      assert leaf.sharding == sharding, (update, leaf.sharding, sharding)
    reduced, report = reducer.finalize_staged(total)
    expected = jax.tree.map(
        lambda a, b, c: a + b + c, table(1.0), table(2.0), table(3.0)
    )
    expected_reduced = dp_training.fixed_dp_sum(
        [tuple(value[rank] for value in expected) for rank in range(2)]
    )
    assert _leaf_bytes(reduced) == _leaf_bytes(expected_reduced)
    assert report["reduction_transactions"] == 1
    assert report["post_reduction_replicas_exact"] is True
  # An operand whose sharding is equivalent but spelled differently (the
  # canonical `P('data', 'model')` for the 1-D leaf's `P('data', 'model')`
  # staged spec is already canonical; use a freshly device_put table with
  # the spec re-created) is accepted and re-emitted in the pinned layout.
  respelled = (
      jax.device_put(table(4.0)[0], NamedSharding(mesh, P("data", "model"))),
      jax.device_put(table(4.0)[1], NamedSharding(mesh, P("data", None, "model"))),
  )
  total = adapter._p59_staged_tree_add(table(1.0), respelled)  # pylint: disable=protected-access
  for leaf, sharding in zip(total, staged_shardings, strict=True):
    assert leaf.sharding == sharding
  # The reducer, too, accepts a re-spelled equivalent table (the second
  # update of the one-host r3 run built its reducer from a `P('dp',)`
  # spelling and received the pinned `P('dp', None)` accumulator).
  respelled_reducer = dp_training.FixedDPRankGradientReducer(
      (
          jax.device_put(jnp.zeros((8,), jnp.float32), NamedSharding(mesh, P("model"))),
          jax.device_put(jnp.zeros((4, 8), jnp.float32), NamedSharding(mesh, P(None, "model"))),
      ),
      dp_size=2,
      dp_axis="data",
      require_distinct_fingerprints=False,
  )
  respelled_reducer.finalize_staged((
      jax.device_put(total[0], NamedSharding(mesh, P("data", "model"))),
      jax.device_put(total[1], NamedSharding(mesh, P("data", None, "model"))),
  ))
  # A genuinely different layout is refused.
  wrong = (
      jax.device_put(table(5.0)[0], NamedSharding(mesh, P("model", "data"))),
      table(5.0)[1],
  )
  with pytest.raises(FME, match="not equivalent to the pinned layout"):
    adapter._p59_staged_tree_add(table(1.0), wrong)  # pylint: disable=protected-access
