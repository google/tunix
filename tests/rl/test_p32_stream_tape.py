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
        "replica_check_flags": self.dp_size,
        "post_reduction_replicas_exact": True,
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


def _run(keep_tape, *, deterministic_repeat=False, loss_fn=None, adapter=None):
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
  adapter._p59_rank_parallel_report_adjoint = types.MethodType(  # pylint: disable=protected-access
      lambda self, state, cotangents: tuple(
          value + jnp.asarray(0, value.dtype) for value in cotangents
      ),
      adapter,
  )
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
        gradient_microbatch_sink=None,
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
