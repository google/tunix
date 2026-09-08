"""Length-sorted grouping (CANON_P32_LENGTH_SORT=1).

The grouped update reads row ``d * L + g`` as rank ``d``'s row of group
``g`` and a group pays its longest row's chunk count, so rows are sorted by
length and dealt across the ranks group by group.  These tests pin the
permutation rule, the one-program row gather (values, untouched scalars,
shardings), the receipt row mapping, and the property the whole change
rests on: a row's forward outputs do not depend on which rows share its
group.
"""

import importlib.util
import hashlib
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter_module
from tunix.rl import common

_HARNESS_PATH = Path(__file__).with_name("test_p71_fwd_scan.py")
_spec = importlib.util.spec_from_file_location("p71_harness_sort", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

FME = adapter_module.FunctionalMappingError


def test_selector_parses_fail_closed():
  parse = adapter_module._p32_length_sort  # pylint: disable=protected-access
  for value, expected in (("", False), ("0", False), ("1", True)):
    with mock.patch.dict(os.environ, {"CANON_P32_LENGTH_SORT": value}, clear=False):
      assert parse() is expected
  with mock.patch.dict(os.environ, {"CANON_P32_LENGTH_SORT": "yes"}, clear=False):
    with pytest.raises(FME, match="CANON_P32_LENGTH_SORT"):
      parse()


def test_runtime_receipt_pins_geometry_and_permutation():
  perm = np.asarray([3, 1, 2, 0], dtype=np.int32)
  receipt = adapter_module._p32_length_sort_receipt(  # pylint: disable=protected-access
      perm, 2
  )
  digest = hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()
  assert receipt == (
      "[P32.LENGTH_SORT] enabled=1 rows=4 dp=2 groups=2 "
      f"permutation_sha256={digest}"
  )
  with pytest.raises(FME, match="rank-one permutation divisible"):
    adapter_module._p32_length_sort_receipt(  # pylint: disable=protected-access
        perm[:3], 2
    )


def test_phase0_negative_selectors_and_faults_are_discriminating():
  parse = adapter_module._v2_p0_negative_control  # pylint: disable=protected-access
  for value in ("", "length-sort-no-inverse", "reduce-once-reassociate-tail"):
    with mock.patch.dict(
        os.environ, {"V2_P0_NEGATIVE_CONTROL": value}, clear=False
    ):
      assert parse() == value
  with mock.patch.dict(
      os.environ, {"V2_P0_NEGATIVE_CONTROL": "unknown"}, clear=False
  ), pytest.raises(FME, match="V2_P0_NEGATIVE_CONTROL"):
    parse()

  values = tuple(np.float32(value) for value in (1.0e8, 1.0, -1.0e8, 1.0))
  ordinary = None
  held = None
  accumulate = (  # pylint: disable=protected-access
      adapter_module._v2_p0_accumulate_staged
  )
  for index, value in enumerate(values):
    ordinary, held, injected = accumulate(
        ordinary,
        held,
        value,
        index=index,
        group_count=len(values),
        add=lambda left, right: np.float32(left + right),
        mode="",
    )
    assert not injected
  reassociated = None
  held = None
  for index, value in enumerate(values):
    reassociated, held, injected = accumulate(
        reassociated,
        held,
        value,
        index=index,
        group_count=len(values),
        add=lambda left, right: np.float32(left + right),
        mode="reduce-once-reassociate-tail",
    )
  assert injected and held is None
  assert ordinary.tobytes() != reassociated.tobytes()


def test_permutation_deals_sorted_rows_across_ranks():
  lengths = np.asarray([5, 90, 7, 60, 8, 40, 1, 20])  # batch 8, data_size 2 -> 4 groups
  perm = adapter_module._p32_length_sorted_permutation(lengths, 2)  # pylint: disable=protected-access
  assert sorted(perm.tolist()) == list(range(8))
  local = 4
  # Group g holds rows perm[d * local + g] for d in ranks: the two longest
  # rows in group 0, the next two in group 1, ...
  groups = [[int(perm[d * local + g]) for d in range(2)] for g in range(local)]
  assert groups == [[1, 3], [5, 7], [4, 2], [0, 6]]
  group_max = [max(lengths[r] for r in rows) for rows in groups]
  assert group_max == [90, 40, 8, 5]
  with pytest.raises(FME, match="divisible"):
    adapter_module._p32_length_sorted_permutation(lengths[:7], 2)  # pylint: disable=protected-access


def test_gather_rows_permutes_batch_leaves_only_and_keeps_shardings():
  batch = 8
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model"))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  example = common.TrainExample(
      prompt_ids=jax.device_put(jnp.arange(batch * 3, dtype=jnp.int32).reshape(batch, 3), sharding),
      prompt_mask=jax.device_put(jnp.ones((batch, 3), jnp.bool_), sharding),
      completion_ids=jax.device_put(jnp.arange(batch * 2, dtype=jnp.int32).reshape(batch, 2) + 100, sharding),
      completion_mask=jax.device_put(jnp.ones((batch, 2), jnp.bool_), sharding),
      advantages=jax.device_put(jnp.arange(batch, dtype=jnp.float32) * 0.5, sharding),
      ref_per_token_logps=None,
      old_per_token_logps=jax.device_put(-jnp.arange(batch * 2, dtype=jnp.float32).reshape(batch, 2), sharding),
      is_update_step=jnp.asarray(True),
  )
  perm = np.asarray([3, 0, 7, 1, 6, 2, 5, 4])
  out = adapter_module._p32_gather_rows(example, perm)  # pylint: disable=protected-access
  programs = len(adapter_module._P32_ROW_GATHER_PROGRAMS)  # pylint: disable=protected-access
  again = adapter_module._p32_gather_rows(example, perm[::-1].copy())  # pylint: disable=protected-access
  # A second permutation reuses the per-leaf programs.
  assert len(adapter_module._P32_ROW_GATHER_PROGRAMS) == programs  # pylint: disable=protected-access
  for name in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages", "old_per_token_logps"):
    got, want = np.asarray(getattr(out, name)), np.asarray(getattr(example, name))[perm]
    assert got.tobytes() == want.tobytes() and got.dtype == want.dtype
    assert getattr(out, name).sharding.is_equivalent_to(sharding, got.ndim)
  assert out.ref_per_token_logps is None
  assert bool(out.is_update_step) is True and out.is_update_step.ndim == 0
  assert np.asarray(again.advantages).tolist() == np.asarray(example.advantages)[perm[::-1]].tolist()


def test_gather_rows_handles_leaves_on_different_devices():
  """Token arrays on the mesh, loss arrays on one device: both get gathered."""
  if len(jax.devices()) < 4:
    pytest.skip("needs the forced CPU devices")
  batch = 8
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4]).reshape(4, 1), ("data", "model"))
  on_mesh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
  one_device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  example = common.TrainExample(
      prompt_ids=jax.device_put(jnp.arange(batch * 3, dtype=jnp.int32).reshape(batch, 3), on_mesh),
      prompt_mask=jax.device_put(jnp.ones((batch, 3), jnp.bool_), on_mesh),
      completion_ids=jax.device_put(jnp.arange(batch * 2, dtype=jnp.int32).reshape(batch, 2) + 100, on_mesh),
      completion_mask=jax.device_put(jnp.ones((batch, 2), jnp.bool_), on_mesh),
      advantages=jax.device_put(jnp.arange(batch, dtype=jnp.float32) * 0.5, one_device),
      ref_per_token_logps=None,
      old_per_token_logps=jax.device_put(-jnp.arange(batch * 2, dtype=jnp.float32).reshape(batch, 2), one_device),
  )
  perm = np.asarray([6, 1, 4, 3, 2, 5, 0, 7])
  out = adapter_module._p32_gather_rows(example, perm)  # pylint: disable=protected-access
  assert np.asarray(out.prompt_ids).tolist() == np.asarray(example.prompt_ids)[perm].tolist()
  assert np.asarray(out.advantages).tolist() == np.asarray(example.advantages)[perm].tolist()
  assert out.prompt_ids.sharding.is_equivalent_to(on_mesh, 2)
  assert out.advantages.sharding == one_device
  assert out.old_per_token_logps.sharding == one_device
  # Host-made (uncommitted) leaves stay uncommitted, so the mesh programs
  # downstream can place them; a committed single-device copy would be
  # refused ("incompatible devices"), which is how the first carrier run
  # of the sort failed.
  host = common.TrainExample(
      prompt_ids=jnp.asarray(np.arange(batch * 3, dtype=np.int32).reshape(batch, 3)),
      prompt_mask=jnp.asarray(np.ones((batch, 3), bool)),
      completion_ids=jnp.asarray(np.arange(batch * 2, dtype=np.int32).reshape(batch, 2)),
      completion_mask=jnp.asarray(np.ones((batch, 2), bool)),
      advantages=jnp.asarray(np.arange(batch, dtype=np.float32)),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )
  assert not host.prompt_ids.committed
  gathered = adapter_module._p32_gather_rows(host, perm)  # pylint: disable=protected-access
  # NumPy leaves with a batch leading axis are rows as well.
  numpy_tree = {"policy_version": np.arange(batch, dtype=np.int64), "scalar": np.float32(1.5)}
  numpy_out = adapter_module._p32_gather_rows(numpy_tree, perm)  # pylint: disable=protected-access
  assert isinstance(numpy_out["policy_version"], np.ndarray)
  assert numpy_out["policy_version"].tolist() == np.arange(batch)[perm].tolist()
  assert numpy_out["scalar"] == np.float32(1.5)
  assert not gathered.prompt_ids.committed and not gathered.advantages.committed
  assert np.asarray(gathered.prompt_ids).tolist() == np.asarray(host.prompt_ids)[perm].tolist()
  placed = jax.jit(lambda x: x + 1, in_shardings=on_mesh)(gathered.prompt_ids)
  assert placed.sharding.is_equivalent_to(on_mesh, 2)


def test_receipt_rows_map_grouped_positions_back_to_batch_rows():
  adapter, _ = harness._group_adapter(rank_parallel=False)  # pylint: disable=protected-access
  adapter._p32_row_permutation = None  # pylint: disable=protected-access
  assert adapter._p32_receipt_rows((0, 16, 32)) == (0, 16, 32)  # pylint: disable=protected-access
  adapter._p32_row_permutation = np.arange(48)[::-1]  # pylint: disable=protected-access
  assert adapter._p32_receipt_rows((0, 16, 32)) == (47, 31, 15)  # pylint: disable=protected-access


def test_row_forward_does_not_depend_on_its_group_mates():
  """The property the sort rests on: per-row outputs are neighbour-free."""
  adapter, runner = harness._group_adapter(rank_parallel=False)  # pylint: disable=protected-access
  row = jnp.arange(16, dtype=jnp.int32)[:, None]
  probe = 5
  prompt = jnp.concatenate([1 + row % (2 + i) for i in range(3)], axis=1)
  completion = jnp.concatenate([2 + row % (2 + i) for i in range(3)], axis=1)
  prompt = prompt.at[probe].set(jnp.asarray([2, 1, 3]))
  completion = completion.at[probe].set(jnp.asarray([1, 2, 1]))
  with mock.patch.dict(os.environ, dict(harness._SEGMENTED_ENV), clear=False):  # pylint: disable=protected-access
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    engine = adapter_module.build_p28_segmented_engine_forward(runner)
    leaves = tuple(runner.state_leaves)
    # Same probe row, two groups: mates are full 3+3 rows (two chunks) in one
    # and masked down to 2+1 tokens in the other (still two chunks because of
    # the probe row itself, but every neighbour is shorter).
    full = adapter._p32_group_spec(  # pylint: disable=protected-access
        prompt, completion,
        jnp.ones_like(prompt, dtype=bool), jnp.ones_like(completion, dtype=bool), 1.0,
    )
    prompt_mask = jnp.ones_like(prompt, dtype=bool).at[:, 2].set(False).at[probe].set(True)
    completion_mask = jnp.ones_like(completion, dtype=bool).at[:, 1:].set(False).at[probe].set(True)
    mixed = adapter._p32_group_spec(prompt, completion, prompt_mask, completion_mask, 1.0)  # pylint: disable=protected-access
    assert full["num_chunks"] == 2 and mixed["num_chunks"] == 2
    assert int(np.asarray(full["n_real"])[probe]) == 6
    assert int(np.asarray(mixed["n_real"])[probe]) == 6
    assert int(np.asarray(mixed["n_real"])[0]) == 3
    out_full = adapter._p32_forward_group(engine, leaves, full, keep_cache_inputs=False)  # pylint: disable=protected-access
    out_mixed = adapter._p32_forward_group(engine, leaves, mixed, keep_cache_inputs=False)  # pylint: disable=protected-access
    for key in ("logps", "entropy"):
      a = np.asarray(out_full[key])[probe]
      b = np.asarray(out_mixed[key])[probe]
      assert a.tobytes() == b.tobytes(), key


# ---------------------------------------------------------------------------
# Whole update, flag off vs on: per-row outputs come back in the batch's
# order, receipts name the original rows, and the longest rows share group 0.
# ---------------------------------------------------------------------------

_STREAM_PATH = Path(__file__).with_name("test_p32_stream_tape.py")


def _stream_module():
  spec = importlib.util.spec_from_file_location("p32_stream_for_sort", _STREAM_PATH)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _ragged_example():
  """256 rows with distinct real lengths (prompt 2..17, completion 1..16)."""
  rows = 256
  prompt = np.zeros((rows, 4096), np.int32)
  completion = np.zeros((rows, 2048), np.int32)
  for row in range(rows):
    prompt[row, : 2 + row % 16] = 1 + (row % 5)
    completion[row, : 1 + (row * 7) % 16] = 2 + (row % 3)
  prompt = jnp.asarray(prompt)
  completion = jnp.asarray(completion)
  old = -0.5 * jnp.abs(jax.random.normal(jax.random.PRNGKey(11), (rows, 2048)))
  # The learner hands a single TrainExample through as-is, so its loss-side
  # leaves keep whatever type they were built with; the advantages arrive
  # as a NumPy array.  The first carrier run of the sort left NumPy leaves
  # in arrival order and every row got another row's advantage.
  return common.TrainExample(
      prompt_ids=prompt,
      prompt_mask=prompt != 0,
      completion_ids=completion,
      completion_mask=completion != 0,
      old_per_token_logps=old.astype(jnp.float32),
      ref_per_token_logps=None,
      advantages=np.linspace(-1.0, 1.0, rows, dtype=np.float32),
      sampler_is_weights=None,
      segment_ids=None,
  )


def _run_update(stream, length_sort):
  import contextlib  # pylint: disable=g-import-not-at-top
  import io  # pylint: disable=g-import-not-at-top
  import types  # pylint: disable=g-import-not-at-top

  case = stream.harness.CanonicalQwen3AdapterTest(
      "test_p32_dp16_rejects_the_legacy_data1_segmented_reverse"
  )
  adapter, _ = case._make_p32_group_adapter(sequence_bucket=256)  # pylint: disable=protected-access
  adapter._max_model_len = 6144  # pylint: disable=protected-access
  adapter._p32_d3b_segmented_engine = object()  # pylint: disable=protected-access
  adapter._p59_report_dp_axis = "data"  # pylint: disable=protected-access

  def forward_group(self, segmented, leaves, spec, *, keep_cache_inputs, keep_tape=False):
    del self, segmented, leaves, keep_cache_inputs
    # Content-dependent: a row's logps encode its own real length, so a
    # row that came back attached to another row's slot would show.
    valid = spec["completion_valid"]
    lengths = spec["n_real"].astype(jnp.float32)[:, None]
    ramp = 0.001 * jnp.arange(valid.shape[1], dtype=jnp.float32)[None, :]
    logps = jnp.where(valid, -(lengths + ramp), 0.0)
    return {
        "logps": logps,
        "entropy": jnp.where(valid, 0.5 + ramp, 0.0),
        "counts": {"forward": 1},
        "cache_inputs": (),
        "final_caches": (),
        "hidden_inputs": ((jnp.ones((2,)),),) if keep_tape else (),
        "final_hiddens": (jnp.ones((2,)),) if keep_tape else (),
    }

  def reverse_group(self, segmented, leaves, spec, dlogps, dentropy, replay=None):
    del self, segmented, leaves, spec, dentropy
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
  adapter._p33_gradient_reducer_factory = stream._FakeReducer  # pylint: disable=protected-access
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
  mapped = adapter_module.FunctionalEngineLeaves(
      paths=(), leaves=(jnp.asarray([1.0]),), source_to_target=()
  )
  env = dict(stream._ENV)  # pylint: disable=protected-access
  env["CANON_P32_KEEP_TAPE"] = "stream"
  env["CANON_DP_REDUCE_ONCE"] = "1"
  env["CANON_P32_LENGTH_SORT"] = "1" if length_sort else "0"
  env["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
  ).strip()
  with contextlib.ExitStack() as stack:
    stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
    stack.enter_context(mock.patch.object(
        adapter_module, "map_trainer_state_to_engine_leaves", return_value=mapped
    ))
    stack.enter_context(mock.patch.object(
        adapter_module, "build_p28_segmented_engine_forward", return_value=object()
    ))
    output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    result = adapter.segmented_dp_grpo_value_and_grad(
        trainer_state=trainer_state,
        train_example=_ragged_example(),
        algo_config=stream._ALGO,  # pylint: disable=protected-access
        pad_id=0,
        eos_id=2,
        gradient_microbatch_sink=None,
    )
  return adapter, result, output.getvalue()


def test_whole_update_returns_rows_in_batch_order_and_balances_groups():
  if len(jax.devices()) < 64:
    pytest.skip("needs the stream harness's 64 forced CPU devices")
  stream = _stream_module()
  _, plain, plain_output = _run_update(stream, length_sort=False)
  adapter, sorted_run, sorted_output = _run_update(stream, length_sort=True)
  assert "[P32.LENGTH_SORT]" not in plain_output
  assert sorted_output.count("[P32.LENGTH_SORT] enabled=1 ") == 1
  # Per-row outputs: identical bytes in the batch's own row order.
  for key in ("per_token_logps", "token_entropy"):
    assert np.asarray(sorted_run[key]).tobytes() == np.asarray(plain[key]).tobytes(), key
  # The receipts name the original rows; together they cover the batch once.
  rows = [row for report in sorted_run["reports"] for row in report["trajectory_rows"]]
  assert sorted(rows) == list(range(256))
  plain_rows = [row for report in plain["reports"] for row in report["trajectory_rows"]]
  assert plain_rows != rows
  # Group 0 holds the 16 longest rows; each group's spread is one row
  # length at most (there are 256 rows over 16 length values... use n_real).
  example = _ragged_example()
  n_real = np.asarray(example.prompt_mask.sum(1) + example.completion_mask.sum(1))
  first = sorted_run["reports"][0]["trajectory_rows"]
  assert min(n_real[list(first)]) >= max(
      n_real[[r for report in sorted_run["reports"][1:] for r in report["trajectory_rows"]]]
  )
  perm = adapter._p32_row_permutation  # pylint: disable=protected-access
  assert perm is not None and sorted(perm.tolist()) == list(range(256))
  # The gradient is the same multiset of per-row contributions, so the
  # sorted update's gradient differs from the plain one only by summation
  # order (the fake reverse sums each row's cotangent, the fake reducer
  # sums over ranks and groups).
  plain_grad = np.asarray(jax.tree.leaves(plain["gradients"])[0], np.float64)
  sorted_grad = np.asarray(jax.tree.leaves(sorted_run["gradients"])[0], np.float64)
  assert plain_grad.shape == sorted_grad.shape
  assert np.allclose(plain_grad, sorted_grad, rtol=1e-5, atol=1e-6), (plain_grad, sorted_grad)
  assert not np.array_equal(plain_grad, sorted_grad) or plain_grad.size == 1
