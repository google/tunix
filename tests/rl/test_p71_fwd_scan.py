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

"""P71-E1 regression suite: the grouped-reverse forward tape scan.

CL 5017c279 "Build the grouped reverse forward tape with one scan program
per chunk" added ``CANON_P71_SCAN=fwd`` (default off): the grouped reverse
pass rebuilds each chunk's forward tape with ONE ``lax.scan`` program
(``run_layers_fwd_tape_scan``, module ``zt_tr_fwd_scan``; carry =
cache/hidden, ys = the stacked tape) instead of one jitted fwd_layer
program per layer. The serial branch consumes the stacked tape through the
existing static-slice tape pullbacks; the rank-parallel branch keeps its
mapped per-layer pullbacks on per-layer operands (hidden tape unstacked by
one jitted program). ``bwd``/``full`` are reserved-fatal E2/E3 rungs.

This suite pins the CL's claims: the scan-built stacked tape bitwise
equals the per-layer-built stack (layer counts {2, 4}, fp32 and bf16
stacks, -0.0 and subnormal payloads), full reverse-group gradients are
bitwise identical legacy-vs-fwd on both the serial branch and a forced
16-device rank-parallel mesh, the flag ladder parses exactly (off
synonyms, fwd, reserved-fatal bwd/full, fail-closed diagnostics
conflicts), and the flag-off behavior matches a digest frozen from the
landed HEAD.

Rebuilt 2026-08-27 after the original scratch suite was lost; assertion
inventory reconstructed from
tasks/v1_hp_zero_tim/phases/v1-p71-scan-fusion.md (E1 result log:
"CPU 17/17 bitwise parity").
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16"
  ).strip()

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter

FunctionalMappingError = canonical_qwen3_adapter.FunctionalMappingError

_SEGMENTED_ENV = {
    "CANON_P28_SEGMENTED_FORWARD": "1",
    "CANON_P28_SEGMENTED_TRAIN": "1",
    "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
}

_FIXTURES_NAME = "p70_adapter_test_fixtures"

# sha256 over the flag-off serial reverse group's engine gradients and
# replay logps on the fixed fixture below, frozen from the landed HEAD
# (de48b9b4) inside the pinned CPU image
# (tunix_frozenlake_image:vllm-tpu0.25.0, JAX_PLATFORMS=cpu). Any bitwise
# drift of the default (legacy, pristine) reverse path trips this before
# a parity pair could mask it. Regenerate — only after a deliberately
# admitted numerical change — by running this file with
# P71_PRINT_DIGEST=1 and copying the printed value.
FLAG_OFF_DIGEST = (
    "25bcd0c9db3eafb8137f4e05545a6c357a930945f46172720972e26b8c55ee54"
)


def _load_module_by_path(path, name):
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _fixtures():
  path = (
      Path(canonical_qwen3_adapter.__file__).resolve().parents[2]
      / "tests"
      / "rl"
      / "canonical_qwen3_adapter_test.py"
  )
  if not path.exists():
    path = Path(__file__).resolve().with_name(
        "canonical_qwen3_adapter_test.py"
    )
  return _load_module_by_path(path, _FIXTURES_NAME)


def _tree_bytes(tree):
  return tuple(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(tree))


class _CountingWrapper:

  def __init__(self, wrapped):
    self.wrapped = wrapped
    self.calls = 0

  def __call__(self, *args, **kwargs):
    self.calls += 1
    return self.wrapped(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tiny uniform layer stacks with signed-zero / subnormal payloads
# ---------------------------------------------------------------------------


class _PayloadEmbed(nnx.Module):
  """Embed table that plants -0.0 and subnormal bits into the tape."""

  def __init__(self, dtype):
    table = np.asarray(
        [[0.5], [-0.0], [1e-42], [-2.5], [0.25]], np.float32
    )
    if dtype == ml_dtypes.bfloat16:
      table = np.asarray(
          [[0.5], [-0.0], [1e-40], [-2.5], [0.25]], np.float32
      )
    self.weight = nnx.Param(jnp.asarray(table.astype(dtype)))

  def __call__(self, token_ids):
    return self.weight[token_ids]


class _PayloadLayer(nnx.Module):

  def __init__(self, scale, dtype):
    self.scale = nnx.Param(jnp.asarray(scale, dtype))

  def __call__(self, cache, hidden, metadata):
    dtype = hidden.dtype
    output = (
        hidden * self.scale[...]
        + metadata.astype(dtype)
        + cache * jnp.asarray(0.125, dtype)
    )
    return cache + output, output


class _PayloadNorm(nnx.Module):

  def __init__(self, dtype):
    self.scale = nnx.Param(jnp.asarray(0.25, dtype))

  def __call__(self, hidden):
    return hidden * self.scale[...]


class _PayloadHead(nnx.Module):

  def __init__(self, dtype):
    self.scale = nnx.Param(jnp.asarray(0.75, dtype))

  def __call__(self, hidden):
    return hidden * self.scale[...]


def _payload_runner(num_layers, dtype):
  backbone = types.SimpleNamespace()

  class _Backbone(nnx.Module):

    def __init__(self):
      self.embed_tokens = _PayloadEmbed(dtype)
      self.layers = nnx.List(
          [
              _PayloadLayer(1.25 + 0.5 * index, dtype)
              for index in range(num_layers)
          ]
      )
      self.start_layer = 0
      self.end_layer = num_layers
      self.norm = _PayloadNorm(dtype)

  class _Model(nnx.Module):

    def __init__(self):
      self.model = _Backbone()
      self.lm_head = _PayloadHead(dtype)

  runner = types.SimpleNamespace()
  runner.model = _Model()
  _, runner.state = nnx.split(runner.model)
  runner.state_leaves = tuple(jax.tree.leaves(runner.state))
  runner.kv_caches = [
      jnp.asarray(0.0, dtype) for _ in range(num_layers)
  ]
  runner.is_first_rank = True
  runner.is_last_rank = True
  runner.mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")
  )
  del backbone
  return runner


def _payload_engine(num_layers, dtype):
  runner = _payload_runner(num_layers, dtype)
  with mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False):
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
  return engine, runner


def _per_layer_reference_tape(engine, leaves, caches, hidden, metadata):
  """The legacy per-layer tape build the scan replaces, stacked."""
  hidden_ins = []
  for layer_index, cache in enumerate(caches):
    hidden_ins.append(hidden)
    _, hidden = engine.run_layer_forward(
        layer_index, leaves, cache, hidden, metadata
    )
  stacked_caches = jax.tree.map(lambda *xs: jnp.stack(xs), *caches)
  stacked_hidden = jnp.stack(hidden_ins)
  return stacked_caches, stacked_hidden, hidden


# ---------------------------------------------------------------------------
# Flag ladder (the CURRENT landed parser)
# ---------------------------------------------------------------------------


def test_flag_ladder_off_synonyms_fwd_and_reserved_fatal():
  parse = canonical_qwen3_adapter._p71_scan_mode  # pylint: disable=protected-access
  for off_value in (None, "", "0", "off"):
    env = {} if off_value is None else {"CANON_P71_SCAN": off_value}
    with mock.patch.dict(os.environ, env, clear=False):
      if off_value is None:
        os.environ.pop("CANON_P71_SCAN", None)
      assert parse() == ""
  with mock.patch.dict(
      os.environ, {"CANON_P71_SCAN": "fwd"}, clear=False
  ):
    assert parse() == "fwd"
  # E2-prime landed: bwd is a valid ladder value (unrolled blocks);
  # full remains reserved.
  with mock.patch.dict(
      os.environ, {"CANON_P71_SCAN": "bwd"}, clear=False
  ):
    assert parse() == "bwd"
  for reserved in ("full",):
    with mock.patch.dict(
        os.environ, {"CANON_P71_SCAN": reserved}, clear=False
    ):
      with pytest.raises(
          FunctionalMappingError, match="reserved"
      ):
        parse()
  with mock.patch.dict(
      os.environ, {"CANON_P71_SCAN": "scan"}, clear=False
  ):
    with pytest.raises(
        FunctionalMappingError, match="must be unset/0/off/fwd"
    ):
      parse()


# ---------------------------------------------------------------------------
# Scan-built stacked tape bitwise equals the per-layer-built stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_layers,dtype",
    [
        (2, jnp.float32),
        (4, jnp.float32),
        (2, ml_dtypes.bfloat16),
        (4, ml_dtypes.bfloat16),
    ],
    ids=["L2-f32", "L4-f32", "L2-bf16", "L4-bf16"],
)
def test_fwd_tape_scan_bitwise_equals_per_layer_stack(num_layers, dtype):
  engine, runner = _payload_engine(num_layers, dtype)
  leaves = tuple(runner.state_leaves)
  # Token 1 plants -0.0, token 2 plants a subnormal into the tape's first
  # hidden row; caches carry a -0.0 payload too.
  token_ids = jnp.asarray([1, 2, 0, 3, 4, 0], jnp.int32)
  hidden = engine.run_embed_forward(token_ids, state_leaves=leaves)
  planted = np.asarray(hidden).reshape(-1)
  assert planted[0] == 0.0 and np.signbit(planted[0])  # -0.0 present
  metadata = jnp.asarray(0.125, jnp.float32)
  caches = [
      jnp.full((), -0.0, dtype) if index == 0 else jnp.asarray(0.0, dtype)
      for index in range(num_layers)
  ]
  scan_caches, scan_hidden, scan_out = engine.run_layers_fwd_tape_scan(
      leaves, caches, hidden, metadata
  )
  ref_caches, ref_hidden, ref_out = _per_layer_reference_tape(
      engine, leaves, caches, hidden, metadata
  )
  assert _tree_bytes(scan_caches) == _tree_bytes(ref_caches)
  assert _tree_bytes(scan_hidden) == _tree_bytes(ref_hidden)
  assert _tree_bytes(scan_out) == _tree_bytes(ref_out)
  assert jax.tree.leaves(scan_hidden)[0].dtype == dtype
  # The planted signed-zero and subnormal bits survived into the tape.
  tape_row0 = np.asarray(scan_hidden)[0].reshape(-1)
  assert np.signbit(tape_row0[0]) and tape_row0[0] == 0.0


def test_fwd_tape_scan_single_cached_program_and_signature_guard():
  engine, runner = _payload_engine(2, jnp.float32)
  leaves = tuple(runner.state_leaves)
  hidden = engine.run_embed_forward(
      jnp.asarray([1, 2, 3], jnp.int32), state_leaves=leaves
  )
  metadata = jnp.asarray(0.125, jnp.float32)
  caches = [jnp.asarray(0.0, jnp.float32)] * 2
  assert engine._p71_fwd_scan_fn is None  # pylint: disable=protected-access
  engine.run_layers_fwd_tape_scan(leaves, caches, hidden, metadata)
  built = engine._p71_fwd_scan_fn  # pylint: disable=protected-access
  assert built is not None
  spy = _CountingWrapper(built)
  engine._p71_fwd_scan_fn = spy  # pylint: disable=protected-access
  engine.run_layers_fwd_tape_scan(leaves, caches, hidden, metadata)
  # ONE scanned program rebuilds the whole chunk tape.
  assert spy.calls == 1
  engine._p71_fwd_scan_fn = built  # pylint: disable=protected-access
  assert built._cache_size() == 1
  with pytest.raises(
      FunctionalMappingError,
      match="P71 fwd-scan operand signature changed",
  ):
    engine.run_layers_fwd_tape_scan(
        leaves, caches, hidden[:2], metadata
    )


# ---------------------------------------------------------------------------
# Full reverse-group bitwise parity, legacy vs fwd
# ---------------------------------------------------------------------------


def _group_adapter(rank_parallel):
  fixtures = _fixtures()
  case = fixtures.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, runner = case._make_p32_group_adapter()  # pylint: disable=protected-access
  if not rank_parallel:
    runner.mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1), ("data", "model")
    )
    return adapter, runner
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:16]).reshape(16, 1), ("data", "model")
  )
  adapter._tp_size = 1  # pylint: disable=protected-access
  graphdef, state = nnx.split(runner.model)
  state = jax.tree.map(
      lambda value: jax.device_put(
          value,
          jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
      ),
      state,
  )
  runner.model = nnx.merge(graphdef, state)
  _, runner.state = nnx.split(runner.model)
  runner.state_leaves = tuple(jax.tree.leaves(runner.state))
  runner.mesh = mesh
  adapter._engine_state_contract = runner.state  # pylint: disable=protected-access
  cache_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data")
  )
  adapter._fresh_caches = types.MethodType(  # pylint: disable=protected-access
      lambda self: [
          jax.device_put(jnp.zeros((64, 1), jnp.float32), cache_sharding),
          jax.device_put(jnp.zeros((64, 1), jnp.float32), cache_sharding),
      ],
      adapter,
  )

  def sharded_group_chunk_inputs(self, group_spec, chunk_index):
    start = chunk_index * self._sequence_bucket
    end = start + self._sequence_bucket
    return (
        jax.device_put(
            group_spec["packed_ids"][:, start:end].reshape(-1),
            cache_sharding,
        ),
        jax.device_put(
            group_spec["next_ids"][:, start:end].reshape(-1),
            cache_sharding,
        ),
        jnp.asarray(0.125, jnp.float32),
    )

  adapter._p32_group_chunk_inputs = types.MethodType(  # pylint: disable=protected-access
      sharded_group_chunk_inputs, adapter
  )
  return adapter, runner


def _two_chunk_spec(adapter):
  # Three valid prompt + three valid completion tokens per rank:
  # n_real = 6 > sequence_bucket = 4, so the group reverses TWO chunks —
  # the scan program dispatches once per chunk and the cross-chunk
  # tree_add path runs.
  row = jnp.arange(16, dtype=jnp.int32)[:, None]
  prompt = jnp.concatenate(
      (1 + row % 2, 2 + row % 2, 1 + row % 3), axis=1
  )
  completion = jnp.concatenate(
      (2 + row % 2, 1 + row % 3, 1 + row % 2), axis=1
  )
  spec = adapter._p32_group_spec(  # pylint: disable=protected-access
      prompt,
      completion,
      jnp.ones_like(prompt, dtype=bool),
      jnp.ones_like(completion, dtype=bool),
      1.0,
  )
  assert spec["num_chunks"] == 2
  return spec


def _reverse_pair(rank_parallel):
  adapter, runner = _group_adapter(rank_parallel)
  spec = _two_chunk_spec(adapter)
  env = dict(_SEGMENTED_ENV)
  if rank_parallel:
    env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  with mock.patch.dict(os.environ, env, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
    leaves = tuple(runner.state_leaves)
    forward = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=False
    )
    dlogps = jnp.ones_like(forward["logps"])
    dentropy = jnp.zeros_like(forward["entropy"])
    legacy = adapter._p32_reverse_group(  # pylint: disable=protected-access
        engine, leaves, spec, dlogps, dentropy
    )
    with mock.patch.dict(
        os.environ, {"CANON_P71_SCAN": "fwd"}, clear=False
    ):
      scanned = adapter._p32_reverse_group(  # pylint: disable=protected-access
          engine, leaves, spec, dlogps, dentropy
      )
  return forward, legacy, scanned


@pytest.mark.parametrize(
    "rank_parallel", [False, True], ids=["serial", "rank-parallel-16dev"]
)
def test_reverse_group_gradients_bitwise_legacy_vs_fwd(rank_parallel):
  forward, legacy, scanned = _reverse_pair(rank_parallel)
  assert _tree_bytes(scanned["engine_gradients"]) == _tree_bytes(
      legacy["engine_gradients"]
  )
  assert _tree_bytes(scanned["initial_cache_cotangents"]) == _tree_bytes(
      legacy["initial_cache_cotangents"]
  )
  assert _tree_bytes(scanned["replay_logps"]) == _tree_bytes(
      legacy["replay_logps"]
  )
  assert _tree_bytes(legacy["replay_logps"]) == _tree_bytes(
      forward["logps"]
  )
  # The scan kept the per-layer forward accounting (28-per-chunk shape of
  # the real carrier; num_layers * num_chunks here).
  assert scanned["counts"]["layer_forward"] == legacy["counts"][
      "layer_forward"
  ]
  if rank_parallel:
    first = jax.tree.leaves(scanned["engine_gradients"])[0]
    assert first.sharding.spec[0] == "data"  # staged rank axis intact


# ---------------------------------------------------------------------------
# Flag-off digest frozen from HEAD
# ---------------------------------------------------------------------------


def test_flag_off_serial_reverse_matches_digest_frozen_from_head():
  _, legacy, _ = _reverse_pair(False)
  payload = b"".join(
      _tree_bytes(legacy["engine_gradients"])
      + _tree_bytes(legacy["replay_logps"])
  )
  digest = hashlib.sha256(payload).hexdigest()
  if os.environ.get("P71_PRINT_DIGEST", "") == "1":
    print(f"[P71] flag-off serial digest: {digest}", flush=True)
  assert digest == FLAG_OFF_DIGEST


# ---------------------------------------------------------------------------
# Fail-closed combinations
# ---------------------------------------------------------------------------


def test_fwd_conflicts_fail_closed():
  adapter, runner = _group_adapter(False)
  spec = _two_chunk_spec(adapter)
  with mock.patch.dict(os.environ, _SEGMENTED_ENV, clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
    leaves = tuple(runner.state_leaves)
    forward = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=False
    )
    dlogps = jnp.ones_like(forward["logps"])
    dentropy = jnp.zeros_like(forward["entropy"])
    with mock.patch.dict(
        os.environ,
        {"CANON_P71_SCAN": "fwd", "CANON_P66_BACKWARD_ARM": "tp4-serial"},
        clear=False,
    ):
      with pytest.raises(
          FunctionalMappingError,
          match="cannot combine with CANON_P66_BACKWARD_ARM",
      ):
        adapter._p32_reverse_group(  # pylint: disable=protected-access
            engine, leaves, spec, dlogps, dentropy
        )
    with mock.patch.dict(
        os.environ,
        {"CANON_P71_SCAN": "fwd", "CANON_P28_LAYER_SCAN": "1"},
        clear=False,
    ):
      with pytest.raises(
          FunctionalMappingError,
          match="requires CANON_P28_LAYER_SCAN unset",
      ):
        adapter._p32_reverse_group(  # pylint: disable=protected-access
            engine, leaves, spec, dlogps, dentropy
        )
