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
"""Exercise chunk-start metadata through the real checked report VJP."""

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
        "XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") +
        " --xla_force_host_platform_device_count=4").strip()

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as module


def _state(tree):
    return nnx.State(jax.tree.map(nnx.Param, tree))


@contextlib.contextmanager
def _no_host_round_trip():
    # CPU-backed NumPy reads can be zero-copy and evade transfer_guard.
    # Catch those materializations too; D2D placement remains permitted.
    original_asarray, original_array = np.asarray, np.array

    def checked(convert, value, *args, **kwargs):
        if any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(value)):
            raise AssertionError("Host materialization of a JAX array")
        return convert(value, *args, **kwargs)

    with (
            jax.transfer_guard_host_to_device("disallow_explicit"),
            jax.transfer_guard_device_to_host("disallow_explicit"),
            mock.patch.object(
                np,
                "asarray",
                side_effect=lambda value, *a, **k: checked(
                    original_asarray, value, *a, **k),
            ),
            mock.patch.object(
                np,
                "array",
                side_effect=lambda value, *a, **k: checked(
                    original_array, value, *a, **k),
            ),
            mock.patch.object(
                jax,
                "device_get",
                side_effect=AssertionError(
                    "Host materialization via device_get"),
            ),
    ):
        yield


@pytest.mark.parametrize("dp,tp", [(4, 1), (2, 2)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("donate", [False, True])
@pytest.mark.parametrize("mapped_producer", [False, True])
def test_tree_start_preserves_real_checked_mapping(dp, tp, dtype, donate,
                                                   mapped_producer):
    if (mapped_producer and tp > 1
            and not hasattr(jax.core.ShapedArray((), jnp.float32), "mat")):
        pytest.skip(
            "The checked producer requires the pinned image's JAX MAT API; "
            "run this cell there")
    assert len(jax.devices()) >= 4
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(dp, tp), ("dp", "tp"))

    def named(*spec):
        return jax.sharding.NamedSharding(mesh,
                                          jax.sharding.PartitionSpec(*spec))

    def put(value, *spec):
        return jax.device_put(np.asarray(value).astype(dtype), named(*spec))

    trainer = _state({
        "trainer": {
            "a": put(np.arange(32).reshape(8, 4), "tp", None),
            "b": put(np.arange(4), None),
        }
    })
    contract = _state({
        "engine": {
            "a": put(np.zeros((4, 8)), None, "tp"),
            "b": put(np.zeros(4), None),
        }
    })
    adapter = object.__new__(module.Qwen3EngineForwardAdapter)
    adapter._runner = SimpleNamespace(model_config=SimpleNamespace(
        get_total_num_kv_heads=lambda: 1, get_head_size=lambda: 1))
    adapter._engine_state_contract = contract
    adapter._key_mappings = {
        "trainer.a": ("engine.a", (None, None)),
        "trainer.b": ("engine.b", (None, )),
    }
    adapter._transpose_keys = {"trainer.a": (1, 0)}
    adapter._hook_fns = None
    adapter._tp_size, adapter._data_size, adapter._dp_axis = tp, dp, "data"
    producer = None
    engine_values = tuple(jax.tree.leaves(contract))

    def pack():
        nonlocal producer
        result = (
            put(
                np.arange(dp * 32).reshape(dp, 4, 8) / 7 + 3, "dp", None,
                "tp"),
            put(np.arange(dp * 4).reshape(dp, 4) / 5 + 3, "dp", None),
        )
        if mapped_producer:
            if producer is None:
                segmented = object.__new__(module._P28SegmentedEngineForward)
                segmented._engine_mesh = mesh

                def local_pullback(local_values, local_cotangents):
                    _, pullback = jax.vjp(lambda values: values, local_values)
                    gradient = pullback(
                        tuple(jnp.squeeze(x, 0) for x in local_cotangents))[0]
                    return jax.tree.map(lambda x: jnp.expand_dims(x, 0),
                                        gradient)

                with mock.patch.dict(
                        os.environ,
                    {"CANON_P66_P59_CHECK_VMA": "1" if tp > 1 else "0"}):
                    producer = segmented._p59_parallel_map(
                        local_pullback, (engine_values, result),
                        lambda axis, size, aligned, manual: module.
                        _rank_staged_specs(aligned[0], axis, manual),
                        rank_local_arg_indices=(1, ),
                        module_name="zt_test_mapping_producer",
                        scope_name="test/mapping_producer")
            result = producer(engine_values, result)
        return result

    with mock.patch.dict(os.environ,
                         {"CANON_P75_REPORT_ADJOINT_BUCKETS": "0"}):
        expected = adapter._p59_rank_parallel_report_adjoint(trainer, pack())
        # Initialize runtime-zero constants before the guarded warm dispatch.
        jax.block_until_ready(
            adapter._p70_grad_tree_start(pack(), donate_pack=donate))
        inputs = pack()
        desired = tuple(value.sharding for value in inputs)
        accumulator = jax.tree.map(jnp.zeros_like, expected)
        checked_calls = []
        original = jax.shard_map

        def record(*args, **kwargs):
            checked_calls.append(kwargs.get("check_vma"))
            return original(*args, **kwargs)

        # The existing runtime-zero operand is replicated device-to-device.
        # This new integration check forbids both host directions, not D2D;
        # the existing report-only all-transfer guard remains unchanged.
        with mock.patch.object(jax, "shard_map",
                               side_effect=record), _no_host_round_trip():
            started = adapter._p70_grad_tree_start(inputs, donate_pack=donate)
            assert tuple(value.sharding for value in started) == desired
            actual, receipts = adapter._p59_rank_parallel_report_adjoint_accumulate(
                trainer, started, accumulator)
            jax.block_until_ready((actual, receipts))
        assert checked_calls and all(value is True for value in checked_calls)
        for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual),
                                              jax.tree.leaves(expected),
                                              strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf).view(np.uint32),
                np.asarray(expected_leaf).view(np.uint32))
        if donate:
            assert all(value.is_deleted() for value in inputs)


def test_tree_start_rejects_new_physical_layout_before_dispatch():
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
    named = lambda *spec: jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec))
    adapter = object.__new__(module.Qwen3EngineForwardAdapter)
    original = jax.device_put(np.ones((4, 4), np.float32), named("dp", "tp"))
    changed = jax.device_put(np.ones((4, 4), np.float32), named("tp", "dp"))
    jax.block_until_ready(adapter._p70_grad_tree_start((original, )))
    with mock.patch.object(adapter, "_p70_tree_start_fn") as program:
        with pytest.raises(module.FunctionalMappingError,
                           match="layout changed at leaf 0"):
            adapter._p70_grad_tree_start((changed, ))
        program.assert_not_called()


def test_tree_start_accepts_physically_equal_unit_axis_spelling():
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(4, 1), ("dp", "tp"))
    named = lambda *spec: jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec))
    adapter = object.__new__(module.Qwen3EngineForwardAdapter)
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    explicit = jax.device_put(data, named("dp", "tp"))
    equivalent = jax.device_put(data, named("dp"))
    jax.block_until_ready(adapter._p70_grad_tree_start((explicit, )))
    with _no_host_round_trip():
        result, = adapter._p70_grad_tree_start((equivalent, ))
        jax.block_until_ready(result)
    assert result.sharding == explicit.sharding
    np.testing.assert_array_equal(
        np.asarray(result).view(np.uint32), data.view(np.uint32))


def test_host_guards_reject_injected_materializations():
    host = np.arange(4, dtype=np.float32)
    program = jax.jit(lambda value: value + 1)
    device = jax.block_until_ready(program(jax.device_put(host)))
    with _no_host_round_trip():
        with pytest.raises(jax.errors.JaxRuntimeError,
                           match="Disallowed host-to-device transfer"):
            program(host + 1)
        with pytest.raises(AssertionError, match="Host materialization"):
            np.asarray(device)
        with pytest.raises(AssertionError, match="Host materialization"):
            np.array([device])
        with pytest.raises(AssertionError, match="Host materialization"):
            jax.device_get(device)
