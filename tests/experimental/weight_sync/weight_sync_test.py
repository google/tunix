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

"""Unit and integration tests for weight synchronization contract and execution."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from typing import Any, Optional

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.weight_sync import raiden_handler
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.experimental.weight_sync import raiden_weight_sync_delegate
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.weight_sync import weight_sync_coordinator
from tunix.experimental.worker import rollout_worker
from tunix.experimental.worker import trainer_worker
from tunix.generate import base_sampler
from tunix.generate import sampler as vanilla_sampler_lib
from tunix.generate import utils as gen_utils
from tunix.rl import reshard
from tunix.tests import test_common

# pylint: disable=g-import-not-at-top
import multiprocessing

TrainingConfig = peft_trainer_v2.TrainingConfig
PeftTrainer = peft_trainer_v2.PeftTrainer
TrainerWorker = trainer_worker.TrainerWorker
RolloutConfig = rollout_worker.RolloutConfig
WeightSyncMode = weight_sync.WeightSyncMode


def _tensor(**overrides) -> weight_sync.TensorMetadata:
  fields = dict(
      name="w",
      shape=(8, 4),
      mesh_shape=(1, 2),
      layout=(1, 0),
      item_size=4,
      sharding_spec=("", "tp"),
  )
  fields.update(overrides)
  return weight_sync.TensorMetadata(**fields)


class WorkUnitIdTest(absltest.TestCase):

  def test_preserves_the_complete_transport_neutral_identity(self):
    unit = weight_sync.WorkUnitId(
        job_name="trainer",
        job_replica_id="host-2",
        data_name="model.layers.0.mlp.weight",
        data_replica_idx=3,
    )

    self.assertEqual(unit.job_name, "trainer")
    self.assertEqual(unit.job_replica_id, "host-2")
    self.assertEqual(unit.data_name, "model.layers.0.mlp.weight")
    self.assertEqual(unit.data_replica_idx, 3)

  def test_rejects_an_empty_job_or_negative_replica(self):
    with self.assertRaisesRegex(ValueError, "job_name"):
      weight_sync.WorkUnitId(job_name="")
    with self.assertRaisesRegex(ValueError, "non-negative"):
      weight_sync.WorkUnitId(job_name="trainer", data_replica_idx=-1)


class TensorMetadataTest(absltest.TestCase):

  def test_accepts_partial_layout_and_replicated_dimensions(self):
    tensor = _tensor(layout=(-1, 0))

    self.assertEqual(tensor.layout, (-1, 0))
    self.assertEqual(tensor.sharding_spec, ("", "tp"))

  def test_rejects_an_empty_name_or_invalid_shape(self):
    with self.assertRaisesRegex(ValueError, "name"):
      _tensor(name="")
    for shape in ((), (8, 0), (-1, 4)):
      with self.subTest(shape=shape):
        with self.assertRaisesRegex(ValueError, "invalid shape"):
          _tensor(shape=shape)

  def test_rejects_invalid_mesh_or_layout_rank(self):
    with self.assertRaisesRegex(ValueError, "mesh_shape"):
      _tensor(mesh_shape=(2,))
    with self.assertRaisesRegex(ValueError, "positive dimensions"):
      _tensor(mesh_shape=(1, 0))
    with self.assertRaisesRegex(ValueError, "layout"):
      _tensor(layout=(0,))

  def test_rejects_invalid_item_size_or_layer_index(self):
    with self.assertRaisesRegex(ValueError, "item_size"):
      _tensor(item_size=0)
    with self.assertRaisesRegex(ValueError, "layer_idx"):
      _tensor(layer_idx=-1)

  def test_rejects_invalid_sharding_spec(self):
    with self.assertRaisesRegex(ValueError, "sharding_spec"):
      _tensor(sharding_spec=("tp",))
    with self.assertRaisesRegex(ValueError, "may not shard two"):
      _tensor(sharding_spec=("tp", "tp"))


class NeutralContractTest(absltest.TestCase):

  def test_work_unit_metadata_carries_a_multi_tensor_manifest(self):
    tensors = (_tensor(name="w0"), _tensor(name="w1", layer_idx=1))
    metadata = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId("trainer"),
        variables=tensors,
        mesh_shape=(1, 2),
        mesh_axes=("fsdp", "tp"),
    )

    self.assertEqual(metadata.variables, tensors)
    self.assertEqual(metadata.mesh_axes, ("fsdp", "tp"))

  def test_handler_boundary_exposes_no_raiden_types(self):
    with self.assertRaises(TypeError):
      weight_sync.WeightSyncHandler()
    for method_name in ("register_work_unit", "transfer"):
      signature = inspect.signature(
          getattr(weight_sync.WeightSyncHandler, method_name)
      )
      self.assertNotIn("Raiden", str(signature))
    self.assertFalse(hasattr(weight_sync.TensorMetadata, "to_proto"))

  def test_module_does_not_expose_the_raiden_implementation(self):
    self.assertFalse(hasattr(weight_sync, "RaidenHandler"))
    self.assertFalse(hasattr(weight_sync, "RaidenTransferOptions"))
    self.assertFalse(hasattr(weight_sync, "raiden_controller"))


class MockVllmSampler(base_sampler.BaseSampler):
  """Mock vLLM sampler exposing VllmToyTransformer state and update_params."""

  def __init__(self, config: test_common.ModelConfig, *, rngs: nnx.Rngs):
    self._transformer = test_common.VllmToyTransformer(config, rngs=rngs)
    self._transformer_state = nnx.state(self._transformer)
    self.mesh = None

  @property
  def transformer(self) -> nnx.Module:
    return self._transformer

  @property
  def transformer_state(self) -> Any:
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: Any) -> None:
    self._transformer_state = state

  def update_params(
      self,
      params: Any,
      filter_types: Optional[tuple[Any, ...]] = None,
  ) -> None:
    del filter_types
    converted = gen_utils.transfer_state_with_mappings(
        src_state=params,
        dst_state=self._transformer_state,
        key_mappings=test_common.TOY_TRANSFORMER_TO_HF_MAPPINGS,
        reshard_fn=reshard.reshard_pytree,
        rollout_engine="vllm_jax",
    )
    self._transformer_state = converted

  def __call__(self, *args, **kwargs):
    return base_sampler.SamplerOutput(
        text=["mock completion"],
        logits=None,
        tokens=np.array([1, 2, 3], dtype=np.int32),
        padded_prompt_tokens=np.array([[1]], dtype=np.int32),
        logprobs=None,
    )

  def tokenize(self, input_string: str) -> np.ndarray:
    del input_string
    return np.array([1, 2], dtype=np.int32)

  def get_target_state(self) -> Any:
    return jax.tree.map(
        lambda x: nnx.Param(jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)),
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def stop(self) -> None:
    pass


def _to_flat_dict(state: Any) -> dict[str, Any]:
  if hasattr(state, "flat_state"):
    return {".".join(str(p) for p in path): var for path, var in state.flat_state()}
  return dict(state)


def _trainer_process_fn(
    conn,
    model_config_kwargs: dict[str, Any],
    sampler_type: str,
    target_state: Any,
):
  model_config = test_common.ModelConfig(**model_config_kwargs)
  toy_trainer_model = test_common.ToyTransformer(
      model_config, rngs=nnx.Rngs(0)
  )
  # Initialize any lazy module parameters
  toy_trainer_model(
      jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32)
  )
  trainer_config = TrainingConfig(
      eval_every_n_steps=2,
      max_steps=10,
  )
  trainer = PeftTrainer(
      model=toy_trainer_model,
      optimizer=optax.sgd(1e-3),
      training_config=trainer_config,
      target_state=target_state,
      sampler_type=sampler_type,
  )
  while True:
    try:
      msg = conn.recv()
    except EOFError:
      break
    cmd = msg[0]
    if cmd == "prepare_weight_sync":
      try:
        res = trainer.prepare_weight_sync(sync_request=msg[1])
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "release_weight_sync":
      try:
        res = trainer.release_weight_sync(sync_request=msg[1])
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "stop":
      break


def _rollout_process_fn(
    conn,
    model_config_kwargs: dict[str, Any],
    adapter_config_kwargs: dict[str, Any],
):
  model_config = test_common.ModelConfig(**model_config_kwargs)
  sampler_type = adapter_config_kwargs.get("sampler_type", "inprocess_vllm")
  delegate = raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
  adapter_config = RolloutConfig(**adapter_config_kwargs)

  if sampler_type == "vanilla":
    vanilla_model = test_common.ToyTransformer(model_config, rngs=nnx.Rngs(1))
    vanilla_model(
        jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32)
    )
    adapter = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="vanilla_sampler_coord_0",
        config=adapter_config,
        raiden_sync_delegate=delegate,
    )
    adapter.sampler = vanilla_sampler_lib.Sampler(
        transformer=vanilla_model,
        tokenizer=test_common.MockVocab(),
        cache_config=vanilla_sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
  else:
    mock_vllm_sampler = MockVllmSampler(model_config, rngs=nnx.Rngs(1))
    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_sampler_coord_0",
        config=adapter_config,
        raiden_sync_delegate=delegate,
    )
    adapter.vllm_sampler = mock_vllm_sampler

  while True:
    try:
      msg = conn.recv()
    except EOFError:
      break
    cmd = msg[0]
    if cmd == "get_target_state":
      conn.send(("ok", adapter.get_target_state()))
    elif cmd == "bind_weight_sync":
      try:
        res = asyncio.run(adapter.bind_weight_sync())
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "get_weight_sync_metadata":
      try:
        res = asyncio.run(adapter.get_weight_sync_metadata())
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "pre_weight_sync":
      try:
        res = asyncio.run(adapter.pre_weight_sync(msg[1]))
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "weight_sync":
      try:
        res = asyncio.run(adapter.weight_sync(msg[1]))
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "post_weight_sync":
      try:
        res = asyncio.run(adapter.post_weight_sync(msg[1]))
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "abort_weight_sync":
      try:
        res = asyncio.run(adapter.abort_weight_sync(msg[1]))
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "get_weight_sync_status":
      try:
        res = asyncio.run(adapter.get_weight_sync_status())
        conn.send(("ok", res))
      except Exception as e:  # pylint: disable=broad-exception-caught
        conn.send(("error", str(e)))
    elif cmd == "is_bounded":
      conn.send(("ok", delegate.is_bounded()))
    elif cmd == "stop":
      break


class RemoteTrainerWorker:
  """Proxy to communicate with TrainerWorker running in a separate process."""

  def __init__(self, conn):
    self._conn = conn
    self._info = datatypes.WorkerInfo(
        worker_id="trainer_proc_0",
        roles=frozenset({datatypes.Role.ACTOR.value}),
    )

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  async def prepare_weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("prepare_weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Trainer prepare failed: {res}")
    return res

  async def release_weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("release_weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Trainer release failed: {res}")
    return res


class RemoteRolloutWorker:
  """Proxy to communicate with RolloutWorker running in a separate process."""

  def __init__(self, conn):
    self._conn = conn
    self._info = datatypes.WorkerInfo(
        worker_id="rollout_proc_0",
        roles=frozenset({datatypes.Role.ROLLOUT.value}),
    )

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  async def bind_weight_sync(self, *args, **kwargs):
    self._conn.send(("bind_weight_sync",))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout bind failed: {res}")
    return res

  async def get_weight_sync_metadata(self, *args, **kwargs):
    self._conn.send(("get_weight_sync_metadata",))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout metadata failed: {res}")
    return res

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("pre_weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout pre_weight_sync failed: {res}")
    return res

  async def weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout weight_sync failed: {res}")
    return res

  async def post_weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("post_weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout post_weight_sync failed: {res}")
    return res

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs):
    self._conn.send(("abort_weight_sync", sync_request))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout abort failed: {res}")
    return res

  async def get_weight_sync_status(self, *args, **kwargs):
    self._conn.send(("get_weight_sync_status",))
    status, res = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"Rollout status failed: {res}")
    return res


class WeightSyncE2ETest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_config = test_common.ModelConfig(
        num_layers=2,
        vocab_size=128,
    )
    self.toy_trainer_model = test_common.ToyTransformer(
        self.model_config, rngs=nnx.Rngs(0)
    )
    self.mock_vllm_sampler = MockVllmSampler(
        self.model_config, rngs=nnx.Rngs(1)
    )

  def test_fallback_mode_weight_sync_with_vanilla_adapter(self):
    """Verifies Fallback mode weight transfer from PeftTrainer to VanillaSamplerAdapter."""
    adapter_config = RolloutConfig(
        sampler_type="vanilla",
        weight_sync_mode=WeightSyncMode.FALLBACK,
    )
    vanilla_model = test_common.ToyTransformer(
        self.model_config, rngs=nnx.Rngs(2)
    )
    adapter = vanilla_sampler_adapter.VanillaSamplerAdapter(
        server_id="vanilla_sampler_fallback_0",
        config=adapter_config,
    )
    adapter.sampler = vanilla_sampler_lib.Sampler(
        transformer=vanilla_model,
        tokenizer=test_common.MockVocab(),
        cache_config=vanilla_sampler_lib.CacheConfig(
            cache_size=64,
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
        ),
    )

    trainer_config = TrainingConfig(
        eval_every_n_steps=2,
        max_steps=10,
    )
    trainer = PeftTrainer(
        model=self.toy_trainer_model,
        optimizer=optax.sgd(1e-3),
        training_config=trainer_config,
        sampler_type=adapter_config.sampler_type,
    )

    # Mutate trainer model weights
    new_embedding = jnp.ones_like(self.toy_trainer_model.emb.embedding.value) * 99.0
    self.toy_trainer_model.emb.embedding.value = new_embedding

    # Execute weight sync in fallback mode
    sync_request = base_sampler_lib.WeightSyncRequest(
        weights=nnx.state(trainer.model),
    )
    result = asyncio.run(adapter.weight_sync(sync_request))
    self.assertTrue(result)

    # Verify adapter sampler received the updated weights directly
    tgt_flat = _to_flat_dict(adapter.sampler.transformer_state)
    np.testing.assert_allclose(
        np.array(tgt_flat["emb.embedding"].value),
        np.array(new_embedding),
    )

  def test_fallback_mode_weight_sync_with_vllm_adapter(self):
    """Verifies Fallback mode weight transfer from PeftTrainer to InprocessVllmSamplerAdapter."""
    adapter_config = RolloutConfig(
        sampler_type="inprocess_vllm",
        weight_sync_mode=WeightSyncMode.FALLBACK,
    )
    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_sampler_fallback_0",
        config=adapter_config,
    )
    adapter.vllm_sampler = self.mock_vllm_sampler

    trainer_config = TrainingConfig(
        eval_every_n_steps=2,
        max_steps=10,
    )
    trainer = PeftTrainer(
        model=self.toy_trainer_model,
        optimizer=optax.sgd(1e-3),
        training_config=trainer_config,
        target_state=adapter.get_target_state(),
        sampler_type=adapter_config.sampler_type,
    )

    # Mutate trainer model weights
    new_embedding = jnp.ones_like(self.toy_trainer_model.emb.embedding.value) * 42.0
    self.toy_trainer_model.emb.embedding.value = new_embedding

    # Execute weight sync in fallback mode
    sync_request = base_sampler_lib.WeightSyncRequest(
        weights=nnx.state(trainer.model),
    )
    result = asyncio.run(adapter.weight_sync(sync_request))
    self.assertTrue(result)

    # Verify mock_vllm_sampler received the converted parameter under the vLLM key
    tgt_flat = _to_flat_dict(self.mock_vllm_sampler.transformer_state)
    np.testing.assert_allclose(
        np.array(tgt_flat["model.embed_tokens.embedding"].value),
        np.array(new_embedding),
    )

  @absltest.skipIf(
      raiden_synchronizer._ws_lib is None,
      "Requires TPU Raiden C++ extension (_ws_lib)",
  )
  def test_raiden_mode_multiprocess_weight_sync_with_vanilla_sampler(self):
    """Verifies end-to-end Raiden weight sync with VanillaSampler across separate processes."""
    ctx = multiprocessing.get_context("spawn")
    rollout_parent_conn, rollout_child_conn = ctx.Pipe()
    trainer_parent_conn, trainer_child_conn = ctx.Pipe()

    model_config_dict = dataclasses.asdict(self.model_config)
    adapter_config = RolloutConfig(
        sampler_type="vanilla",
        weight_sync_mode=WeightSyncMode.RAIDEN,
    )
    adapter_config_dict = dataclasses.asdict(adapter_config)

    # 1. Start Rollout Process
    rollout_proc = ctx.Process(
        target=_rollout_process_fn,
        args=(rollout_child_conn, model_config_dict, adapter_config_dict),
    )
    rollout_proc.start()

    # Extract target state structure from Rollout Process
    rollout_parent_conn.send(("get_target_state",))
    status, target_state = rollout_parent_conn.recv()
    self.assertEqual(status, "ok")

    # 2. Start Trainer Process
    trainer_proc = ctx.Process(
        target=_trainer_process_fn,
        args=(
            trainer_child_conn,
            model_config_dict,
            adapter_config.sampler_type,
            target_state,
        ),
    )
    trainer_proc.start()

    # 3. Register remote workers in orchestrator registry
    trainer_worker_proxy = RemoteTrainerWorker(trainer_parent_conn)
    rollout_worker_proxy = RemoteRolloutWorker(rollout_parent_conn)

    registry = worker_registry.WorkerRegistry()
    registry.register(trainer_worker_proxy)
    registry.register(rollout_worker_proxy)

    # 4. Run real RaidenHandler and WeightSyncCoordinator in main process
    handler = raiden_handler.RaidenHandler(port=0)
    try:
      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=registry,
          handler=handler,
      )
      try:
        result = asyncio.run(coordinator.sync(policy_version=1))
        self.assertTrue(result)
        self.assertEqual(coordinator.last_committed_version, 1)
      except weight_sync_coordinator.WeightSyncError as e:
        self.fail(f"WeightSync failed with issues: {e.result.failures}")
    finally:
      handler.close()
      trainer_parent_conn.send(("stop",))
      rollout_parent_conn.send(("stop",))
      trainer_proc.join(timeout=5)
      rollout_proc.join(timeout=5)

  @absltest.skipIf(
      raiden_synchronizer._ws_lib is None,
      "Requires TPU Raiden C++ extension (_ws_lib)",
  )
  def test_raiden_mode_multiprocess_weight_sync_with_vllm_sampler(self):
    """Verifies end-to-end Raiden weight sync with vLLM Sampler across separate processes."""
    ctx = multiprocessing.get_context("spawn")
    rollout_parent_conn, rollout_child_conn = ctx.Pipe()
    trainer_parent_conn, trainer_child_conn = ctx.Pipe()

    model_config_dict = dataclasses.asdict(self.model_config)
    adapter_config = RolloutConfig(
        sampler_type="inprocess_vllm",
        weight_sync_mode=WeightSyncMode.RAIDEN,
    )
    adapter_config_dict = dataclasses.asdict(adapter_config)

    # 1. Start Rollout Process
    rollout_proc = ctx.Process(
        target=_rollout_process_fn,
        args=(rollout_child_conn, model_config_dict, adapter_config_dict),
    )
    rollout_proc.start()

    # Extract target state structure from Rollout Process
    rollout_parent_conn.send(("get_target_state",))
    status, target_state = rollout_parent_conn.recv()
    self.assertEqual(status, "ok")

    # 2. Start Trainer Process
    trainer_proc = ctx.Process(
        target=_trainer_process_fn,
        args=(
            trainer_child_conn,
            model_config_dict,
            adapter_config.sampler_type,
            target_state,
        ),
    )
    trainer_proc.start()

    # 3. Register remote workers in orchestrator registry
    trainer_worker_proxy = RemoteTrainerWorker(trainer_parent_conn)
    rollout_worker_proxy = RemoteRolloutWorker(rollout_parent_conn)

    registry = worker_registry.WorkerRegistry()
    registry.register(trainer_worker_proxy)
    registry.register(rollout_worker_proxy)

    # 4. Run real RaidenHandler and WeightSyncCoordinator in main process
    handler = raiden_handler.RaidenHandler(port=0)
    try:
      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=registry,
          handler=handler,
      )
      result = asyncio.run(coordinator.sync(policy_version=1))
      self.assertTrue(result)
      self.assertEqual(coordinator.last_committed_version, 1)
    finally:
      handler.close()
      trainer_parent_conn.send(("stop",))
      rollout_parent_conn.send(("stop",))
      trainer_proc.join(timeout=5)
      rollout_proc.join(timeout=5)


if __name__ == "__main__":
  absltest.main()

