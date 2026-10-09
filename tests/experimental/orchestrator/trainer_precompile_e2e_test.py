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

"""End-to-end tests verifying trainer AOT precompilation hits JIT cache on step 1."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.train import abstract_trainer
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker
from tunix.rl import algorithm_config
from tunix.tests import test_common as tc


def _make_trajectory_item(
    prompt_id: str,
    group_index: int,
    prompt_len: int,
    completion_len: int,
    reward: float,
) -> datatypes.TrajectoryItem:
  prompt_ids = np.arange(1, prompt_len + 1, dtype=np.int32)
  completion_ids = np.arange(10, 10 + completion_len, dtype=np.int32)
  completion_logprobs = np.full(completion_len, -0.25, dtype=np.float32)
  return datatypes.TrajectoryItem(
      prompt_id=prompt_id,
      group_index=group_index,
      traj={
          "prompt_tokens": prompt_ids,
          "conversation_tokens": completion_ids,
          "conversation_masks": np.ones(completion_len, dtype=np.float32),
          "old_logprobs": completion_logprobs,
          "trajectory_reward": reward,
          "status": datatypes.TrajectoryStatus.SUCCEEDED,
      },
      metadata={"policy_version": 0},
  )


class _DirectWorkerHandle:
  """In-process adapter matching the `ActorHandle` subset used by `configure_worker`."""

  def __init__(self, worker: trainer_worker.TrainerWorker):
    self._worker = worker

  def submit(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(self._worker, method_name)(*args, **kwargs)


class _DummyMaxTextModel(nnx.Module):
  """Minimal MaxText-compatible NNX model for AOT compilation tests."""

  def __init__(self, vocab_size: int = 128, rngs: nnx.Rngs | None = None):
    if rngs is None:
      rngs = nnx.Rngs(0)
    self.embed = nnx.Embed(num_embeddings=vocab_size, features=16, rngs=rngs)
    self.head = nnx.Linear(16, vocab_size, rngs=rngs)

  def __call__(
      self,
      input_tokens: jax.Array,
      positions: jax.Array | None = None,
      cache: Any = None,
      attention_mask: jax.Array | None = None,
      output_hidden_states: bool = False,
      segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, None]:
    del (
        positions,
        cache,
        attention_mask,
        output_hidden_states,
        segment_ids,
    )
    x = self.embed(input_tokens)
    return self.head(x), None


def _jit_cache_size(fn: Any) -> int:
  while hasattr(fn, "func"):
    fn = fn.func
  return fn._fun._cache_size()


def _is_array_leaf(x: Any) -> bool:
  return isinstance(x, (jax.Array, np.ndarray))


def _batch_signature(dynamic_batch: dict[str, Any], static_batch: dict[str, Any]):
  dyn_treedef = jax.tree.structure(dynamic_batch)
  dyn_leaves = tuple(
      (np.asarray(leaf).shape, np.asarray(leaf).dtype)
      for leaf in jax.tree.leaves(dynamic_batch)
  )
  static_items = tuple(sorted(static_batch.items()))
  return (dyn_treedef, dyn_leaves, static_items)


class _AOTMaxTextStyleTrainer(abstract_trainer.AbstractTrainer):
  """AOT-compiling trainer mirroring `MaxTextTrainingEngine.compile_kernels`."""

  def __init__(self, model: nnx.Module, mesh: shd.Mesh):
    self.model = model
    self._mesh = mesh
    self.optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)
    self.loss_fn: Callable[..., Any] | None = None
    self.gen_model_input_fn: Callable[[Any], dict[str, Any]] = lambda x: {
        "inputs": x
    }
    self.train_steps = 0
    self.compile_kernels_calls = 0
    self._compiled = False
    self._compiled_signature: Any = None
    self._compiled_fwd_bwd: Any = None
    self._compiled_update: Any = None
    self._grad_accum: Any = None

  @property
  def _train_step(self) -> int:
    return self.train_steps

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "_AOTMaxTextStyleTrainer":
    del has_aux
    self.loss_fn = loss_fn
    self._compiled = False
    self._compiled_signature = None
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "_AOTMaxTextStyleTrainer":
    self.gen_model_input_fn = gen_model_input_fn
    self._compiled = False
    self._compiled_signature = None
    return self

  def _split_batch(
      self, dummy_or_real: Any
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = self.gen_model_input_fn(dummy_or_real)
    dynamic_batch: dict[str, Any] = {}
    static_batch: dict[str, Any] = {}
    for k, v in raw.items():
      leaves = jax.tree.leaves(v)
      if leaves and any(_is_array_leaf(leaf) for leaf in leaves):
        dynamic_batch[k] = jax.tree.map(
            lambda x: jnp.asarray(x) if _is_array_leaf(x) else x, v
        )
      else:
        static_batch[k] = v
    return dynamic_batch, static_batch

  def compile(self, dummy_data: Any = None) -> None:
    if dummy_data is None:
      return
    dynamic_batch, static_batch = self._split_batch(dummy_data)
    self._compile_kernels(dynamic_batch, static_batch)

  def _compile_kernels(
      self, dynamic_batch: dict[str, Any], static_batch: dict[str, Any]
  ) -> None:
    self.compile_kernels_calls += 1
    loss_fn = self.loss_fn
    assert loss_fn is not None

    with jax.set_mesh(self._mesh):
      graphdef, state = nnx.split((self.model, self.optimizer))
      model_state = nnx.state(self.model, nnx.Param)
      self._grad_accum = jax.tree.map(jnp.zeros_like, model_state)

      def _fwd_bwd(gdef, st, dyn_b):
        model, opt = nnx.merge(gdef, st)
        del opt

        def _loss(m):
          out = loss_fn(m, **dyn_b, **static_batch)
          if hasattr(out, "primary_loss"):
            return out.primary_loss.compute(), out.aux_metrics
          return out

        (loss, aux), grads = nnx.value_and_grad(
            _loss, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
        )(model)
        del aux
        return loss, grads

      def _update(gdef, st, grads):
        model, opt = nnx.merge(gdef, st)
        opt.update(model, grads)
        return nnx.state((model, opt))

      self._compiled_fwd_bwd = (
          jax.jit(_fwd_bwd).lower(graphdef, state, dynamic_batch).compile()
      )
      self._compiled_update = (
          jax.jit(_update).lower(graphdef, state, self._grad_accum).compile()
      )
      self._compiled = True
      self._compiled_signature = _batch_signature(dynamic_batch, static_batch)

  def fwd_bwd(self, batch: Any, **kwargs: Any) -> Any:
    del kwargs
    dynamic_batch, static_batch = self._split_batch(batch)
    sig = _batch_signature(dynamic_batch, static_batch)
    if not self._compiled or sig != self._compiled_signature:
      self._compile_kernels(dynamic_batch, static_batch)
    with jax.set_mesh(self._mesh):
      graphdef, state = nnx.split((self.model, self.optimizer))
      loss, self._grad_accum = self._compiled_fwd_bwd(
          graphdef, state, dynamic_batch
      )
      return loss

  def update(self, **kwargs: Any) -> int:
    del kwargs
    with jax.set_mesh(self._mesh):
      graphdef, state = nnx.split((self.model, self.optimizer))
      new_state = self._compiled_update(graphdef, state, self._grad_accum)
      nnx.update((self.model, self.optimizer), new_state)
    self.train_steps += 1
    return self.train_steps

  def train_step(self, batch: Any) -> Any:
    self.fwd_bwd(batch)
    return self.update()

  def eval_step(self, batch: Any) -> Any:
    return jnp.array(0.0)

  def save_checkpoint(self, step: int, **kwargs: Any) -> None:
    pass

  def restore_checkpoint(self, **kwargs: Any) -> None:
    pass

  def prepare_weight_sync(self, **kwargs: Any) -> None:
    pass

  def weight_sync(self, **kwargs: Any) -> None:
    pass

  def get_metrics(self) -> Any:
    return None

  def close(self) -> None:
    pass

  @contextlib.contextmanager
  def model_scope(self, *args: Any, **kwargs: Any):
    with jax.set_mesh(self._mesh):
      yield self.model, args, kwargs


class TrainerPrecompileE2ETest(parameterized.TestCase):
  """Verifies `configure_worker` AOT precompilation prevents step-1 recompiles."""

  def _make_trainer_worker(self) -> tuple[
      trainer_worker.TrainerWorker,
      peft_trainer_v2.PeftTrainer,
  ]:
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    config = peft_trainer_v2.TrainingConfig(
        eval_every_n_steps=10,
        max_steps=4,
        gradient_accumulation_steps=1,
    )
    trainer = peft_trainer_v2.PeftTrainer(
        model, optax.sgd(1e-3), config
    )
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: trainer,
    )
    return worker, trainer

  def _make_maxtext_trainer_worker(self) -> tuple[
      trainer_worker.TrainerWorker,
      _AOTMaxTextStyleTrainer,
  ]:
    mesh = shd.Mesh(np.array(jax.devices()[:1]), ("fsdp",))
    with jax.set_mesh(mesh):
      model = _DummyMaxTextModel(
          vocab_size=tc.ModelConfig().vocab_size, rngs=nnx.Rngs(0)
      )
    trainer = _AOTMaxTextStyleTrainer(model=model, mesh=mesh)
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: trainer,
        execution_context=mesh,
    )
    return worker, trainer

  @parameterized.named_parameters(
      dict(
          testcase_name="padded",
          use_packing=False,
          can_fuse_agreement_in_loss=True,
          sampler_is=None,
      ),
      dict(
          testcase_name="packed",
          use_packing=True,
          can_fuse_agreement_in_loss=True,
          sampler_is="token",
      ),
      dict(
          testcase_name="packed_non_fused_agreement",
          use_packing=True,
          can_fuse_agreement_in_loss=False,
          sampler_is="token",
      ),
  )
  def test_configure_worker_precompiles_and_hits_jit_cache_on_real_step(
      self,
      use_packing: bool,
      can_fuse_agreement_in_loss: bool,
      sampler_is: str | None,
  ):
    worker, trainer = self._make_trainer_worker()
    algo_config = algorithm_config.GRPOConfig(
        num_generations=2,
        num_iterations=2,
        beta=0.04,
        temperature=1.0,
        sampler_is=sampler_is,
    )
    algo = algorithm_adapter.GRPOAdapter(algo_config)

    if use_packing:
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=2,
          num_generations=2,
          mini_batch_size=1,
          max_packed_len=16,
          pad_id=0,
          max_segments_per_packed_row=2,
          segment_align_multiple=4,
      )
    else:
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=6,
          max_response_length=6,
          pad_id=0,
          num_generations=2,
          mini_batch_size=1,
      )

    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[mock.MagicMock(spec=remote_execution.ActorHandle)],
        trainer_workers={datatypes.Role.ACTOR: _DirectWorkerHandle(worker)},
        inference_workers={
            datatypes.Role.REFERENCE: mock.MagicMock(
                spec=remote_execution.ActorHandle
            )
        },
    )

    fwd_bwd_traces = 0
    update_traces = 0

    def traced_fwd_bwd(*args, _orig=trainer._fwd_bwd_step, **kwargs):
      nonlocal fwd_bwd_traces
      fwd_bwd_traces += 1
      return _orig(*args, **kwargs)

    def traced_update(*args, _orig=trainer._update_step, **kwargs):
      nonlocal update_traces
      update_traces += 1
      return _orig(*args, **kwargs)

    with (
        mock.patch.object(
            trainer, "create_fwd_bwd_step_fn", return_value=traced_fwd_bwd
        ),
        mock.patch.object(
            trainer, "create_update_step_fn", return_value=traced_update
        ),
    ):
      # 1. `configure_worker` must AOT-compile both `fwd_bwd` and `update`.
      engine.configure_worker(
          role=datatypes.Role.ACTOR,
          algo=algo,
          assembler=assembler,
          can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
      )
      self.assertEqual(fwd_bwd_traces, 1)
      self.assertEqual(update_traces, 1)
      self.assertIsNotNone(trainer._jitted_fwd_bwd_step_fn)
      self.assertIsNotNone(trainer._jitted_update_step_fn)
      self.assertEqual(trainer.train_steps, 0)

      # 2. Build a real training micro-batch through `GRPOAdapter` + `assembler`
      #    and run `fwd_bwd` + `update` on `worker`. Neither step may re-trace!
      trajectories = [
          _make_trajectory_item(
              "p0", 0, prompt_len=4, completion_len=4, reward=1.0
          ),
          _make_trajectory_item(
              "p0", 1, prompt_len=4, completion_len=3, reward=0.0
          ),
      ]
      raw_payloads = algo.create_trainer_payloads(
          trajectories, rewards=[1.0, 0.0]
      )
      batches = list(assembler.feed(raw_payloads))
      self.assertNotEmpty(batches)
      real_payload = batch_assembly.with_ref_per_token_logps(
          batches[0].payload,
          np.zeros_like(batches[0].payload.completion_ids, dtype=np.float32),
      )
      if sampler_is == "token" and not can_fuse_agreement_in_loss:
        real_payload = dataclasses.replace(
            real_payload,
            sampler_is_weights=np.ones_like(
                real_payload.completion_ids, dtype=np.float32
            ),
        )

      worker.fwd_bwd(datatypes.TrainRequest(payload=real_payload))
      worker.update()

      self.assertEqual(fwd_bwd_traces, 1)
      self.assertEqual(update_traces, 1)
      self.assertEqual(trainer.train_steps, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="maxtext_padded",
          use_packing=False,
          can_fuse_agreement_in_loss=True,
          sampler_is=None,
      ),
      dict(
          testcase_name="maxtext_packed",
          use_packing=True,
          can_fuse_agreement_in_loss=True,
          sampler_is="token",
      ),
      dict(
          testcase_name="maxtext_packed_non_fused_agreement",
          use_packing=True,
          can_fuse_agreement_in_loss=False,
          sampler_is="token",
      ),
  )
  def test_maxtext_configure_worker_precompiles_and_hits_aot_cache_on_real_step(
      self,
      use_packing: bool,
      can_fuse_agreement_in_loss: bool,
      sampler_is: str | None,
  ):
    worker, trainer = self._make_maxtext_trainer_worker()
    algo_config = algorithm_config.GRPOConfig(
        num_generations=2,
        num_iterations=2,
        beta=0.04,
        temperature=1.0,
        sampler_is=sampler_is,
    )
    algo = algorithm_adapter.GRPOAdapter(algo_config)

    if use_packing:
      assembler = batch_assembly.SequencePackedBatchAssembler(
          batch_size=2,
          num_generations=2,
          mini_batch_size=1,
          max_packed_len=16,
          pad_id=0,
          max_segments_per_packed_row=2,
          segment_align_multiple=4,
      )
    else:
      assembler = batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=6,
          max_response_length=6,
          pad_id=0,
          num_generations=2,
          mini_batch_size=1,
      )

    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[mock.MagicMock(spec=remote_execution.ActorHandle)],
        trainer_workers={datatypes.Role.ACTOR: _DirectWorkerHandle(worker)},
        inference_workers={
            datatypes.Role.REFERENCE: mock.MagicMock(
                spec=remote_execution.ActorHandle
            )
        },
    )

    # 1. `bring_up_workers` calls `compile(None)` first, then `configure_worker`
    #    calls `compile(dummy_payload)` and AOT-lowers/compiles `fwd_bwd` + `update`.
    worker.compile(None)
    self.assertEqual(trainer.compile_kernels_calls, 0)

    engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo=algo,
        assembler=assembler,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
    )
    self.assertEqual(trainer.compile_kernels_calls, 1)
    self.assertTrue(trainer._compiled)
    self.assertEqual(trainer.train_steps, 0)

    # 2. Build a real training micro-batch through `GRPOAdapter` + `assembler`
    #    and run `fwd_bwd` + `update` on `worker`. Signature must match the
    #    AOT-compiled dummy batch with zero recompilation!
    trajectories = [
        _make_trajectory_item(
            "p0", 0, prompt_len=4, completion_len=4, reward=1.0
        ),
        _make_trajectory_item(
            "p0", 1, prompt_len=4, completion_len=3, reward=0.0
        ),
    ]
    raw_payloads = algo.create_trainer_payloads(
        trajectories, rewards=[1.0, 0.0]
    )
    batches = list(assembler.feed(raw_payloads))
    self.assertNotEmpty(batches)
    real_payload = batch_assembly.with_ref_per_token_logps(
        batches[0].payload,
        np.zeros_like(batches[0].payload.completion_ids, dtype=np.float32),
    )
    if sampler_is == "token" and not can_fuse_agreement_in_loss:
      real_payload = dataclasses.replace(
          real_payload,
          sampler_is_weights=np.ones_like(
              real_payload.completion_ids, dtype=np.float32
          ),
      )

    worker.fwd_bwd(datatypes.TrainRequest(payload=real_payload))
    worker.update()

    self.assertEqual(trainer.compile_kernels_calls, 1)
    self.assertEqual(trainer.train_steps, 1)


if __name__ == "__main__":
  absltest.main()
