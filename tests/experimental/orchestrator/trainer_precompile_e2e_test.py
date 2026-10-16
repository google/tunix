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

"""End-to-end check that `configure_worker` precompiles the trainer.

Drives the exact production sequence against a real `PeftTrainer` on CPU:

1. `LifecycleDriver.bring_up`: `initialize()`, `compile(None)`, `start()`.
2. `DistributedRLEngine.configure_worker`: `with_loss_fn`,
   `with_gen_model_input_fn`, then `compile(assembler.build_dummy_payload())`.
3. First real microbatch from `GRPOAdapter` + the same assembler.

and asserts step 3 triggers neither a new trace nor an XLA backend compile of
any trainer step executable (`fwd_bwd`, `update`, `eval`).
"""

import dataclasses
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np
import optax
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker
from tunix.rl import algorithm_config
from tunix.tests import test_common as tc

try:
  from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top

  MAXTEXT_AVAILABLE = True
except ImportError:
  MAXTEXT_AVAILABLE = False

_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
# Names of the jitted trainer step bodies as they appear in compile events;
# `traced_*` are the wrappers installed by `_build_trainer` below.
_STEP_EXECUTABLES = ("traced_fwd_bwd", "traced_update", "_eval_step")
_MAXTEXT_STEP_EXECUTABLES = ("fwd_bwd", "fwd_bwd_accum", "update", "kernel")


def _rollouts(
    routed_experts_shape: tuple[int, int] | None = None,
) -> list[datatypes.TrajectoryItem]:
  return [
      datatypes.TrajectoryItem(
          traj={
              "prompt_tokens": np.array([1, 2, 3], dtype=np.int32),
              "conversation_tokens": np.array([4, 5], dtype=np.int32),
              "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
              "old_logprobs": np.array([-0.5, -0.5], dtype=np.float32),
              **(
                  {
                      "routed_experts": np.zeros(
                          (5, *routed_experts_shape), dtype=np.int16
                      )
                  }
                  if routed_experts_shape is not None
                  else {}
              ),
          },
          is_valid=True,
      ),
      datatypes.TrajectoryItem(
          traj={
              "prompt_tokens": np.array([1, 2], dtype=np.int32),
              "conversation_tokens": np.array([6, 7, 2], dtype=np.int32),
              "conversation_masks": np.array([1.0, 1.0, 1.0], dtype=np.float32),
              "old_logprobs": np.array([-0.4, -0.4, -0.4], dtype=np.float32),
              **(
                  {
                      "routed_experts": np.zeros(
                          (5, *routed_experts_shape), dtype=np.int16
                      )
                  }
                  if routed_experts_shape is not None
                  else {}
              ),
          },
          is_valid=True,
      ),
  ]


def _make_assembler(kind: str) -> batch_assembly.BatchAssembler[Any]:
  if kind == "padded":
    return batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=3,
        pad_id=0,
        num_generations=2,
        mini_batch_size=1,
    )
  if kind == "packed":
    return batch_assembly.SequencePackedBatchAssembler(
        batch_size=2,
        num_generations=2,
        mini_batch_size=1,
        max_packed_len=8,
        pad_id=0,
        max_segments_per_packed_row=4,
    )
  raise ValueError(f"Unknown assembler kind: {kind}")


@dataclasses.dataclass(kw_only=True)
class _ToyMoEModelConfig(tc.ModelConfig):
  num_experts: int = 4
  num_experts_per_tok: int = 2


class _ToyMoETransformer(tc.ToyTransformer):
  """Toy MoE transformer advertising `forced_routed_experts` in `__call__`."""

  def __call__(
      self,
      x,
      positions,
      cache=None,
      attention_mask=None,
      output_hidden_states=False,
      forced_routed_experts=None,
  ):
    logits, hidden = super().__call__(
        x,
        positions,
        cache=cache,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )
    if forced_routed_experts is not None:
      # Consume `forced_routed_experts` in the forward graph so XLA traces its
      # exact shape and dtype.
      routing_bias = (
          jax.numpy.sum(
              forced_routed_experts.astype(logits.dtype), axis=(-2, -1)
          )[..., None]
          * 0.0
      )
      logits = logits + routing_bias
    return logits, hidden


class TrainerPrecompileE2ETest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # `fun_name` of every XLA backend compile, e.g. "jit(traced_fwd_bwd)".
    self.compiled_fns: list[str] = []

    def _listener(event: str, duration: float, **kwargs: Any) -> None:
      del duration
      if event == _BACKEND_COMPILE_EVENT:
        self.compiled_fns.append(kwargs["fun_name"])

    jax.monitoring.register_event_duration_secs_listener(_listener)
    self.addCleanup(
        jax.monitoring.unregister_event_duration_listener, _listener
    )

  def _compiled_step_executables(
      self, step_names: tuple[str, ...] = _STEP_EXECUTABLES
  ) -> list[str]:
    """Backend compiles of the trainer step bodies (ignores eager one-op jits)."""
    return [
        name
        for name in self.compiled_fns
        if any(step in name for step in step_names)
    ]

  def _build_trainer(
      self,
      routed_experts_shape: tuple[int, int] | None = None,
  ) -> tuple[peft_trainer_v2.PeftTrainer, dict[str, int]]:
    if routed_experts_shape is not None:
      num_layers, top_k = routed_experts_shape
      model = _ToyMoETransformer(
          config=_ToyMoEModelConfig(
              num_layers=num_layers,
              num_experts=4,
              num_experts_per_tok=top_k,
          ),
          rngs=nnx.Rngs(0),
      )
    else:
      model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    trainer = peft_trainer_v2.PeftTrainer(
        model,
        optax.sgd(1e-2),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
            gradient_accumulation_steps=1,
        ),
    )
    # Count Python traces of the two jitted step bodies: a JIT cache hit never
    # re-enters them.
    trace_counts = {"fwd_bwd": 0, "update": 0}
    orig_fwd_bwd = trainer._fwd_bwd_step
    orig_update = trainer._update_step

    def traced_fwd_bwd(*args, **kwargs):
      trace_counts["fwd_bwd"] += 1
      return orig_fwd_bwd(*args, **kwargs)

    def traced_update(*args, **kwargs):
      trace_counts["update"] += 1
      return orig_update(*args, **kwargs)

    trainer._fwd_bwd_step = traced_fwd_bwd
    trainer._update_step = traced_update
    return trainer, trace_counts

  def _build_maxtext_trainer(self) -> tuple[Any, dict[str, int]]:
    base_yml = os.path.join(
        os.path.dirname(os.path.abspath(pyconfig.__file__)), "base.yml"
    )
    num_devices = jax.device_count()
    data_parallelism = min(2, num_devices)
    tensor_parallelism = num_devices // data_parallelism
    cfg_kwargs: dict[str, Any] = {
        "model_name": "qwen3-0.6b",
        "override_model_config": True,
        "base_num_decoder_layers": 1,
        "base_emb_dim": 64,
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "base_mlp_dim": 128,
        "head_dim": 16,
        "vocab_size": 256,
        "attention": "dot_product",
        "ici_data_parallelism": data_parallelism,
        "ici_fsdp_parallelism": 1,
        "ici_tensor_parallelism": tensor_parallelism,
        "run_name": "maxtext_precompile_e2e_test",
        "base_output_directory": self.create_tempdir().full_path,
        "init_weights_seed": 42,
        "micro_batch_size_to_train_on": 2,
        "max_target_length": 8,
        "gradient_accumulation_steps": 1,
        "warmup_steps_fraction": 0.0,
        "learning_rate": 1e-2,
        "enable_dropout": False,
        "record_internal_nn_metrics": False,
        "enable_tensorboard": False,
        "tensorboard_dir": self.create_tempdir().full_path,
        "skip_jax_distributed_system": True,
        "enable_checkpointing": False,
        "profiler_steps": 0,
        "use_weight_converter": False,
        "dtype": "float32",
        "weight_dtype": "float32",
        "grad_dtype": "float32",
    }
    cfg = pyconfig.initialize([None, base_yml], **cfg_kwargs)
    mesh = jax.sharding.Mesh(
        maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes
    )
    engine = maxtext_engine.MaxTextTrainingEngine(
        cfg,
        mesh=mesh,
        wrap_with_tunix_adapter=True,
        tokenizer_pad_id=0,
    )
    trace_counts = {"fwd_bwd": 0, "update": 0}
    orig_fwd_bwd = engine._fwd_bwd_kernel
    orig_update = engine._update_kernel

    def traced_fwd_bwd(*args, **kwargs):
      trace_counts["fwd_bwd"] += 1
      return orig_fwd_bwd(*args, **kwargs)

    def traced_update(*args, **kwargs):
      trace_counts["update"] += 1
      return orig_update(*args, **kwargs)

    engine._fwd_bwd_kernel = traced_fwd_bwd
    engine._update_kernel = traced_update
    return engine, trace_counts

  def _bring_up_and_configure(
      self,
      trainer,
      *,
      assembler_kind: str = "packed",
      can_fuse_agreement_in_loss: bool = True,
      routed_experts_shape: tuple[int, int] | None = None,
      precompile: bool = True,
  ):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: trainer, worker_id="trainer-0"
    )
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )
    registry = worker_registry.WorkerRegistry()
    registry.register_handle("trainer-0", [datatypes.Role.ACTOR], handle)
    engine = distributed_rl_engine.DistributedRLEngine(registry=registry)
    handle.submit("initialize")
    handle.submit("compile", None)
    handle.submit("start")

    algo_cfg = algorithm_config.GRPOConfig(
        num_generations=2,
        beta=0.0,
        use_rollout_logps=True,
        sampler_is="token" if not can_fuse_agreement_in_loss else None,
        routed_experts_encode_mode="int16",
    )
    algo_cfg.temperature = 1.0
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algo_cfg,
        mini_batch_size=1,
        train_micro_batch_size=2,
        max_packed_len=8,
        max_response_length=3,
    )
    assembler = _make_assembler(assembler_kind)
    self.compiled_fns.clear()
    engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo=adapter,
        assembler=assembler,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
        precompile=precompile,
        return_routed_experts=routed_experts_shape is not None,
    )
    return worker, adapter, assembler

  def _run_first_real_step(
      self,
      worker,
      trainer,
      adapter,
      assembler,
      *,
      can_fuse_agreement_in_loss: bool = True,
      routed_experts_shape: tuple[int, int] | None = None,
  ) -> int:
    (assembled,) = list(
        assembler.feed(
            adapter.create_trainer_payloads(
                _rollouts(routed_experts_shape), [1.0, 0.0]
            )
        )
    )
    real_payload = assembled.payload
    if not can_fuse_agreement_in_loss:
      real_payload = dataclasses.replace(
          real_payload,
          sampler_is_weights=np.ones_like(
              real_payload.completion_ids, dtype=np.float32
          ),
          sampler_agreement_applied=True,
      )
    params_before = jax.tree.map(
        np.asarray, nnx.state(trainer.model, nnx.Param)
    )
    self.compiled_fns.clear()
    worker.fwd_bwd(datatypes.TrainRequest(payload=real_payload))
    step = worker.update()
    params_after = nnx.state(trainer.model, nnx.Param)
    self.assertTrue(
        any(
            not np.allclose(a, b)
            for a, b in zip(
                jax.tree.leaves(params_before),
                jax.tree.leaves(params_after),
                strict=True,
            )
        )
    )
    return step

  @parameterized.named_parameters(
      ("padded", "padded", True, None),
      ("packed", "packed", True, None),
      ("packed_non_fused_agreement", "packed", False, None),
      ("padded_moe_routed_experts", "padded", True, (2, 2)),
      ("packed_moe_routed_experts", "packed", True, (2, 2)),
  )
  def test_first_real_step_is_served_from_jit_cache(
      self,
      assembler_kind,
      can_fuse_agreement_in_loss,
      routed_experts_shape,
  ):
    trainer, trace_counts = self._build_trainer(routed_experts_shape)
    worker, adapter, assembler = self._bring_up_and_configure(
        trainer,
        assembler_kind=assembler_kind,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
        routed_experts_shape=routed_experts_shape,
    )

    # `compile(None)` at bring-up is a no-op; `configure_worker` precompiles
    # `fwd_bwd`, `update`, and `eval` once after binding the loss and input fn.
    self.assertEqual(trace_counts, {"fwd_bwd": 1, "update": 1})
    compiled_at_configure = self._compiled_step_executables()
    for step_name in _STEP_EXECUTABLES:
      self.assertTrue(
          any(step_name in name for name in compiled_at_configure),
          f"{step_name} was not precompiled: {compiled_at_configure}",
      )
    self.assertEqual(trainer._train_steps, 0)
    traces_after_configure = dict(trace_counts)

    step = self._run_first_real_step(
        worker,
        trainer,
        adapter,
        assembler,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
        routed_experts_shape=routed_experts_shape,
    )
    self.assertEqual(step, 1)
    self.assertEqual(trace_counts, traces_after_configure)
    self.assertEmpty(self._compiled_step_executables())

  def test_precompile_disabled_skips_configure_compile_and_compiles_on_first_step(
      self,
  ):
    trainer, trace_counts = self._build_trainer()
    worker, adapter, assembler = self._bring_up_and_configure(
        trainer, precompile=False
    )

    # With `precompile=False` on `configure_worker`, `with_loss_fn` and
    # `with_gen_model_input_fn` invalidate the JIT cache and skip AOT compile.
    self.assertEqual(trace_counts, {"fwd_bwd": 0, "update": 0})
    self.assertEmpty(self._compiled_step_executables())

    step = self._run_first_real_step(worker, trainer, adapter, assembler)
    self.assertEqual(step, 1)
    self.assertEqual(trace_counts, {"fwd_bwd": 1, "update": 1})

  @absltest.skipUnless(MAXTEXT_AVAILABLE, "requires MaxText")
  @parameterized.named_parameters(
      ("padded", "padded", True),
      ("packed", "packed", True),
      ("packed_non_fused_agreement", "packed", False),
  )
  def test_maxtext_engine_first_real_step_is_served_from_jit_cache(
      self,
      assembler_kind,
      can_fuse_agreement_in_loss,
  ):
    trainer, trace_counts = self._build_maxtext_trainer()
    worker, adapter, assembler = self._bring_up_and_configure(
        trainer,
        assembler_kind=assembler_kind,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
    )

    # `compile_kernels` lowers both `fwd_bwd` and `fwd_bwd_accum` (2 traces of
    # `_fwd_bwd_kernel`) and `update` (1 trace of `_update_kernel`).
    self.assertEqual(trace_counts, {"fwd_bwd": 2, "update": 1})
    compiled_at_configure = self._compiled_step_executables(
        _MAXTEXT_STEP_EXECUTABLES
    )
    for step_name in _MAXTEXT_STEP_EXECUTABLES:
      self.assertTrue(
          any(step_name in name for name in compiled_at_configure),
          f"{step_name} was not precompiled: {compiled_at_configure}",
      )
    self.assertEqual(trainer.train_step, 0)
    traces_after_configure = dict(trace_counts)

    step = self._run_first_real_step(
        worker,
        trainer,
        adapter,
        assembler,
        can_fuse_agreement_in_loss=can_fuse_agreement_in_loss,
    )
    self.assertEqual(step, 1)
    self.assertEqual(trace_counts, traces_after_configure)
    self.assertEmpty(self._compiled_step_executables(_MAXTEXT_STEP_EXECUTABLES))


if __name__ == "__main__":
  absltest.main()
