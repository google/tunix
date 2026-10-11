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

"""Tests for the trainer warm compile in `StandardRLProgram`.

The point of the warm compile is that the trainer compiles against a synthetic
microbatch while the first rollouts are still generating, instead of inside its
first `train_step`. That only pays off if the synthetic microbatch has the same
structure as the real ones: a kernel compiled for the wrong shape is not reused,
so the trainer compiles twice and the run is worse off than before.

These tests therefore compare the warm-compile payload against a payload
assembled from rollouts the ordinary way, field by field, for both assemblers.
"""

import asyncio
import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_program
from tunix.rl import algorithm_config


_PROMPT_LEN = 8
_RESPONSE_LEN = 12
_NUM_GENERATIONS = 2
_NUM_LAYERS = 4
_TOP_K = 2


def _signature(payload: datatypes.RLTrainerPayload) -> dict[str, Any]:
  """Returns what a compiled kernel is built against: every field's shape and dtype.

  Values are deliberately excluded. The warm-compile payload holds invented
  tokens; what has to match is the structure, which is what `jax.jit` keys on.
  """
  signature = {}
  for field in dataclasses.fields(payload):
    value = getattr(payload, field.name)
    if value is None:
      signature[field.name] = None
    elif hasattr(value, "shape"):
      signature[field.name] = (tuple(value.shape), np.dtype(value.dtype).name)
    else:
      signature[field.name] = type(value).__name__
  return signature


def _rollout_traj(with_routing: bool = False) -> dict[str, Any]:
  """One rollout of the longest allowed length, as the collector reports it."""
  traj = {
      "prompt_tokens": np.arange(1, _PROMPT_LEN + 1, dtype=np.int32),
      "conversation_tokens": np.arange(1, _RESPONSE_LEN + 1, dtype=np.int32),
      "conversation_masks": np.ones(_RESPONSE_LEN, np.float32),
      "old_logprobs": np.full(_RESPONSE_LEN, -0.5, np.float32),
      "status": "SUCCEEDED",
  }
  if with_routing:
    traj["routed_experts"] = np.ones(
        (_PROMPT_LEN + _RESPONSE_LEN, _NUM_LAYERS, _TOP_K), np.int16
    )
  return traj


class WarmCompilePayloadTest(parameterized.TestCase):
  """The synthetic microbatch must be shaped like the real ones."""

  def _program(
      self,
      algo_config_kwargs: dict[str, Any] | None = None,
      router_replay: bool = False,
      **batch_config_kwargs: Any,
  ) -> rl_program.StandardRLProgram:
    algo = algorithm_adapter.GRPOAdapter(
        algorithm_config.GRPOConfig(
            **{
                "num_generations": _NUM_GENERATIONS,
                "beta": 0.0,
                **(algo_config_kwargs or {}),
            }
        ),
        train_micro_batch_size=_NUM_GENERATIONS,
        max_response_length=_RESPONSE_LEN,
    )
    batch_config = batch_assembly.BatchConfig(
        pad_id=0,
        max_prompt_length=_PROMPT_LEN,
        max_response_length=_RESPONSE_LEN,
        **batch_config_kwargs,
    )
    program = rl_program.StandardRLProgram(
        algo=algo,
        dataset=("prompt_0",),
        max_steps=1,
        batch_config=batch_config,
        generation_args=datatypes.GenerationArgs(
            return_routed_experts=router_replay
        ),
        reward_fns=[lambda *_: 1.0],
    )
    if router_replay:
      # Normally read off the first scored rollout; see `_note_routing_spec`.
      program._routing_spec = ((_NUM_LAYERS, _TOP_K), np.dtype(np.int16))
    return program

  def _real_payload(
      self, program: rl_program.StandardRLProgram
  ) -> datatypes.RLTrainerPayload:
    """Assembles a microbatch the way `train_stage` does, from real rollouts."""
    assembler = program.assembler
    rollouts_needed = program.mini_batch_size * _NUM_GENERATIONS
    payloads = []
    index = 0
    routing = bool(program.generation_args.return_routed_experts)
    while len(payloads) < rollouts_needed:
      group = [
          datatypes.TrajectoryItem(
              prompt_id=f"p{index}",
              group_index=g,
              traj=_rollout_traj(with_routing=routing),
          )
          for g in range(_NUM_GENERATIONS)
      ]
      payloads.extend(
          program.algo.create_trainer_payloads(
              group, rewards=[1.0, 0.0][:_NUM_GENERATIONS]
          )
      )
      index += 1
    batches = assembler.feed(payloads[:rollouts_needed]) + assembler.flush()
    self.assertNotEmpty(batches)
    batch = batches[0].payload
    # The two steps `train_stage` runs between assembly and the trainer.
    if program.algo.requires_reference_kl:
      batch = batch_assembly.with_ref_per_token_logps(
          batch, np.zeros(np.shape(batch.completion_ids), np.float32)
      )
    if program.sampler_is == "token" and batch.old_per_token_logps is not None:
      batch = dataclasses.replace(
          batch,
          sampler_is_weights=np.ones(
              np.shape(batch.completion_mask), np.float32
          ),
      )
    return batch

  @parameterized.named_parameters(
      ("padded", {}, {}),
      (
          "sequence_packed",
          {"max_seq_token_per_tpu": 2 * (_PROMPT_LEN + _RESPONSE_LEN)},
          {},
      ),
      ("reference_kl", {}, {"beta": 0.1}),
      ("truncated_importance_sampling", {}, {"sampler_is": "token"}),
  )
  def test_warm_payload_matches_assembled_payload(
      self, batch_config_kwargs, algo_config_kwargs
  ):
    program = self._program(algo_config_kwargs, **batch_config_kwargs)

    warm = program._build_warm_compile_payload()
    self.assertIsNotNone(warm)
    real = self._real_payload(program)

    self.assertEqual(_signature(warm), _signature(real))

  @parameterized.named_parameters(
      ("padded", {}),
      (
          "sequence_packed",
          {"max_seq_token_per_tpu": 2 * (_PROMPT_LEN + _RESPONSE_LEN)},
      ),
  )
  def test_warm_payload_matches_under_router_replay(self, batch_config_kwargs):
    program = self._program(router_replay=True, **batch_config_kwargs)

    warm = program._build_warm_compile_payload()
    self.assertIsNotNone(warm)
    real = self._real_payload(program)

    # Both sides must actually carry routing, or the comparison below passes
    # for the wrong reason.
    self.assertIsNotNone(warm.routed_experts)
    self.assertIsNotNone(real.routed_experts)
    self.assertEqual(
        np.shape(warm.routed_experts)[-2:], (_NUM_LAYERS, _TOP_K)
    )
    self.assertEqual(_signature(warm), _signature(real))

  def test_declines_until_a_rollout_reports_its_routing(self):
    program = self._program(router_replay=True)
    program._routing_spec = None

    self.assertIsNone(program._build_warm_compile_payload())


  def test_a_single_unrouted_rollout_drops_routing_from_the_batch(self):
    """Pins the assumption the warm payload rests on under router replay.

    `batch_assembly` carries routing only when every row has it, so one
    trajectory without routing makes the microbatch a different kernel from
    the fully routed one the warm compile builds. If this ever stops being
    true, the comment in `_build_warm_compile_payload` is wrong.
    """
    program = self._program(router_replay=True)
    group = [
        datatypes.TrajectoryItem(
            prompt_id="p0",
            group_index=g,
            # The second rollout reports no routing, as an aborted episode does.
            traj=_rollout_traj(with_routing=(g == 0)),
        )
        for g in range(_NUM_GENERATIONS)
    ]
    payloads = program.algo.create_trainer_payloads(
        group, rewards=[1.0, 0.0][:_NUM_GENERATIONS]
    )
    self.assertIsNotNone(payloads[0].routed_experts)
    self.assertIsNone(payloads[1].routed_experts)

    batches = program.assembler.feed(payloads) + program.assembler.flush()
    mixed = batches[0].payload

    self.assertIsNone(mixed.routed_experts)
    warm = program._build_warm_compile_payload()
    self.assertIsNotNone(warm.routed_experts)
    self.assertNotEqual(_signature(warm), _signature(mixed))

  def test_warm_payload_leaves_the_live_assembler_untouched(self):
    """A warm payload must not consume the assembler the real rollouts feed."""
    program = self._program()

    program._build_warm_compile_payload()
    real = self._real_payload(program)

    # If the warm payload had been fed through `program.assembler`, the first
    # real flush would carry its synthetic rollouts.
    self.assertIsNotNone(real)
    self.assertEqual(_signature(real), _signature(self._real_payload(program)))

  def test_skips_without_a_response_length(self):
    program = self._program()
    program.batch_config = dataclasses.replace(
        program.batch_config, max_response_length=None
    )

    self.assertIsNone(program._build_warm_compile_payload())


class RoutingSpecCaptureTest(absltest.TestCase):
  """Reading the routing shape off the rollouts, which is what starts the compile."""

  def _program(self, router_replay: bool = True) -> rl_program.StandardRLProgram:
    algo = algorithm_adapter.GRPOAdapter(
        algorithm_config.GRPOConfig(
            num_generations=_NUM_GENERATIONS, beta=0.0
        ),
        train_micro_batch_size=_NUM_GENERATIONS,
        max_response_length=_RESPONSE_LEN,
    )
    return rl_program.StandardRLProgram(
        algo=algo,
        dataset=("prompt_0",),
        max_steps=1,
        batch_config=batch_assembly.BatchConfig(
            pad_id=0,
            max_prompt_length=_PROMPT_LEN,
            max_response_length=_RESPONSE_LEN,
        ),
        generation_args=datatypes.GenerationArgs(
            return_routed_experts=router_replay
        ),
        reward_fns=[lambda *_: 1.0],
    )

  def _payloads(self, with_routing: bool):
    program = self._program()
    group = [
        datatypes.TrajectoryItem(
            prompt_id="p0",
            group_index=g,
            traj=_rollout_traj(with_routing=with_routing),
        )
        for g in range(_NUM_GENERATIONS)
    ]
    return program, program.algo.create_trainer_payloads(
        group, rewards=[1.0, 0.0][:_NUM_GENERATIONS]
    )

  def test_captures_the_shape_and_starts_the_compile(self):
    program, payloads = self._payloads(with_routing=True)
    program.engine = mock.MagicMock()
    program.engine.warm_compile = mock.AsyncMock()

    async def _run():
      program._note_routing_spec(payloads)
      self.assertIsNotNone(program._warm_compile_task)
      await program._warm_compile_task

    asyncio.run(_run())

    self.assertEqual(program._routing_spec[0], (_NUM_LAYERS, _TOP_K))
    program.engine.warm_compile.assert_awaited_once()
    (payload,), _ = program.engine.warm_compile.await_args
    self.assertEqual(
        np.shape(payload.routed_experts)[-2:], (_NUM_LAYERS, _TOP_K)
    )

  def test_waits_when_the_rollouts_carry_no_routing(self):
    program, payloads = self._payloads(with_routing=False)

    async def _run():
      program._note_routing_spec(payloads)

    asyncio.run(_run())

    self.assertIsNone(program._routing_spec)
    self.assertIsNone(program._warm_compile_task)

  def test_captures_once(self):
    program, payloads = self._payloads(with_routing=True)
    program.engine = mock.MagicMock()
    program.engine.warm_compile = mock.AsyncMock()

    async def _run():
      program._note_routing_spec(payloads)
      first = program._warm_compile_task
      program._note_routing_spec(payloads)
      self.assertIs(program._warm_compile_task, first)
      await first

    asyncio.run(_run())

    program.engine.warm_compile.assert_awaited_once()

class WarmCompileDispatchTest(absltest.TestCase):
  """How the payload reaches the trainer, and what happens when it cannot."""

  def _program(self) -> rl_program.StandardRLProgram:
    algo = algorithm_adapter.GRPOAdapter(
        algorithm_config.GRPOConfig(num_generations=_NUM_GENERATIONS, beta=0.0),
        train_micro_batch_size=_NUM_GENERATIONS,
        max_response_length=_RESPONSE_LEN,
    )
    return rl_program.StandardRLProgram(
        algo=algo,
        dataset=("prompt_0",),
        max_steps=1,
        batch_config=batch_assembly.BatchConfig(
            pad_id=0,
            max_prompt_length=_PROMPT_LEN,
            max_response_length=_RESPONSE_LEN,
        ),
        reward_fns=[lambda *_: 1.0],
    )

  def test_sends_the_payload_to_the_engine(self):
    program = self._program()
    program.engine = mock.MagicMock()
    program.engine.warm_compile = mock.AsyncMock()

    asyncio.run(program._warm_compile())

    program.engine.warm_compile.assert_awaited_once()
    (payload,), _ = program.engine.warm_compile.await_args
    self.assertIsInstance(payload, datatypes.RLTrainerPayload)
    self.assertEqual(
        np.shape(payload.completion_ids)[-1],
        _RESPONSE_LEN,
    )

  def test_engine_failure_is_not_fatal(self):
    program = self._program()
    program.engine = mock.MagicMock()
    program.engine.warm_compile = mock.AsyncMock(
        side_effect=RuntimeError("trainer unreachable")
    )

    asyncio.run(program._warm_compile())  # must not raise

    program.engine.warm_compile.assert_awaited_once()

  def test_engine_without_warm_compile_is_skipped(self):
    program = self._program()
    engine = mock.MagicMock()
    del engine.warm_compile
    program.engine = engine

    asyncio.run(program._warm_compile())  # must not raise


if __name__ == "__main__":
  absltest.main()
