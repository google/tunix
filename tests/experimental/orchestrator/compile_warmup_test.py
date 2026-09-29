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

"""Tests for the trainer compile warmup.

The warmup only saves the step-0 compile if its dummy micro-batch has exactly
the structure, shapes and dtypes of the real ones, so these build real
micro-batches the way `StandardRLProgram.train_stage` does and compare.
"""

import asyncio
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import jax
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import compile_warmup
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.worker import trainer_worker
from tunix.rl import algorithm_config

_NUM_GENERATIONS = 4
_MAX_PROMPT = 8
_MAX_RESPONSE = 24
_ROUTED_SHAPE = (3, 2)


def _program(
    *,
    beta: float = 0.0,
    use_rollout_logps: bool = True,
    sampler_is: str | None = None,
    seq_logprob_error_threshold: float | None = None,
    max_segments_per_packed_row: int | None = 3,
    **kwargs: Any,
) -> rl_program.StandardRLProgram:
  algo = algorithm_adapter.GRPOAdapter(
      algorithm_config.GRPOConfig(
          num_generations=_NUM_GENERATIONS,
          beta=beta,
          use_rollout_logps=use_rollout_logps,
          sampler_is=sampler_is,
          seq_logprob_error_threshold=seq_logprob_error_threshold,
      ),
      mini_batch_size=2,
  )
  return rl_program.StandardRLProgram(
      algo=algo,
      generation_args=datatypes.GenerationArgs(temperature=1.0),
      batch_size=2,
      batch_config=batch_assembly.BatchConfig(
          pad_id=0,
          max_prompt_length=_MAX_PROMPT,
          max_response_length=_MAX_RESPONSE,
          max_seq_token_per_tpu=64,
          max_segments_per_packed_row=max_segments_per_packed_row,
          segment_align_multiple=8,
          trainer_fsdp=2,
      ),
      **kwargs,
  )


def _rollout_group(
    prompt_index: int, *, logprobs: bool, routed: bool
) -> list[datatypes.TrajectoryItem]:
  """A group of rollouts of mixed lengths and outcomes, as collectors send."""
  rng = np.random.default_rng(prompt_index)
  statuses = [
      datatypes.TrajectoryStatus.SUCCEEDED.name,
      datatypes.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED.name,
      datatypes.TrajectoryStatus.SUCCEEDED.name,
      datatypes.TrajectoryStatus.TIMEOUT.name,
  ]
  group = []
  for i in range(_NUM_GENERATIONS):
    prompt_len = int(rng.integers(2, _MAX_PROMPT + 1))
    completion_len = int(rng.integers(1, _MAX_RESPONSE + 1))
    traj: dict[str, Any] = {
        "prompt_tokens": rng.integers(1, 100, prompt_len).tolist(),
        "conversation_tokens": rng.integers(1, 100, completion_len).tolist(),
        "conversation_masks": rng.integers(0, 2, completion_len).tolist(),
        "status": statuses[i % len(statuses)],
    }
    if logprobs:
      traj["old_logprobs"] = (-rng.random(completion_len)).tolist()
    if routed:
      # Rollouts report routing for all but the last token.
      traj["routed_experts"] = rng.integers(
          0, 8, (prompt_len + completion_len - 1, *_ROUTED_SHAPE)
      ).astype(np.int32)
    group.append(
        datatypes.TrajectoryItem(
            prompt_id=f"p{prompt_index}", group_index=i, traj=traj
        )
    )
  return group


class _ScoringEngine:
  """Stands in for the engine in the program's sampler-trainer agreement."""

  async def per_token_logps(self, role, items):
    del role
    return datatypes.LogprobsResponse(
        per_token_logps=-np.random.default_rng(0).random(
            np.shape(items.completion_tokens)
        ),
        model_version=0,
    )


def _real_micro_batches(
    program: rl_program.StandardRLProgram, *, logprobs: bool, routed: bool
) -> list[datatypes.RLTrainerPayload]:
  """Micro-batches built as `train_stage` builds them, up to `train_step`."""
  payloads = []
  for prompt_index in range(6):
    payloads.extend(
        program.algo.create_trainer_payloads(
            _rollout_group(prompt_index, logprobs=logprobs, routed=routed),
            rewards=[float(i) for i in range(_NUM_GENERATIONS)],
        )
    )
  program.engine = _ScoringEngine()
  batches = []
  for mb in program.assembler.feed(payloads) + program.assembler.flush():
    batch = mb.payload
    if program.algo.requires_reference_kl:
      batch = batch_assembly.with_ref_per_token_logps(
          batch, -np.ones(np.shape(batch.completion_ids), np.float64)
      )
    if (
        batch.old_per_token_logps is not None
        and program.algo.algo_config.use_rollout_logps
    ):
      batch = asyncio.run(program._apply_sampler_trainer_agreement(batch, {}))
    batches.append(batch)
  return batches


def _warmup_payload(program: rl_program.StandardRLProgram, routed: bool):
  return compile_warmup.build_trainer_warmup_payload(
      algo=program.algo,
      assembler=program.assembler,
      rollout_logprobs=bool(program.generation_args.return_logprobs),
      routed_experts_shape=_ROUTED_SHAPE if routed else None,
      sampler_is=program.sampler_is,
      sampler_is_threshold=program.sampler_is_threshold,
      seq_logprob_error_threshold=program.seq_logprob_error_threshold,
  )


class BuildTrainerWarmupPayloadTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    try:
      import jax._src.monitoring as jax_monitoring  # pyrefly: ignore[import-error]

      jax_monitoring._scalar_listeners.clear()
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  @parameterized.named_parameters(
      # The MLPerf recipes: trainer-recomputed ratio denominator, a sequence
      # gate on the rollout log-probabilities, router replay.
      dict(
          testcase_name="mlperf",
          program_kwargs=dict(
              use_rollout_logps=False, seq_logprob_error_threshold=2.0
          ),
          routed=True,
      ),
      dict(
          testcase_name="reference_kl_and_token_is",
          program_kwargs=dict(beta=0.04, sampler_is="token"),
          routed=False,
      ),
      dict(
          testcase_name="rollout_logps_with_gate",
          program_kwargs=dict(seq_logprob_error_threshold=2.0),
          routed=True,
      ),
      dict(
          testcase_name="no_rollout_logprobs",
          program_kwargs=dict(use_rollout_logps=False),
          routed=False,
      ),
      dict(
          testcase_name="unbounded_segments",
          program_kwargs=dict(max_segments_per_packed_row=None),
          routed=True,
      ),
  )
  def test_matches_every_real_micro_batch(self, program_kwargs, routed):
    program = _program(**program_kwargs)
    dummy = _warmup_payload(program, routed)
    self.assertIsNotNone(dummy)
    expected = trainer_worker.payload_signature(dummy)

    real = _real_micro_batches(
        program,
        logprobs=bool(program.generation_args.return_logprobs),
        routed=routed,
    )

    self.assertGreater(len(real), 1)
    for batch in real:
      actual = trainer_worker.payload_signature(batch)
      self.assertEqual(trainer_worker.signature_mismatch(expected, actual), "")
      self.assertEqual(expected, actual)

  def test_leaves_are_abstract_and_routing_is_widened(self):
    program = _program()
    dummy = _warmup_payload(program, routed=True)

    self.assertEqual(dummy.metadata, {})
    for leaf in jax.tree.leaves(dummy):
      self.assertIsInstance(leaf, jax.ShapeDtypeStruct)
    rows, length = dummy.completion_ids.shape
    self.assertEqual((rows, length), (2, 64))
    self.assertEqual(
        dummy.routed_experts.shape, (rows, length, *_ROUTED_SHAPE)
    )
    self.assertEqual(dummy.routed_experts.dtype, np.int16)
    self.assertEqual(dummy.num_segments, 4)

  def test_leaves_the_program_assembler_untouched(self):
    program = _program()
    _warmup_payload(program, routed=True)

    self.assertEqual(program.assembler.flush(), [])
    self.assertEqual(program.assembler._batch_counter, 0)

  def test_survives_the_rpc_round_trip(self):
    program = _program(seq_logprob_error_threshold=2.0)
    dummy = _warmup_payload(program, routed=True)

    restored = cloudpickle.loads(cloudpickle.dumps(dummy))

    self.assertEqual(
        trainer_worker.payload_signature(restored),
        trainer_worker.payload_signature(dummy),
    )

  def test_returns_none_for_padded_assembler(self):
    program = _program()
    padded = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=_MAX_PROMPT,
        max_response_length=_MAX_RESPONSE,
        pad_id=0,
        num_generations=_NUM_GENERATIONS,
        mini_batch_size=2,
    )

    self.assertIsNone(
        compile_warmup.build_trainer_warmup_payload(
            algo=program.algo, assembler=padded, rollout_logprobs=True
        )
    )

  def test_mismatch_names_the_differing_fields(self):
    program = _program()
    with_routing = trainer_worker.payload_signature(
        _warmup_payload(program, routed=True)
    )
    without_routing = trainer_worker.payload_signature(
        _warmup_payload(program, routed=False)
    )

    mismatch = trainer_worker.signature_mismatch(with_routing, without_routing)

    self.assertIn("routed_experts", mismatch)
    self.assertNotIn("completion_ids", mismatch)

  def test_describe_payload(self):
    program = _program()
    description = compile_warmup.describe_payload(
        _warmup_payload(program, routed=True)
    )

    self.assertIn("completion_ids[2,64] int32", description)
    self.assertIn("routed_experts[2,64,3,2] int16", description)
    self.assertIn("num_segments=4", description)


class _WarmupEngine:

  def __init__(self, routed_shape=_ROUTED_SHAPE, error=None):
    self.warmups = []
    self.routed_shape = routed_shape
    self.error = error

  async def warmup_trainer_compile(self, dummy_data, role):
    del role
    if self.error is not None:
      raise self.error
    self.warmups.append(dummy_data)
    return {"warmup_compile_started": True}

  async def trainer_routed_experts_shape(self, role):
    del role
    return self.routed_shape


class ProgramWarmupTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    try:
      import jax._src.monitoring as jax_monitoring  # pyrefly: ignore[import-error]

      jax_monitoring._scalar_listeners.clear()
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def test_sends_routed_dummy_batch(self):
    program = _program(
        trainer_compile_warmup=True, trainer_warmup_routed_experts=True
    )
    program.engine = _WarmupEngine()

    asyncio.run(program._warmup_trainer_compile())

    (dummy,) = program.engine.warmups
    self.assertEqual(dummy.routed_experts.shape, (2, 64, *_ROUTED_SHAPE))

  def test_skips_routing_when_trainer_reports_none(self):
    program = _program(
        trainer_compile_warmup=True, trainer_warmup_routed_experts=True
    )
    program.engine = _WarmupEngine(routed_shape=None)

    asyncio.run(program._warmup_trainer_compile())

    (dummy,) = program.engine.warmups
    self.assertIsNone(dummy.routed_experts)

  def test_failures_do_not_propagate(self):
    program = _program(trainer_compile_warmup=True)
    program.engine = _WarmupEngine(error=RuntimeError("worker gone"))

    asyncio.run(program._warmup_trainer_compile())

    program.engine = object()  # No warmup_trainer_compile at all.
    asyncio.run(program._warmup_trainer_compile())

  def _run_with_stub_stages(self, program) -> list[str]:
    started = []

    def _stage(name, delay=0.0):
      async def _run(*args, **kwargs):
        del args, kwargs
        started.append(name)
        await asyncio.sleep(delay)

      return _run

    program._resume_from_checkpoint = _stage("resume")
    program.rollout_dispatch_stage = _stage("dispatch", 0.05)
    program.polling_stage = _stage("polling", 0.05)
    program.critique_stage = _stage("critique", 0.05)
    program.train_stage = _stage("train", 0.05)
    program._warmup_trainer_compile = _stage("warmup")
    engine = mock.MagicMock()
    engine.prepare_rollout_policy = mock.AsyncMock(return_value=0)
    asyncio.run(program.run_async(engine=engine))
    return started

  def test_run_starts_warmup_after_the_stages(self):
    program = _program(trainer_compile_warmup=True)

    started = self._run_with_stub_stages(program)

    self.assertEqual(
        started,
        ["resume", "train", "dispatch", "polling", "critique", "warmup"],
    )

  def test_run_without_flag_does_not_warm_up(self):
    program = _program()

    started = self._run_with_stub_stages(program)

    self.assertNotIn("warmup", started)


if __name__ == "__main__":
  absltest.main()
