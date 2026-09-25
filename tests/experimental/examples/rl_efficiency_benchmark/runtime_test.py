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

"""Completion tests using real JAX state and the shared episode lifecycle."""

import asyncio
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.experimental.examples.rl_efficiency_benchmark import runtime
from tunix.rl.agentic.trajectory import trajectory_collect_engine


class RuntimeTest(unittest.TestCase):

  def test_real_jax_state_is_ready_after_training_and_sync_waits(self):
    inputs = jnp.ones((1024, 1024), dtype=jnp.float32)
    inputs.block_until_ready()
    compiled = jax.jit(lambda x: x @ x).lower(inputs).compile()
    arrays = [compiled(inputs) for _ in range(3)]
    trainer = types.SimpleNamespace(**{
        name: nnx.Dict(value=nnx.Param(array))
        for name, array in zip(
            ("model", "optimizer", "grad_accumulator"), arrays
        )
    })
    with mock.patch.object(
        jax, "block_until_ready", wraps=jax.block_until_ready
    ) as wait:
      runtime._wait_for_trainer(trainer)
    leaves = jax.tree.leaves(wait.call_args.args[0])
    self.assertEqual({id(x) for x in leaves}, {id(x) for x in arrays})
    self.assertTrue(all(array.is_ready() for array in arrays))

    sampler = types.SimpleNamespace(
        transformer_state={"weight": compiled(inputs)}
    )
    runtime._wait_for_sampler((sampler,), {}, None)
    self.assertTrue(sampler.transformer_state["weight"].is_ready())
    anchor = compiled(inputs)
    engine = types.SimpleNamespace(
        actor_trainer=trainer, _anchor_policy_state={"weight": anchor}
    )
    runtime._wait_for_sync((engine,), {}, None)
    self.assertTrue(anchor.is_ready())
    logps = compiled(inputs)
    runtime._wait_for_result((), {}, logps)
    self.assertTrue(logps.is_ready())

  def test_collection_waits_for_model_reward_and_real_environment_close(self):
    self._check_collection(close_timeout=False)

  def test_swallowed_environment_close_timeout_invalidates_measurement(self):
    self._check_collection(close_timeout=True)

  def _check_collection(self, close_timeout):
    clock = [0.0]
    completed = []

    async def model_call():
      await asyncio.sleep(0)
      clock[0] += 4
      completed.append("generation")
      return types.SimpleNamespace(tokens=[[1]], prompt_lengths=[1])

    async def step(engine):
      self.assertEqual(engine.max_response_length, 512)
      await engine.model_call()
      return True

    async def reward(engine):
      await asyncio.sleep(0)
      clock[0] += 3
      completed.append("reward")

    def close():
      self.assertEqual(completed, ["generation", "reward"])
      clock[0] += 3
      if close_timeout:
        raise TimeoutError("environment cleanup did not complete")
      completed.append("close")

    cls = trajectory_collect_engine.TrajectoryCollectEngine
    env = types.SimpleNamespace(
        close=close, task={"policy_version": 0}, extra_kwargs={}
    )
    engine = cls(
        types.SimpleNamespace(
            trajectory=types.SimpleNamespace(steps=[], reward=1)
        ),
        env,
        model_call=model_call,
        max_response_length=1024,
    )
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "events.jsonl"
      recorder = runtime.Recorder(path, clock=lambda: clock[0])
      # Keep the real collect/finally/_close and executor timing path. Only
      # replace episode work; restore every class hook after the test.
      with mock.patch.object(cls, "collect", cls.collect), mock.patch.object(
          cls, "_run_with_timing", cls._run_with_timing
      ), mock.patch.object(cls, "_reset", mock.AsyncMock()), mock.patch.object(
          cls, "_one_step", step
      ), mock.patch.object(
          cls, "_append_final_reward", reward
      ):
        runtime._install_trajectory_hooks(recorder)
        asyncio.run(engine.collect(mode="Trajectory"))
      self.assertEqual(engine.max_response_length, 1024)
      recorder.close()
      events = [json.loads(line) for line in path.read_text().splitlines()]
    generation, trajectory = events
    self.assertEqual(generation["seconds"], 4)
    self.assertEqual(trajectory["seconds"], 10)
    self.assertEqual(trajectory["ok"], not close_timeout)
    if not close_timeout:
      self.assertEqual(completed, ["generation", "reward", "close"])


if __name__ == "__main__":
  unittest.main()
