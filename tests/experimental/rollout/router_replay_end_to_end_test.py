# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout MoE routing, from vLLM's output object into a trainer payload.

Drives the real `VllmSampler` with a stubbed vLLM engine that returns genuine
`vllm.outputs.CompletionOutput` objects carrying `routed_experts`, so this
half of the chain runs on the exact type vLLM produces:

    CompletionOutput -> VllmSampler -> SamplerOutput -> sampler adapter
      -> RLTrainerPayload -> batch assembler

The other half -- payload to model -- is shared with the non-experimental
stack and lives in `tests/rl/router_replay_maxtext_test.py`.

Stubbing the engine keeps this a CPU test with no checkpoint and no TPU. What
it therefore does NOT cover is tpu-inference's own capture kernel, which fills
`routed_experts` in the first place; everything downstream of that is real.
"""

import asyncio
from unittest import mock

from absl.testing import absltest
import jax
from jax.sharding import Mesh
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.generate import base_sampler
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout

try:
  from vllm.outputs import CompletionOutput, RequestOutput

  from tunix.generate import vllm_sampler

  VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
  VLLM_AVAILABLE = False

PROMPT_LEN = 4
GEN_LEN = 4
NUM_LAYERS = 2
TOP_K = 2
NUM_EXPERTS = 4


class _StubTokenizer:
  """Minimal tokenizer surface used by VllmSampler."""

  pad_token_id = 0
  eos_token_id = 2

  def encode(self, text):
    del text
    return list(range(10, 10 + PROMPT_LEN))

  def decode(self, ids):
    return " ".join(str(int(i)) for i in ids)

  def bos_id(self):
    return None

  def eos_id(self):
    return 2

  def pad_id(self):
    return 0

  def dedup_bos_ids(self, ids):
    return ids


def _routing(length, fill):
  return np.full((length, NUM_LAYERS, TOP_K), fill, dtype=np.int32)


def _request_output(routed_experts):
  """A genuine vLLM RequestOutput carrying captured routing."""
  completion = CompletionOutput(
      index=0,
      text="answer",
      token_ids=list(range(20, 20 + GEN_LEN)),
      cumulative_logprob=None,
      logprobs=None,
      routed_experts=routed_experts,
      finish_reason="stop",
  )
  return RequestOutput(
      request_id="req-0",
      prompt="q",
      prompt_token_ids=list(range(10, 10 + PROMPT_LEN)),
      prompt_logprobs=None,
      outputs=[completion],
      finished=True,
  )


@absltest.skipUnless(VLLM_AVAILABLE, "requires vLLM")
class VllmSamplerRoutingTest(absltest.TestCase):
  """`VllmSampler` must surface what vLLM put on the CompletionOutput."""

  def _sampler(self, routed_experts, return_routed_experts=True):
    config = vllm_sampler.VllmConfig(
        return_logprobs=False,
        return_routed_experts=return_routed_experts,
        mesh=Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")),
        tensor_parallel_size=1,
        data_parallel_size=1,
        engine_kwargs={"model": "stub", "max_model_len": 64},
    )
    with mock.patch.object(vllm_sampler, "LLM") as llm_cls:
      llm = llm_cls.return_value
      llm.get_default_sampling_params.return_value = (
          vllm_sampler.SamplingParams()
      )
      llm.generate.return_value = [_request_output(routed_experts)]
      sampler = vllm_sampler.VllmSampler(
          tokenizer=_StubTokenizer(), config=config
      )
    return sampler

  def test_routing_survives_the_sampler(self):
    """The flag must reach vLLM's engine args, and routing must come back."""
    routed = _routing(PROMPT_LEN + GEN_LEN, 3)
    sampler = self._sampler(routed)
    self.assertTrue(sampler.args.get("enable_return_routed_experts"))

    out = sampler(input_strings=["q"], max_generation_steps=GEN_LEN)
    self.assertIsNotNone(out.routed_experts, "sampler dropped the routing")
    np.testing.assert_array_equal(np.asarray(out.routed_experts[0]), routed)

  def test_routing_withheld_when_not_requested(self):
    """Opt-in: capture off means the trainer sees a normal dense-style step."""
    sampler = self._sampler(
        _routing(PROMPT_LEN + GEN_LEN, 3), return_routed_experts=False
    )
    self.assertNotIn("enable_return_routed_experts", sampler.args)

    out = sampler(input_strings=["q"], max_generation_steps=GEN_LEN)
    self.assertIsNone(out.routed_experts)


class SamplerToPayloadTest(absltest.TestCase):
  """Routing must reach a batched RLTrainerPayload with its layout intact."""

  def _sampling_response(self, fill):
    stub = mock.MagicMock()
    stub.config = mock.Mock(return_routed_experts=True)
    stub.return_value = base_sampler.SamplerOutput(
        text=["answer"],
        logits=None,
        tokens=[np.arange(20, 20 + GEN_LEN, dtype=np.int32)],
        padded_prompt_tokens=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32)[
            None, :
        ],
        logprobs=None,
        routed_experts=[_routing(PROMPT_LEN + GEN_LEN, fill)],
    )

    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="rollout"
    )
    adapter.vllm_sampler = stub
    request = base_sampler_lib.SamplingRequest(
        request_id="req-0",
        prompt=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=GEN_LEN, return_routed_experts=True
        ),
    )
    return asyncio.run(adapter.sample([request]))[0]

  def test_chain_reaches_a_batched_payload(self):
    response = self._sampling_response(fill=3)
    self.assertIsNotNone(response.routed_experts)

    payload = datatypes.RLTrainerPayload(
        advantages=np.zeros(GEN_LEN, dtype=np.float32),
        prompt_ids=np.asarray(response.prompt_token_ids, dtype=np.int32),
        prompt_mask=np.ones(PROMPT_LEN, dtype=np.float32),
        completion_ids=np.asarray(response.token_ids, dtype=np.int32),
        completion_mask=np.ones(GEN_LEN, dtype=np.float32),
        routed_experts=np.asarray(response.routed_experts),
    )
    packed = batch_assembly.PaddedBatchAssembler(
        batch_size=1,
        max_prompt_length=PROMPT_LEN,
        max_response_length=GEN_LEN,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
    ).pack([payload])

    routed = packed[0].routed_experts
    self.assertIsNotNone(routed, "routing lost between adapter and payload")
    self.assertEqual(routed.shape, (1, PROMPT_LEN + GEN_LEN, NUM_LAYERS, TOP_K))
    # Nothing was truncated, so every slot should be the captured value.
    np.testing.assert_array_equal(routed[0], 3)

  def test_multi_turn_routed_experts_prompt_start_to_payload(self):
    """Multi-turn rollout via routed_experts_prompt_start yields unpadded bit-exact payload."""
    stub = mock.MagicMock()
    stub.config = mock.Mock(return_routed_experts=True)

    env_len = 3
    turn0_routing = _routing(PROMPT_LEN + GEN_LEN, 3)
    turn1_routing = _routing(env_len + GEN_LEN, 7)

    stub.side_effect = [
        base_sampler.SamplerOutput(
            text=["asst0"],
            logits=None,
            tokens=[np.arange(20, 20 + GEN_LEN, dtype=np.int32)],
            padded_prompt_tokens=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32)[
                None, :
            ],
            logprobs=None,
            routed_experts=[turn0_routing],
        ),
        base_sampler.SamplerOutput(
            text=["asst1"],
            logits=None,
            tokens=[np.arange(30, 30 + GEN_LEN, dtype=np.int32)],
            padded_prompt_tokens=np.arange(
                10, 10 + PROMPT_LEN + GEN_LEN + env_len, dtype=np.int32
            )[None, :],
            logprobs=None,
            routed_experts=[turn1_routing],
        ),
    ]

    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="rollout"
    )
    adapter.vllm_sampler = stub

    req0 = base_sampler_lib.SamplingRequest(
        request_id="req-0",
        prompt=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=GEN_LEN,
            return_routed_experts=True,
            routed_experts_prompt_start=0,
        ),
    )
    resp0 = asyncio.run(adapter.sample([req0]))[0]

    req1 = base_sampler_lib.SamplingRequest(
        request_id="req-1",
        prompt=np.arange(
            10, 10 + PROMPT_LEN + GEN_LEN + env_len, dtype=np.int32
        ),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=GEN_LEN,
            return_routed_experts=True,
            routed_experts_prompt_start=PROMPT_LEN + GEN_LEN,
        ),
    )
    resp1 = asyncio.run(adapter.sample([req1]))[0]

    self.assertEqual(
        stub.call_args_list[1].kwargs.get("routed_experts_prompt_start"),
        [PROMPT_LEN + GEN_LEN],
    )

    total_len = PROMPT_LEN + GEN_LEN + env_len + GEN_LEN
    full_routed = np.concatenate(
        [
            resp0.routed_experts,
            resp1.routed_experts,
        ],
        axis=0,
    )
    self.assertEqual(full_routed.shape, (total_len, NUM_LAYERS, TOP_K))
    np.testing.assert_array_equal(full_routed[: PROMPT_LEN + GEN_LEN], 3)
    np.testing.assert_array_equal(full_routed[PROMPT_LEN + GEN_LEN :], 7)

    payload = datatypes.RLTrainerPayload(
        advantages=np.zeros(GEN_LEN + env_len + GEN_LEN, dtype=np.float32),
        prompt_ids=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32),
        prompt_mask=np.ones(PROMPT_LEN, dtype=np.float32),
        completion_ids=np.arange(
            20, 20 + GEN_LEN + env_len + GEN_LEN, dtype=np.int32
        ),
        completion_mask=np.ones(GEN_LEN + env_len + GEN_LEN, dtype=np.float32),
        routed_experts=full_routed,
    )
    packed = batch_assembly.PaddedBatchAssembler(
        batch_size=1,
        max_prompt_length=PROMPT_LEN,
        max_response_length=GEN_LEN + env_len + GEN_LEN,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
    ).pack([payload])

    routed = packed[0].routed_experts
    self.assertEqual(routed.shape, (1, total_len, NUM_LAYERS, TOP_K))
    np.testing.assert_array_equal(routed[0, : PROMPT_LEN + GEN_LEN], 3)
    np.testing.assert_array_equal(routed[0, PROMPT_LEN + GEN_LEN :], 7)
    self.assertFalse((routed == -1).any())

  def test_heterogeneous_batch_routed_experts_prompt_start(self):
    """Batched requests with heterogeneous prompt_start offsets do not collapse or truncate each other."""
    stub = mock.MagicMock()
    stub.config = mock.Mock(return_routed_experts=True)

    env_len = 3
    # req_a is Turn 0 (prompt_start=0): 8 tokens total
    # req_b is Turn 1 (prompt_start=8): 7 tokens total
    routing_a = _routing(PROMPT_LEN + GEN_LEN, 3)
    routing_b = _routing(env_len + GEN_LEN, 7)

    stub.return_value = base_sampler.SamplerOutput(
        text=["asst_a", "asst_b"],
        logits=None,
        tokens=[
            np.arange(20, 20 + GEN_LEN, dtype=np.int32),
            np.arange(30, 30 + GEN_LEN, dtype=np.int32),
        ],
        padded_prompt_tokens=np.zeros((2, 16), dtype=np.int32),
        logprobs=None,
        routed_experts=[routing_a, routing_b],
    )

    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="rollout"
    )
    adapter.vllm_sampler = stub

    req_a = base_sampler_lib.SamplingRequest(
        request_id="req-a",
        prompt=np.arange(10, 10 + PROMPT_LEN, dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=GEN_LEN,
            return_routed_experts=True,
            routed_experts_prompt_start=0,
        ),
    )
    req_b = base_sampler_lib.SamplingRequest(
        request_id="req-b",
        prompt=np.arange(
            10, 10 + PROMPT_LEN + GEN_LEN + env_len, dtype=np.int32
        ),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=GEN_LEN,
            return_routed_experts=True,
            routed_experts_prompt_start=PROMPT_LEN + GEN_LEN,
        ),
    )

    # Concurrently sample the batch in a single call
    responses = asyncio.run(adapter.sample([req_a, req_b]))
    self.assertEqual(len(responses), 2)

    # Verify heterogeneous offsets were passed as a sequence, not collapsed via max()
    forwarded_starts = stub.call_args.kwargs.get("routed_experts_prompt_start")
    self.assertEqual(forwarded_starts, [0, PROMPT_LEN + GEN_LEN])

    # Verify both responses retain their distinct routing lengths and values
    np.testing.assert_array_equal(responses[0].routed_experts, routing_a)
    np.testing.assert_array_equal(responses[1].routed_experts, routing_b)

  @absltest.skipUnless(VLLM_AVAILABLE, "requires vLLM")
  @mock.patch.object(agentic_utils, "tokenize_and_generate_masks")
  def test_full_multi_turn_vllm_to_trajectory_engine_to_batch_assembler(
      self, mock_tokenize_masks
  ):
    """End-to-end 2-turn rollout: CompletionOutput -> VllmSampler -> Adapter -> Engine -> BatchAssembler."""
    env_len = 3
    prompt0_ids = list(range(10, 10 + PROMPT_LEN))
    gen0_ids = list(range(20, 20 + GEN_LEN))
    prompt1_ids = list(range(10, 10 + PROMPT_LEN + GEN_LEN + env_len))
    gen1_ids = list(range(40, 40 + GEN_LEN))

    mock_tokenize_masks.side_effect = [
        (prompt0_ids, [1] * PROMPT_LEN),
        (list(range(30, 30 + env_len)), [1] * env_len),
    ]

    # Turn 0: vLLM routes P_0 + G_0 - 1 tokens (4 prompt + 3 generated = 7 tokens)
    turn0_routed = np.concatenate(
        [_routing(PROMPT_LEN, 3), _routing(GEN_LEN - 1, 5)], axis=0
    )
    req_out0 = RequestOutput(
        request_id="req-0",
        prompt="q0",
        prompt_token_ids=prompt0_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="asst0",
                token_ids=gen0_ids,
                cumulative_logprob=None,
                logprobs=None,
                routed_experts=turn0_routed,
                finish_reason="stop",
            )
        ],
        finished=True,
    )

    # Turn 1: vLLM prefill starts at offset 7 (4 prompt + 3 gen).
    # Prefill routes: 1 delayed assistant token (fill 5) + 3 env tokens (fill 6).
    # Decode routes: GEN_LEN - 1 = 3 tokens (fill 7). Total = 7 tokens.
    turn1_routed = np.concatenate(
        [_routing(1, 5), _routing(env_len, 6), _routing(GEN_LEN - 1, 7)], axis=0
    )
    req_out1 = RequestOutput(
        request_id="req-1",
        prompt="q1",
        prompt_token_ids=prompt1_ids,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="asst1",
                token_ids=gen1_ids,
                cumulative_logprob=None,
                logprobs=None,
                routed_experts=turn1_routed,
                finish_reason="stop",
            )
        ],
        finished=True,
    )

    config = vllm_sampler.VllmConfig(
        return_logprobs=False,
        return_routed_experts=True,
        mesh=Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")),
        tensor_parallel_size=1,
        data_parallel_size=1,
        engine_kwargs={"model": "stub", "max_model_len": 64},
    )
    with mock.patch.object(vllm_sampler, "LLM") as llm_cls:
      llm = llm_cls.return_value
      llm.get_default_sampling_params.side_effect = (
          lambda: vllm_sampler.SamplingParams()
      )
      llm.generate.side_effect = [[req_out0], [req_out1]]
      sampler = vllm_sampler.VllmSampler(
          tokenizer=_StubTokenizer(), config=config
      )

    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="rollout"
    )
    adapter.vllm_sampler = sampler

    mock_agent = mock.MagicMock()
    mock_agent.trajectory = agent_types.Trajectory()
    mock_agent.chat_completions = [{"role": "user", "content": "initial_obs"}]

    def _update_from_model(text):
      step = agent_types.Step(model_response=text)
      mock_agent.trajectory.steps.append(step)
      mock_agent.chat_completions.append({"role": "assistant", "content": text})
      return mock.Mock(action="act")

    def _update_from_env(obs, rew, done, info):
      del info
      if mock_agent.trajectory.steps:
        mock_agent.trajectory.steps[-1].observation = obs
        mock_agent.trajectory.steps[-1].reward = rew
        mock_agent.trajectory.steps[-1].done = done
      mock_agent.chat_completions.append({"role": "user", "content": str(obs)})

    mock_agent.update_from_model.side_effect = _update_from_model
    mock_agent.update_from_env.side_effect = _update_from_env
    mock_agent.get_current_step.side_effect = (
        lambda: mock_agent.trajectory.steps[-1]
        if mock_agent.trajectory.steps
        else None
    )

    mock_env = mock.MagicMock()
    mock_env.max_steps = 2
    mock_env.reset.return_value = ("initial_obs", {})
    mock_env.step.side_effect = [
        ("env_obs_1", 1.0, False, {}),
        ("env_obs_2", 1.0, True, {}),
    ]
    mock_env.task = {}
    mock_env.extra_kwargs = {}

    mock_chat_parser = mock.MagicMock()
    mock_chat_parser.update_assistant_end_tokens.side_effect = (
        lambda tokens: (tokens, 0)
    )

    async def _model_call(
        chat_completions, env=None, max_generation_steps=GEN_LEN, **kwargs
    ):
      del chat_completions, env
      req = base_sampler_lib.SamplingRequest(
          request_id="req",
          prompt=np.array(prompt0_ids, dtype=np.int32),
          sampling_params=base_sampler_lib.SamplingParams(
              max_tokens=max_generation_steps,
              return_routed_experts=True,
              routed_experts_prompt_start=kwargs.get(
                  "routed_experts_prompt_start", 0
              ),
          ),
      )
      resp = (await adapter.sample([req]))[0]
      return base_rollout.RolloutOutput(
          text=[resp.text],
          logits=None,
          tokens=[np.asarray(resp.output_tokens, dtype=np.int32)],
          left_padded_prompt_tokens=np.asarray([prompt0_ids], dtype=np.int32),
          logprobs=[np.zeros(GEN_LEN, dtype=np.float32)],
          routed_experts=[resp.routed_experts],
      )

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=mock_agent,
        env=mock_env,
        model_call=_model_call,
        tokenizer=_StubTokenizer(),
        chat_parser=mock_chat_parser,
    )

    token_data = asyncio.run(engine.collect(mode="Token"))

    # Verify vLLM received an int routed_experts_prompt_start=7 on Turn 1
    turn1_params = llm.generate.call_args_list[1].kwargs["sampling_params"]
    if isinstance(turn1_params, list):
      turn1_params = turn1_params[0]
    self.assertEqual(turn1_params.routed_experts_prompt_start, 7)
    self.assertEqual(token_data["routed_experts"].dtype, np.int16)

    # Pack into PaddedBatchAssembler with 2 left-pad prompt tokens and 2 right-pad response tokens
    max_prompt_len = PROMPT_LEN + 2  # 6
    max_resp_len = GEN_LEN + env_len + GEN_LEN + 2  # 13
    payload = datatypes.RLTrainerPayload(
        advantages=np.zeros(GEN_LEN + env_len + GEN_LEN, dtype=np.float32),
        prompt_ids=np.asarray(token_data["prompt_tokens"], dtype=np.int32),
        prompt_mask=np.ones(PROMPT_LEN, dtype=np.float32),
        completion_ids=np.asarray(
            token_data["conversation_tokens"], dtype=np.int32
        ),
        completion_mask=np.asarray(
            token_data["conversation_masks"], dtype=np.float32
        ),
        routed_experts=token_data["routed_experts"],
    )
    packed = batch_assembly.PaddedBatchAssembler(
        batch_size=1,
        max_prompt_length=max_prompt_len,
        max_response_length=max_resp_len,
        pad_id=0,
        num_generations=1,
        mini_batch_size=1,
    ).pack([payload])

    routed = packed[0].routed_experts
    self.assertEqual(routed.dtype, np.int16)
    self.assertEqual(
        routed.shape, (1, max_prompt_len + max_resp_len, NUM_LAYERS, TOP_K)
    )
    # [0, :2] -> left prompt padding (-1)
    np.testing.assert_array_equal(routed[0, :2], -1)
    # [0, 2:6] -> real prompt tokens (fill 3)
    np.testing.assert_array_equal(routed[0, 2:6], 3)
    # [0, 6:10] -> Turn 0 assistant tokens (all 4 tokens fill 5, including stitched token)
    np.testing.assert_array_equal(routed[0, 6:10], 5)
    # [0, 10:13] -> Turn 0 env tokens (all 3 tokens fill 6)
    np.testing.assert_array_equal(routed[0, 10:13], 6)
    # [0, 13:16] -> Turn 1 assistant tokens 0..2 (fill 7)
    np.testing.assert_array_equal(routed[0, 13:16], 7)
    # [0, 16] -> Turn 1 terminal assistant token (-1)
    np.testing.assert_array_equal(routed[0, 16], -1)
    # [0, 17:19] -> right response padding (-1)
    np.testing.assert_array_equal(routed[0, 17:19], -1)


if __name__ == "__main__":
  absltest.main()
