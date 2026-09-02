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
        loss_mask=np.ones(GEN_LEN, dtype=np.float32),
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
        group_size=1,
        mini_batch_size=1,
    ).pack([payload])

    routed = packed[0].routed_experts
    self.assertIsNotNone(routed, "routing lost between adapter and payload")
    self.assertEqual(routed.shape, (1, PROMPT_LEN + GEN_LEN, NUM_LAYERS, TOP_K))
    # Nothing was truncated, so every slot should be the captured value.
    np.testing.assert_array_equal(routed[0], 3)


if __name__ == "__main__":
  absltest.main()
