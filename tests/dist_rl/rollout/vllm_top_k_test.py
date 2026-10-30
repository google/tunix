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

"""Executes both vLLM parameter boundaries with a strict legacy parameter API.

No vLLM installation or engine is needed. The stand-in rejects top_k=0 as
older vLLM releases do; actual parameter construction and overrides run from
the production sampler modules.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from tunix.common import configs
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.rollout import vllm_sampler as async_sampler


class _LegacySamplingParams(SimpleNamespace):

  def __init__(self, **kwargs):
    super().__init__(top_k=-1, stop=None, n=1, truncate_prompt_tokens=None)
    for key, value in kwargs.items():
      setattr(self, key, value)

  def __setattr__(self, name, value):
    if name == "top_k" and value == 0:
      raise ValueError("Legacy vLLM requires top_k=-1 or a positive integer.")
    super().__setattr__(name, value)


def _module(name, **attributes):
  result = ModuleType(name)
  result.__dict__.update(attributes)
  return result


def _vllm_modules():
  return {
      "vllm": _module("vllm", LLM=object),
      "vllm.engine": _module("vllm.engine"),
      "vllm.engine.arg_utils": _module(
          "vllm.engine.arg_utils", EngineArgs=object
      ),
      "vllm.inputs": _module("vllm.inputs", TokensPrompt=dict),
      "vllm.outputs": _module("vllm.outputs", RequestOutput=SimpleNamespace),
      "vllm.sampling_params": _module(
          "vllm.sampling_params",
          SamplingParams=_LegacySamplingParams,
          BeamSearchParams=type("BeamSearchParams", (), {}),
          RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
      ),
      "tunix.generate.vllm_async_driver": _module(
          "tunix.generate.vllm_async_driver", VLLMInProcessDriver=object
      ),
  }


class VllmTopKTest(unittest.TestCase):

  def test_async_builder_normalizes_all_disabled_sources_before_constructor(
      self,
  ):
    sampler = async_sampler.VllmSampler(
        engine_args=SimpleNamespace(model="unused"),
        config=configs.RolloutConfig(weight_sync_mode="none"),
    )
    with mock.patch.dict(sys.modules, _vllm_modules()):
      for source in ("request", "kwargs", "config"):
        for top_k in (0, -1, 7):
          with self.subTest(source=source, top_k=top_k):
            sampler.config.top_k = top_k if source == "config" else None
            request = sampler_lib.SamplingRequest(
                prompt="q",
                sampling_params=sampler_lib.SamplingParams(
                    top_k=top_k if source == "request" else None
                ),
            )
            params = sampler._build_vllm_params(
                request, {"top_k": top_k} if source == "kwargs" else {}
            )
            self.assertEqual(params.top_k, -1 if top_k in (0, -1) else top_k)

  def test_inprocess_backend_normalizes_call_and_config_overrides(self):
    source = (
        Path(__file__).resolve().parents[3] / "tunix/generate/vllm_sampler.py"
    )
    name = "_tunix_test_vllm_top_k_boundary"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**_vllm_modules(), name: module}):
      spec.loader.exec_module(module)
    for top_k, overrides, expected, unfiltered in (
        (0, {}, -1, True),
        (-1, {}, -1, True),
        (-1, {"top_k": 0}, -1, True),
        (7, {}, 7, False),
    ):
      with self.subTest(top_k=top_k, overrides=overrides):
        sampler = object.__new__(module.VllmSampler)
        sampler.args = {"max_model_len": 32}
        sampler.tokenizer = None
        sampler.config = SimpleNamespace(
            return_logprobs=True, eos_tokens=[1], sampling_kwargs=overrides
        )
        sampler._driver = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(get_diff_sampling_param=lambda: {})
            )
        )

        class Submitted(Exception):
          pass

        sampler._generate_server_mode = mock.Mock(side_effect=Submitted)
        with self.assertRaises(Submitted):
          sampler(
              None,
              4,
              temperature=1.0,
              top_p=1.0,
              top_k=top_k,
              prompt_token_ids=[[1, 2]],
              require_unfiltered=unfiltered,
          )
        params = sampler._generate_server_mode.call_args.args[1]
        self.assertEqual(params.top_k, expected)


if __name__ == "__main__":
  unittest.main()
