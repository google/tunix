import sys
import types
import unittest
from unittest import mock


class _FakeSamplingParams:
    def __init__(self):
        self.detokenize = False
        self.max_tokens = 0
        self.n = 1
        self.temperature = 0.0
        self.logprobs = 0
        self.prompt_logprobs = 0
        self.stop_token_ids = []
        self.skip_special_tokens = True
        self.output_kind = None

    @classmethod
    def from_optional(cls, **_kwargs):
        return cls()


class _FakeBeamSearchParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeRequestOutputKind:
    FINAL_ONLY = 1


class _FakeCompletion:
    def __init__(self, token_ids, logprobs):
        self.token_ids = token_ids
        self.logprobs = logprobs


class _FakeTokenLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _FakeRequestOutput:
    def __init__(self, request_id="", outputs=None, finished=True):
        self.request_id = request_id
        self.outputs = outputs or []
        self.finished = finished


class _FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class _FakeEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAsyncEngineArgs(_FakeEngineArgs):
    pass


class _FakeLLM:
    def __init__(self, **_kwargs):
        self._default = _FakeSamplingParams()

    def get_default_sampling_params(self):
        return self._default

    def generate(self, **_kwargs):
        return []

    def shutdown(self):
        return None


class _FakeAsyncLLM:
    instances = []

    def __init__(self):
        self.request_ids = []
        self.default_sampling_params = _FakeSamplingParams()
        self.return_empty = False
        self.raise_in_generate = None

    @classmethod
    def from_engine_args(cls, *_args, **_kwargs):
        inst = cls()
        cls.instances.append(inst)
        return inst

    def get_default_sampling_params(self):
        return self.default_sampling_params

    async def generate(self, prompt, _sampling_params, request_id):
        self.request_ids.append(request_id)
        if self.raise_in_generate is not None:
            raise self.raise_in_generate
        if self.return_empty:
            return
            yield  # pragma: no cover
        base_tok = prompt.prompt_token_ids[-1] if prompt.prompt_token_ids else 7
        # Include eos token (2) so `detokenize` path trims it.
        outputs = [
            _FakeCompletion(
                [base_tok, 2],
                [
                    {base_tok: _FakeTokenLogprob(-0.1)},
                    {2: _FakeTokenLogprob(-0.01)},
                ],
            )
        ]
        yield _FakeRequestOutput(request_id=request_id, outputs=outputs, finished=True)

    def shutdown(self):
        return None


class _FakeDriver:
    @classmethod
    def from_engine_args(cls, *_args, **_kwargs):
        return cls()

    def shutdown(self):
        return None


# Install stubs before importing sampler.
_vllm = types.ModuleType("vllm")
_vllm.LLM = _FakeLLM
sys.modules["vllm"] = _vllm

_vllm_arg_utils = types.ModuleType("vllm.engine.arg_utils")
_vllm_arg_utils.EngineArgs = _FakeEngineArgs
_vllm_arg_utils.AsyncEngineArgs = _FakeAsyncEngineArgs
sys.modules["vllm.engine.arg_utils"] = _vllm_arg_utils

_vllm_async_llm = types.ModuleType("vllm.v1.engine.async_llm")
_vllm_async_llm.AsyncLLM = _FakeAsyncLLM
sys.modules["vllm.v1.engine.async_llm"] = _vllm_async_llm

_vllm_inputs = types.ModuleType("vllm.inputs")
_vllm_inputs.TokensPrompt = _FakeTokensPrompt
sys.modules["vllm.inputs"] = _vllm_inputs

_vllm_outputs = types.ModuleType("vllm.outputs")
_vllm_outputs.RequestOutput = _FakeRequestOutput
sys.modules["vllm.outputs"] = _vllm_outputs

_vllm_sampling = types.ModuleType("vllm.sampling_params")
_vllm_sampling.BeamSearchParams = _FakeBeamSearchParams
_vllm_sampling.SamplingParams = _FakeSamplingParams
_vllm_sampling.RequestOutputKind = _FakeRequestOutputKind
sys.modules["vllm.sampling_params"] = _vllm_sampling

_fake_driver_mod = types.ModuleType("tunix.generate.vllm_async_driver")
_fake_driver_mod.VLLMInProcessDriver = _FakeDriver
sys.modules["tunix.generate.vllm_async_driver"] = _fake_driver_mod

from tunix.generate.vllm_sampler import VllmConfig, VllmSampler


class _DummyTokenizer:
    def encode(self, text):
        return [ord(text[0]) % 13 + 3]

    def bos_id(self):
        return None

    def dedup_bos_ids(self, ids):
        return ids

    def pad_id(self):
        return 0

    def eos_id(self):
        return 2

    def decode(self, ids):
        return f"decoded:{','.join(str(i) for i in ids)}"


class VllmAsyncLLMTests(unittest.TestCase):

    def setUp(self):
        super().setUp()
        _FakeAsyncLLM.instances.clear()
        self._samplers = []

    def tearDown(self):
        for sampler in self._samplers:
            try:
                sampler.stop()
            except Exception:
                pass
        self._samplers.clear()
        super().tearDown()

    def _make_async_sampler(self, **cfg_overrides):
        cfg = VllmConfig(
                server_mode=True,
                use_async_llm_inproc=True,
                enable_log_stats_loop=False,
                init_with_random_weights=False,
        )
        for key, value in cfg_overrides.items():
            setattr(cfg, key, value)

        with (
                mock.patch.object(
                        VllmSampler,
                        "_vllm_config",
                        return_value={
                                "max_model_len": 64,
                                "additional_config": {},
                                "gpu_memory_utilization": 0.5,
                                "tensor_parallel_size": 1,
                                "data_parallel_size": 1,
                        },
                ),
                mock.patch("tunix.generate.vllm_sampler.atexit.register", lambda _fn: None),
                mock.patch(
                        "tunix.generate.vllm_sampler.utils.get_logprobs_from_vllm_output",
                        side_effect=lambda token_ids, _logprobs: [-0.123] * len(token_ids),
                ),
        ):
            sampler = VllmSampler(_DummyTokenizer(), cfg)
        self._samplers.append(sampler)
        return sampler

    def test_async_mode_initializes_async_backend(self):
        sampler = self._make_async_sampler()
        self.assertIsNone(sampler.llm)
        self.assertIsNone(sampler._driver)
        self.assertIsNotNone(sampler.async_llm)
        self.assertIsNotNone(getattr(sampler, "_async_loop", None))
        self.assertTrue(sampler._async_loop_thread.is_alive())

    def test_async_call_returns_sampler_output_for_multiple_prompts(self):
        sampler = self._make_async_sampler()
        out = sampler(
                input_strings=["a", "b"],
                max_generation_steps=8,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                seed=123,
                my_custom_flag=True,
        )

        self.assertEqual(len(out.text), 2)
        self.assertEqual(len(out.tokens), 2)
        self.assertEqual(len(out.logprobs), 2)
        self.assertTrue(all(text.startswith("decoded:") for text in out.text))
        # eos should be trimmed by detokenize(), so generated token length is 1.
        self.assertTrue(all(len(tok) == 1 for tok in out.tokens))
        # request ids are generated sequentially for each prompt.
        self.assertEqual(sampler.async_llm.request_ids, ["0", "1"])

        # Sampling parameters are populated and forwarded.
        self.assertEqual(sampler.sampling_params.max_tokens, 8)
        self.assertEqual(sampler.sampling_params.temperature, 0.7)
        self.assertEqual(sampler.sampling_params.top_p, 0.9)
        self.assertEqual(sampler.sampling_params.top_k, 20)
        self.assertEqual(sampler.sampling_params.seed, 123)
        self.assertEqual(sampler.sampling_params.output_kind, _FakeRequestOutputKind.FINAL_ONLY)
        self.assertTrue(sampler.sampling_params.my_custom_flag)

    def test_async_generate_raises_if_loop_missing(self):
        sampler = self._make_async_sampler()
        sampler._async_loop = None
        with self.assertRaisesRegex(RuntimeError, "Async loop not initialized"):
            sampler._generate_async_llm([], _FakeSamplingParams())

    def test_async_generate_coroutine_raises_when_no_output(self):
        sampler = self._make_async_sampler()
        sampler.async_llm.return_empty = True
        with self.assertRaisesRegex(RuntimeError, "No finished output produced"):
            sampler(["a"], max_generation_steps=6)

    def test_async_call_enforces_max_model_len(self):
        sampler = self._make_async_sampler()
        with self.assertRaisesRegex(ValueError, "max_generation_steps"):
            sampler(["a"], max_generation_steps=10_000)

    def test_stop_cleans_async_resources(self):
        sampler = self._make_async_sampler()
        sampler.stop()
        self.assertIsNone(sampler.async_llm)
        self.assertIsNone(getattr(sampler, "_async_loop", None))
        self.assertIsNone(getattr(sampler, "_async_loop_thread", None))


if __name__ == "__main__":
    unittest.main()
