from unittest import mock

from absl.testing import absltest

from tunix.generate import vllm_sampler


class _DummyTokenizer:

  def encode(self, text):
    return [1, 2]

  def decode(self, tokens):
    return "dummy"

  def bos_id(self):
    return 1

  def eos_id(self):
    return 2

  def pad_id(self):
    return 0


class _FakeDeviceIds:

  def flatten(self):
    return self

  def tolist(self):
    return [0]


class _FakeMesh:
  shape = {"tp": 1}
  device_ids = _FakeDeviceIds()


class VllmSamplerAsyncModeTest(absltest.TestCase):

  def _create_config(self):
    mapping_config = vllm_sampler.MappingConfig(
        to_hf_mappings={},
        lora_to_hf_mappings=None,
        to_hf_hook_fns=None,
        to_hf_transpose_keys=None,
        lora_config=None,
    )
    return vllm_sampler.VllmConfig(
        model_version="fake-model",
        max_model_len=16,
        mesh=_FakeMesh(),
        hbm_utilization=0.5,
        init_with_random_weights=False,
        tpu_backend_type="",
        mapping_config=mapping_config,
        async_mode=True,
        port=9000,
        served_model_name="async-model",
        api_key="abc123",
        extra_args=None,
    )

  def test_async_mode_start_server_in_thread(self):
    config = self._create_config()
    tokenizer = _DummyTokenizer()

    with (
        mock.patch.object(vllm_sampler, "LLM") as mock_llm,
        mock.patch.object(vllm_sampler.threading, "Thread") as mock_thread,
        mock.patch.object(vllm_sampler.asyncio, "run") as mock_asyncio_run,
        mock.patch.object(vllm_sampler, "run_server") as mock_run_server,
    ):
      thread_instance = mock.Mock()
      mock_thread.return_value = thread_instance

      sampler = vllm_sampler.VllmSampler(tokenizer=tokenizer, config=config)

      mock_llm.assert_not_called()
      mock_thread.assert_called_once()
      thread_instance.start.assert_called_once()
      self.assertIs(sampler.llm_thread, thread_instance)

      thread_target = mock_thread.call_args.kwargs["target"]
      thread_target()

      mock_asyncio_run.assert_called_once()
      self.assertIs(
          mock_asyncio_run.call_args.args[0], mock_run_server.return_value
      )

      call_kwargs = mock_run_server.call_args.kwargs
      self.assertEqual(call_kwargs["model"], config.model_version)
      self.assertEqual(call_kwargs["tensor_parallel_size"], 1)
      self.assertEqual(call_kwargs["max_model_len"], config.max_model_len)
      self.assertEqual(
          call_kwargs["gpu_memory_utilization"], config.hbm_utilization
      )
      self.assertEqual(call_kwargs["port"], config.port)
      self.assertEqual(
          call_kwargs["served_model_name"], config.served_model_name
      )
      self.assertEqual(call_kwargs["api_key"], config.api_key)
      self.assertIn("additional_config", call_kwargs)
      self.assertIn("sharding", call_kwargs["additional_config"])


if __name__ == "__main__":
  absltest.main()
