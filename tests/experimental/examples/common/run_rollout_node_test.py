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

"""Unit tests for run_rollout_node JSON configurations, deep-merge, and vLLM sampler creation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock
import unittest

# Dynamically import run_rollout_node with mocks for unavailable TPU/JAX dependencies
try:
  from tunix.experimental.examples.common import run_rollout_node
except Exception:
  class _MockModule(types.ModuleType):
    def __getattr__(self, name):
      val = mock.MagicMock()
      setattr(self, name, val)
      return val

  class _MockLoader:
    def create_module(self, spec):
      mod = _MockModule(spec.name)
      mod.__path__ = []
      mod.__file__ = spec.name + ".py"
      return mod

    def exec_module(self, module):
      pass

  class _AutoMockFinder:
    def find_spec(self, fullname, path, target=None):
      stdlib = {
          "sys", "os", "yaml", "json", "ast", "argparse", "typing",
          "unittest", "re", "importlib", "builtins", "collections",
          "pathlib", "itertools", "functools",
      }
      if fullname.split(".")[0] in sys.builtin_module_names or fullname.split(".")[0] in stdlib:
        return None
      from importlib.machinery import ModuleSpec
      return ModuleSpec(fullname, _MockLoader())

  finder = _AutoMockFinder()
  sys.meta_path.insert(0, finder)
  try:
    _module_path = (
        Path(__file__).resolve().parents[4]
        / "tunix"
        / "experimental"
        / "examples"
        / "common"
        / "run_rollout_node.py"
    )
    _spec = importlib.util.spec_from_file_location(
        "tunix.experimental.examples.common.run_rollout_node", str(_module_path)
    )
    run_rollout_node = importlib.util.module_from_spec(_spec)
    sys.modules["tunix.experimental.examples.common.run_rollout_node"] = run_rollout_node
    _spec.loader.exec_module(run_rollout_node)
  finally:
    if finder in sys.meta_path:
      sys.meta_path.remove(finder)


class RunRolloutNodeTest(unittest.TestCase):

  def test_load_json_config_dict(self):
    cfg = {"sharding": {"expert_parallelism": 8}, "multiplier": 16}
    self.assertEqual(run_rollout_node._load_json_config(cfg), cfg)

  def test_load_json_config_inline_json(self):
    json_str = '{"sharding": {"expert_parallelism": 8}, "multiplier": 16}'
    self.assertEqual(
        run_rollout_node._load_json_config(json_str),
        {"sharding": {"expert_parallelism": 8}, "multiplier": 16},
    )

  def test_load_json_config_from_file(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
      f.write('{"block_size": 32, "max_num_seqs": 8}')
      tmp_path = f.name
    try:
      self.assertEqual(
          run_rollout_node._load_json_config(tmp_path),
          {"block_size": 32, "max_num_seqs": 8},
      )
    finally:
      os.remove(tmp_path)

  def test_load_json_config_empty_and_invalid(self):
    self.assertEqual(run_rollout_node._load_json_config(None), {})
    self.assertEqual(run_rollout_node._load_json_config(""), {})
    self.assertEqual(run_rollout_node._load_json_config("   "), {})
    with self.assertRaises(ValueError):
      run_rollout_node._load_json_config(123)
    with self.assertRaises(ValueError):
      run_rollout_node._load_json_config("{not valid json or yaml:")

  def test_deep_merge_dicts(self):
    base = {
        "a": 1,
        "nested": {"x": 10, "y": 20},
        "list": [1, 2],
    }
    update = {
        "b": 2,
        "nested": {"y": 30, "z": 40},
        "list": [3, 4],
    }
    merged = run_rollout_node._deep_merge_dicts(base, update)
    self.assertEqual(
        merged,
        {
            "a": 1,
            "b": 2,
            "nested": {"x": 10, "y": 30, "z": 40},
            "list": [3, 4],
        },
    )
    self.assertEqual(base["nested"], {"x": 10, "y": 20})

  def test_parse_args_with_vllm_config_json(self):
    argv = [
        "--model_id", "Qwen/Qwen3.5-397B-A17B",
        "--mesh_tp", "2",
        "--vllm_config_json", '{"block_size": 32, "async_scheduling": true, "max_model_len": 4096}',
    ]
    args = run_rollout_node._parse_args(argv)
    self.assertEqual(args.model_id, "Qwen/Qwen3.5-397B-A17B")
    self.assertEqual(args.tensor_parallel_size, 2)
    self.assertEqual(
        run_rollout_node._load_json_config(args.vllm_config_json),
        {"block_size": 32, "async_scheduling": True, "max_model_len": 4096},
    )

  def test_parse_args_with_vllm_config_file(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
      f.write('{"max_num_batched_tokens": 1024}')
      tmp_path = f.name
    try:
      argv = ["--vllm_config_json", tmp_path]
      args = run_rollout_node._parse_args(argv)
      self.assertEqual(
          run_rollout_node._load_json_config(args.vllm_config_json),
          {"max_num_batched_tokens": 1024},
      )
    finally:
      os.remove(tmp_path)

  def test_inprocess_vllm_sampler_with_vllm_config_json(self):
    mock_vllm_sampler = mock.MagicMock()
    mock_vllm_config = mock.MagicMock()
    mock_vllm_sampler.VllmConfig = mock_vllm_config

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    mock_inprocess_adapter = mock.MagicMock()
    mock_rollout_worker = mock.MagicMock()
    inprocess_mocks = {
        "tunix.experimental.rollout": types.ModuleType("tunix.experimental.rollout"),
        "tunix.experimental.rollout.inprocess_vllm_sampler_adapter": types.ModuleType(
            "tunix.experimental.rollout.inprocess_vllm_sampler_adapter"
        ),
        "tunix.experimental.worker": types.ModuleType("tunix.experimental.worker"),
        "tunix.experimental.worker.rollout_worker": mock_rollout_worker,
    }
    if "jax" not in sys.modules:
      inprocess_mocks["jax"] = mock.MagicMock()
    if "tunix.generate" not in sys.modules:
      inprocess_mocks["tunix.generate"] = types.ModuleType("tunix.generate")
      inprocess_mocks["tunix.generate.mappings"] = types.ModuleType("tunix.generate.mappings")
      inprocess_mocks["tunix.generate.mappings"].MappingConfig = mock.MagicMock()
      inprocess_mocks["tunix.generate.tokenizer_adapter"] = types.ModuleType("tunix.generate.tokenizer_adapter")
      inprocess_mocks["tunix.generate.tokenizer_adapter"].TokenizerAdapter = mock.MagicMock()
    if "tunix.models.qwen3" not in sys.modules:
      inprocess_mocks["tunix.models.qwen3"] = types.ModuleType("tunix.models.qwen3")
      inprocess_mocks["tunix.models.qwen3.mapping_vllm_jax"] = types.ModuleType("tunix.models.qwen3.mapping_vllm_jax")
      inprocess_mocks["tunix.models.qwen3.mapping_vllm_jax"].VLLM_JAX_MAPPING = {}
    inprocess_mocks[
        "tunix.experimental.rollout.inprocess_vllm_sampler_adapter"
    ].InprocessVllmSamplerAdapter = mock_inprocess_adapter

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        mesh_tp=2,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=None,
        enable_prefix_caching=False,
        maxtext_model_name="",
        maxtext_attention="",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        vllm_config_json={
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "data_parallel_size": 2,
            "expert_parallel_size": 4,
            "block_size": 32,
            "async_scheduling": True,
            "additional_config": {"extra_opt": 1},
        },
    )

    with mock.patch.dict(sys.modules, inprocess_mocks), mock.patch.object(
        run_rollout_node, "_import_vllm_sampler", return_value=mock_vllm_sampler
    ), mock.patch.object(
        run_rollout_node, "_create_rollout_mesh", return_value=None
    ):
      run_rollout_node._create_inprocess_vllm_sampler(args, mock_tokenizer)

    mock_vllm_config.assert_called_once()
    _, kwargs = mock_vllm_config.call_args
    self.assertEqual(kwargs["tensor_parallel_size"], 2)
    self.assertEqual(kwargs["data_parallel_size"], 2)
    self.assertEqual(kwargs["expert_parallel_size"], 4)
    self.assertEqual(kwargs["hbm_utilization"], 0.85)
    self.assertEqual(kwargs["additional_config"], {"extra_opt": 1})
    engine_kwargs = kwargs["engine_kwargs"]
    self.assertEqual(engine_kwargs["max_model_len"], 8192)
    self.assertEqual(engine_kwargs["block_size"], 32)
    self.assertTrue(engine_kwargs["async_scheduling"])
    self.assertNotIn("tensor_parallel_size", engine_kwargs)
    self.assertNotIn("data_parallel_size", engine_kwargs)
    self.assertNotIn("expert_parallel_size", engine_kwargs)
    self.assertNotIn("gpu_memory_utilization", engine_kwargs)
    self.assertNotIn("hbm_utilization", engine_kwargs)
    self.assertNotIn("additional_config", engine_kwargs)

  def test_inprocess_vllm_sampler_with_maxtext_merging(self):
    mock_vllm_sampler = mock.MagicMock()
    mock_vllm_config = mock.MagicMock()
    mock_vllm_sampler.VllmConfig = mock_vllm_config

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    mock_inprocess_adapter = mock.MagicMock()
    mock_rollout_worker = mock.MagicMock()
    inprocess_mocks = {
        "tunix.experimental.rollout": types.ModuleType("tunix.experimental.rollout"),
        "tunix.experimental.rollout.inprocess_vllm_sampler_adapter": types.ModuleType(
            "tunix.experimental.rollout.inprocess_vllm_sampler_adapter"
        ),
        "tunix.experimental.worker": types.ModuleType("tunix.experimental.worker"),
        "tunix.experimental.worker.rollout_worker": mock_rollout_worker,
    }
    if "jax" not in sys.modules:
      inprocess_mocks["jax"] = mock.MagicMock()
    if "tunix.generate" not in sys.modules:
      inprocess_mocks["tunix.generate"] = types.ModuleType("tunix.generate")
      inprocess_mocks["tunix.generate.mappings"] = types.ModuleType("tunix.generate.mappings")
      inprocess_mocks["tunix.generate.mappings"].MappingConfig = mock.MagicMock()
      inprocess_mocks["tunix.generate.tokenizer_adapter"] = types.ModuleType("tunix.generate.tokenizer_adapter")
      inprocess_mocks["tunix.generate.tokenizer_adapter"].TokenizerAdapter = mock.MagicMock()
    if "tunix.models.qwen3" not in sys.modules:
      inprocess_mocks["tunix.models.qwen3"] = types.ModuleType("tunix.models.qwen3")
      inprocess_mocks["tunix.models.qwen3.mapping_vllm_jax"] = types.ModuleType("tunix.models.qwen3.mapping_vllm_jax")
      inprocess_mocks["tunix.models.qwen3.mapping_vllm_jax"].VLLM_JAX_MAPPING = {}
    inprocess_mocks[
        "tunix.experimental.rollout.inprocess_vllm_sampler_adapter"
    ].InprocessVllmSamplerAdapter = mock_inprocess_adapter

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        maxtext_model_name="Qwen/Qwen3.5-397B-A17B",
        maxtext_attention="vllm_rpa",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        vllm_config_json={
            "scheduling_policy": "priority",
            "additional_config": {
                "custom_mamba_cache_multiplier": 16,
                "maxtext_config": {"scan_layers": False},
            },
        },
    )

    fake_built_maxtext = {
        "maxtext_config": {
            "model_name": "qwen3-0.6b",
            "attention": "vllm_rpa",
            "scan_layers": True,
        }
    }
    with mock.patch.dict(sys.modules, inprocess_mocks), mock.patch.object(
        run_rollout_node, "_import_vllm_sampler", return_value=mock_vllm_sampler
    ), mock.patch.object(
        run_rollout_node, "_create_rollout_mesh", return_value=None
    ), mock.patch.object(
        run_rollout_node.maxtext_utils,
        "build_vllm_maxtext_additional_config",
        return_value=fake_built_maxtext,
    ):
      run_rollout_node._create_inprocess_vllm_sampler(args, mock_tokenizer)

    mock_vllm_config.assert_called_once()
    _, kwargs = mock_vllm_config.call_args
    cfg = kwargs["additional_config"]
    self.assertEqual(cfg["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(cfg["maxtext_config"]["model_name"], "qwen3-0.6b")
    self.assertEqual(cfg["maxtext_config"]["attention"], "vllm_rpa")
    # User override took effect
    self.assertFalse(cfg["maxtext_config"]["scan_layers"])
    self.assertEqual(kwargs["engine_kwargs"]["scheduling_policy"], "priority")

  def test_vllm_sampler_with_vllm_config_json(self):
    mock_async_engine_args = mock.MagicMock()
    mock_vllm_adapter = mock.MagicMock()
    mock_rollout_worker = mock.MagicMock()

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    vllm_mocks = {
        "vllm": types.ModuleType("vllm"),
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": types.ModuleType("vllm.engine.arg_utils"),
        "tunix.experimental.rollout": types.ModuleType("tunix.experimental.rollout"),
        "tunix.experimental.rollout.vllm_sampler_adapter": types.ModuleType(
            "tunix.experimental.rollout.vllm_sampler_adapter"
        ),
        "tunix.experimental.worker": types.ModuleType("tunix.experimental.worker"),
        "tunix.experimental.worker.rollout_worker": mock_rollout_worker,
    }
    vllm_mocks["vllm.engine.arg_utils"].AsyncEngineArgs = mock_async_engine_args
    vllm_mocks[
        "tunix.experimental.rollout.vllm_sampler_adapter"
    ].VllmSamplerAdapter = mock_vllm_adapter

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        mesh_tp=2,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=None,
        enable_prefix_caching=False,
        maxtext_model_name="",
        maxtext_attention="",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        vllm_config_json={
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85,
            "data_parallel_size": 2,
            "block_size": 32,
            "async_scheduling": True,
            "enable_expert_parallel": True,
            "additional_config": {"multiplier": 8},
        },
    )

    with mock.patch.dict(sys.modules, vllm_mocks):
      run_rollout_node._create_vllm_sampler(args, mock_tokenizer)

    mock_async_engine_args.assert_called_once()
    _, kwargs = mock_async_engine_args.call_args
    self.assertEqual(kwargs["tensor_parallel_size"], 2)
    self.assertEqual(kwargs["data_parallel_size"], 2)
    self.assertEqual(kwargs["max_model_len"], 8192)
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.85)
    self.assertEqual(kwargs["block_size"], 32)
    self.assertTrue(kwargs["async_scheduling"])
    self.assertTrue(kwargs["enable_expert_parallel"])
    self.assertEqual(kwargs["additional_config"], {"multiplier": 8})

  def test_vllm_sampler_with_maxtext_merging(self):
    mock_async_engine_args = mock.MagicMock()
    mock_vllm_adapter = mock.MagicMock()
    mock_rollout_worker = mock.MagicMock()

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    vllm_mocks = {
        "vllm": types.ModuleType("vllm"),
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": types.ModuleType("vllm.engine.arg_utils"),
        "tunix.experimental.rollout": types.ModuleType("tunix.experimental.rollout"),
        "tunix.experimental.rollout.vllm_sampler_adapter": types.ModuleType(
            "tunix.experimental.rollout.vllm_sampler_adapter"
        ),
        "tunix.experimental.worker": types.ModuleType("tunix.experimental.worker"),
        "tunix.experimental.worker.rollout_worker": mock_rollout_worker,
    }
    vllm_mocks["vllm.engine.arg_utils"].AsyncEngineArgs = mock_async_engine_args
    vllm_mocks[
        "tunix.experimental.rollout.vllm_sampler_adapter"
    ].VllmSamplerAdapter = mock_vllm_adapter

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        maxtext_model_name="Qwen/Qwen3.5-397B-A17B",
        maxtext_attention="vllm_rpa",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        vllm_config_json={
            "scheduling_policy": "priority",
            "additional_config": {
                "custom_mamba_cache_multiplier": 16,
                "maxtext_config": {"scan_layers": False},
            },
        },
    )

    fake_built_maxtext = {
        "maxtext_config": {
            "model_name": "qwen3-0.6b",
            "scan_layers": True,
        }
    }
    with mock.patch.dict(sys.modules, vllm_mocks), mock.patch.object(
        run_rollout_node.maxtext_utils,
        "build_vllm_maxtext_additional_config",
        return_value=fake_built_maxtext,
    ):
      run_rollout_node._create_vllm_sampler(args, mock_tokenizer)

    mock_async_engine_args.assert_called_once()
    _, kwargs = mock_async_engine_args.call_args
    cfg = kwargs["additional_config"]
    self.assertEqual(cfg["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(cfg["maxtext_config"]["model_name"], "qwen3-0.6b")
    self.assertFalse(cfg["maxtext_config"]["scan_layers"])
    self.assertEqual(kwargs["scheduling_policy"], "priority")

  def test_hbm_utilization_key_in_vllm_config_json(self):
    mock_async_engine_args = mock.MagicMock()
    mock_vllm_adapter = mock.MagicMock()
    mock_rollout_worker = mock.MagicMock()

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    vllm_mocks = {
        "vllm": types.ModuleType("vllm"),
        "vllm.engine": types.ModuleType("vllm.engine"),
        "vllm.engine.arg_utils": types.ModuleType("vllm.engine.arg_utils"),
        "tunix.experimental.rollout": types.ModuleType("tunix.experimental.rollout"),
        "tunix.experimental.rollout.vllm_sampler_adapter": types.ModuleType(
            "tunix.experimental.rollout.vllm_sampler_adapter"
        ),
        "tunix.experimental.worker": types.ModuleType("tunix.experimental.worker"),
        "tunix.experimental.worker.rollout_worker": mock_rollout_worker,
    }
    vllm_mocks["vllm.engine.arg_utils"].AsyncEngineArgs = mock_async_engine_args
    vllm_mocks[
        "tunix.experimental.rollout.vllm_sampler_adapter"
    ].VllmSamplerAdapter = mock_vllm_adapter

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        maxtext_model_name="",
        maxtext_attention="",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        vllm_config_json={"hbm_utilization": 0.77},
    )

    with mock.patch.dict(sys.modules, vllm_mocks):
      run_rollout_node._create_vllm_sampler(args, mock_tokenizer)

    mock_async_engine_args.assert_called_once()
    _, kwargs = mock_async_engine_args.call_args
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.77)
    self.assertNotIn("hbm_utilization", kwargs)


if __name__ == "__main__":
  unittest.main()
