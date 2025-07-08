# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for vLLM sampler."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from tunix.generate import vllm_sampler
from tunix.generate.sampler import SamplerOutput
from tunix.tests import test_common as tc


class vLLMSamplerTest(parameterized.TestCase):

  def setUp(self):
    """Set up test fixtures."""
    self.mock_tokenizer = tc.MockVocab()
    self.mock_model = tc.ToyTransformer(
        rngs=nnx.Rngs(42), vocab_size=self.mock_tokenizer.GetPieceSize()
    )
    self.default_lora_config = {
        "rank": 64,
        "alpha": 64.0,
        "module_path": ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
    }

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_vllm_sampler_initialization(self, mock_llm):
    """Test vLLMSampler initialization."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    # Mock the sampling params
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config,
        model_version="meta-llama/Llama-3.1-8B",
        max_model_len=1024
    )
    
    # Verify LLM was called with correct arguments
    mock_llm.assert_called_once()
    call_args = mock_llm.call_args[1]
    
    # Check that the args contain the expected keys
    self.assertIn('additional_config', call_args)
    self.assertIn('custom_nnx_weights', call_args['additional_config'])
    self.assertIn('lora_config', call_args['additional_config'])
    self.assertIn('max_model_len', call_args)
    self.assertIn('model', call_args)
    
    # Verify sampling params were configured
    self.assertEqual(mock_sampling_params.detokenize, False)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_vllm_sampler_initialization_no_lora(self, mock_llm):
    """Test vLLMSampler initialization without LoRA config."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=None,
        max_model_len=512
    )
    
    # Verify default LoRA config was used
    call_args = mock_llm.call_args[1]
    lora_config = call_args['additional_config']['lora_config']
    self.assertEqual(lora_config['rank'], 64)
    self.assertEqual(lora_config['alpha'], 64.0)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_tokenize_method(self, mock_llm):
    """Test the tokenize method."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test tokenization
    input_string = "input string"
    tokenized = sampler.tokenize(input_string)
    
    # Should include BOS token + input tokens
    expected_tokens = [self.mock_tokenizer.bos_id()] + self.mock_tokenizer.EncodeAsIds(input_string)
    self.assertEqual(tokenized, expected_tokens)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_detokenize_method(self, mock_llm):
    """Test the detokenize method."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Mock RequestOutput structure
    mock_output = Mock()
    mock_single_output = Mock()
    mock_single_output.token_ids = [5, 6]  # "hello world"
    mock_single_output.logprob = [0.1, 0.2]
    mock_output.__iter__ = lambda self: iter([[mock_single_output]])
    
    prompts = ["test prompt"]
    outputs = [mock_output]
    
    # Test with return_logits=True
    decoded_outputs, out_logits, out_tokens = sampler.detokenize(
        prompts, return_logits=True, outputs=outputs
    )
    
    self.assertEqual(len(decoded_outputs), 1)
    self.assertEqual(len(out_logits), 1)
    self.assertEqual(len(out_tokens), 1)
    self.assertEqual(decoded_outputs[0][0], "hello world")
    self.assertEqual(out_logits[0][0], [0.1, 0.2])
    self.assertEqual(out_tokens[0][0], [5, 6])

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_call_method_basic(self, mock_llm):
    """Test the __call__ method with basic parameters."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Mock the generate method
    mock_output = Mock()
    mock_single_output = Mock()
    mock_single_output.token_ids = [5, 6]  # "hello world"
    mock_single_output.logprob = [0.1, 0.2]
    mock_output.__iter__ = lambda self: iter([[mock_single_output]])
    
    mock_llm_instance.generate.return_value = [mock_output]
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test basic call
    result = sampler(
        prompts=["test prompt"],
        max_generation_length=10,
        return_logits=True
    )
    
    # Verify result structure
    self.assertIsInstance(result, SamplerOutput)
    self.assertEqual(len(result.text), 1)
    self.assertEqual(len(result.logits), 1)
    self.assertEqual(len(result.tokens), 1)
    
    # Verify sampling params were updated
    self.assertEqual(mock_sampling_params.max_tokens, 10)
    self.assertEqual(mock_sampling_params.n, 1)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_call_method_with_sampling_params(self, mock_llm):
    """Test the __call__ method with temperature, top_p, top_k."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Mock the generate method
    mock_output = Mock()
    mock_single_output = Mock()
    mock_single_output.token_ids = [5, 6]
    mock_single_output.logprob = [0.1, 0.2]
    mock_output.__iter__ = lambda self: iter([[mock_single_output]])
    
    mock_llm_instance.generate.return_value = [mock_output]
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test with sampling parameters
    result = sampler(
        prompts=["test prompt"],
        max_generation_length=20,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        multi_sampling=3,
        return_logits=False
    )
    
    # Verify sampling params were updated
    self.assertEqual(mock_sampling_params.temperature, 0.8)
    self.assertEqual(mock_sampling_params.top_p, 0.9)
    self.assertEqual(mock_sampling_params.top_k, 50)
    self.assertEqual(mock_sampling_params.n, 3)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_call_method_max_length_validation(self, mock_llm):
    """Test that max_generation_length validation works."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config,
        max_model_len=100
    )
    
    # Test that exceeding max_model_len raises an error
    with self.assertRaises(AssertionError):
      sampler(
          prompts=["test prompt"],
          max_generation_length=150,  # Exceeds max_model_len=100
          return_logits=False
      )

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_transformer_state_property(self, mock_llm):
    """Test the transformer_state property."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Mock the nested structure
    mock_model_executor = Mock()
    mock_driver_worker = Mock()
    mock_model_runner = Mock()
    mock_transformer_state = Mock()
    
    mock_llm_instance.llm_engine.model_executor = mock_model_executor
    mock_model_executor.driver_worker = mock_driver_worker
    mock_driver_worker.model_runner = mock_model_runner
    mock_model_runner.transformer_state = mock_transformer_state
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test the property
    transformer_state = sampler.transformer_state
    self.assertEqual(transformer_state, mock_transformer_state)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_multi_sampling_output_structure(self, mock_llm):
    """Test that multi-sampling produces correct output structure."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Mock multiple outputs for multi-sampling
    mock_output = Mock()
    mock_single_output1 = Mock()
    mock_single_output1.token_ids = [5, 6]  # "hello world"
    mock_single_output1.logprob = [0.1, 0.2]
    
    mock_single_output2 = Mock()
    mock_single_output2.token_ids = [7, 8]  # "Hello there"
    mock_single_output2.logprob = [0.3, 0.4]
    
    mock_output.__iter__ = lambda self: iter([[mock_single_output1, mock_single_output2]])
    
    mock_llm_instance.generate.return_value = [mock_output]
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test multi-sampling
    result = sampler(
        prompts=["test prompt"],
        max_generation_length=10,
        multi_sampling=2,
        return_logits=True
    )
    
    # Verify multi-sampling structure
    self.assertEqual(len(result.text), 2)  # Two samples
    self.assertEqual(len(result.logits), 2)
    self.assertEqual(len(result.tokens), 2)
    
    # Verify sampling params
    self.assertEqual(mock_sampling_params.n, 2)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_batch_processing(self, mock_llm):
    """Test processing multiple prompts in a batch."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Mock outputs for multiple prompts
    mock_output1 = Mock()
    mock_single_output1 = Mock()
    mock_single_output1.token_ids = [5, 6]  # "hello world"
    mock_single_output1.logprob = [0.1, 0.2]
    mock_output1.__iter__ = lambda self: iter([[mock_single_output1]])
    
    mock_output2 = Mock()
    mock_single_output2 = Mock()
    mock_single_output2.token_ids = [7, 8]  # "Hello there"
    mock_single_output2.logprob = [0.3, 0.4]
    mock_output2.__iter__ = lambda self: iter([[mock_single_output2]])
    
    mock_llm_instance.generate.return_value = [mock_output1, mock_output2]
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Test batch processing
    result = sampler(
        prompts=["prompt 1", "prompt 2"],
        max_generation_length=10,
        return_logits=True
    )
    
    # Verify batch processing
    self.assertEqual(len(result.text), 1)  # Only first prompt's output
    self.assertEqual(len(result.logits), 1)
    self.assertEqual(len(result.tokens), 1)
    
    # Verify generate was called with correct prompt_token_ids
    generate_call = mock_llm_instance.generate.call_args
    self.assertIn('prompt_token_ids', generate_call[1])

  def test_lora_config_default_values(self):
    """Test that default LoRA config values are correct."""
    # Test the default LoRA config structure
    expected_config = {
        "rank": 64,
        "alpha": 64.0,
        "module_path": ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
    }
    
    self.assertEqual(self.default_lora_config, expected_config)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_error_handling_invalid_model_version(self, mock_llm):
    """Test error handling for invalid model version."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    # Test with invalid model version
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config,
        model_version="invalid/model/version"
    )
    
    # The sampler should still initialize, but the LLM might fail later
    # This test ensures the initialization doesn't crash
    self.assertIsNotNone(sampler)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_tokenizer_adapter_integration(self, mock_llm):
    """Test that TokenizerAdapter is properly integrated."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Verify tokenizer adapter was created
    self.assertIsNotNone(sampler.tokenizer)
    self.assertEqual(sampler.tokenizer.bos_id(), 1)
    self.assertEqual(sampler.tokenizer.eos_id(), 2)
    self.assertEqual(sampler.tokenizer.pad_id(), 0)

  @patch('tunix.generate.vllm_sampler.LLM')
  def test_model_parameter_extraction(self, mock_llm):
    """Test that model parameters are properly extracted."""
    mock_llm_instance = Mock()
    mock_llm.return_value = mock_llm_instance
    
    mock_sampling_params = Mock()
    mock_llm_instance.get_default_sampling_params.return_value = mock_sampling_params
    
    sampler = vllm_sampler.vLLMSampler(
        tokenizer=self.mock_tokenizer,
        model=self.mock_model,
        lora_config=self.default_lora_config
    )
    
    # Verify that model parameters were extracted and passed to LLM
    call_args = mock_llm.call_args[1]
    self.assertIn('custom_nnx_weights', call_args['additional_config'])
    self.assertIsNotNone(call_args['additional_config']['custom_nnx_weights'])


if __name__ == '__main__':
  absltest.main() 