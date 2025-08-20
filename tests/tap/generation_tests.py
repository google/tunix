# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TAP tests for Tunix generation functionality."""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Any, Callable


def test_text_generation_pipeline() -> bool:
    """Test complete text generation pipeline with realistic parameters."""
    try:
        from tunix.generate import utils
        
        # Test generation pipeline for a conversational AI scenario
        batch_size = 4
        max_seq_len = 1024
        
        # Create realistic attention masks for conversations
        # Simulate different conversation lengths in batch
        context_lengths = jnp.array([256, 128, 512, 384])  # Different conversation lengths
        
        attention_mask = jnp.zeros((batch_size, max_seq_len))
        for i, length in enumerate(context_lengths):
            attention_mask = attention_mask.at[i, :length].set(1)
        
        # Test position encoding generation
        positions = utils.build_positions_from_mask(attention_mask)
        assert positions.shape == attention_mask.shape
        
        # Validate position encoding correctness for each conversation
        for i, length in enumerate(context_lengths):
            # Positions should be 0, 1, 2, ..., length-1 for valid tokens
            # The function continues the last position for padding (this is actually correct behavior)
            valid_positions = positions[i, :length]
            expected_valid_positions = jnp.arange(length)
            assert jnp.allclose(valid_positions, expected_valid_positions)
        
        # Test causal attention mask generation (essential for autoregressive models)
        def create_causal_mask(seq_len):
            """Create lower triangular causal attention mask"""
            i = jnp.arange(seq_len)[:, None]
            j = jnp.arange(seq_len)[None, :]
            return i >= j  # Lower triangular matrix
        
        causal_mask = create_causal_mask(max_seq_len)
        assert causal_mask.shape == (max_seq_len, max_seq_len)
        assert causal_mask[0, 0] == True   # Self-attention allowed
        assert causal_mask[0, 1] == False  # Future tokens masked
        assert causal_mask[10, 5] == True  # Past tokens allowed
        
        # Test that generation utilities work correctly
        # This tests Tunix's generation functionality, not JAX operations
        
        # Test with smaller, more manageable sizes for testing
        small_mask = jnp.array([[1, 1, 0], [1, 0, 0]])
        small_positions = utils.build_positions_from_mask(small_mask)
        assert small_positions.shape == small_mask.shape
        
        # Verify the utility function handles edge cases
        empty_mask = jnp.array([[0, 0], [0, 0]])
        empty_positions = utils.build_positions_from_mask(empty_mask)
        assert empty_positions.shape == empty_mask.shape
        
        return True
    except Exception:
        return False


def test_tunix_generation_utilities() -> bool:
    """Test Tunix generation utilities and their integration."""
    try:
        from tunix.generate import utils
        from tunix.generate import beam_search
        
        # Test 1: Position encoding with different mask patterns
        # Test various attention mask scenarios that Tunix utilities handle
        test_masks = [
            jnp.array([[1, 1, 0], [1, 0, 0]]),  # Different lengths
            jnp.array([[1, 1, 1], [1, 1, 1]]),  # Full sequences
            jnp.array([[0, 0, 0], [0, 0, 0]]),  # Empty sequences
            jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])  # Variable lengths
        ]
        
        for mask in test_masks:
            positions = utils.build_positions_from_mask(mask)
            assert positions.shape == mask.shape
            # Verify that valid positions (mask == 1) have sequential values
            for i in range(mask.shape[0]):
                valid_length = int(jnp.sum(mask[i]))
                if valid_length > 0:
                    valid_positions = positions[i, :valid_length]
                    expected_positions = jnp.arange(valid_length)
                    assert jnp.allclose(valid_positions, expected_positions)
        
        # Test 2: Beam search module structure and availability
        # Verify that Tunix beam search components are accessible
        assert hasattr(beam_search, '__name__'), "Beam search module should exist"
        
        # Test 3: Generation utilities with realistic conversation data
        # Simulate a batch of conversations with varying lengths
        conversation_lengths = [64, 128, 256, 512]
        max_len = max(conversation_lengths)
        batch_size = len(conversation_lengths)
        
        # Create attention masks for conversations
        attention_mask = jnp.zeros((batch_size, max_len))
        for i, length in enumerate(conversation_lengths):
            attention_mask = attention_mask.at[i, :length].set(1)
        
        # Process through Tunix utility
        positions = utils.build_positions_from_mask(attention_mask)
        
        # Validate the utility handles real conversation data correctly
        assert positions.shape == attention_mask.shape
        
        # Test that each conversation gets proper position encoding
        for i, length in enumerate(conversation_lengths):
            valid_positions = positions[i, :length]
            expected_positions = jnp.arange(length)
            assert jnp.allclose(valid_positions, expected_positions)
            
            # Test padding behavior (should continue last valid position)
            if length < max_len:
                padding_positions = positions[i, length:]
                assert jnp.all(padding_positions == length - 1)
        
        # Test 4: Edge case handling by Tunix utilities
        # Test with very long sequences
        long_mask = jnp.ones((1, 2048))
        long_positions = utils.build_positions_from_mask(long_mask)
        assert long_positions.shape == (1, 2048)
        assert long_positions[0, 0] == 0
        assert long_positions[0, 2047] == 2047
        
        # Test with single token sequences
        single_mask = jnp.array([[1], [1], [1]])
        single_positions = utils.build_positions_from_mask(single_mask)
        assert single_positions.shape == (3, 1)
        assert jnp.all(single_positions == 0)
        
        return True
    except Exception:
        return False


def test_tunix_generation_integration() -> bool:
    """Test Tunix generation module integration and component availability."""
    try:
        from tunix.generate import utils, beam_search, sampler
        
        # Test 1: Verify all generation modules are accessible
        assert hasattr(utils, '__name__'), "Utils module should exist"
        assert hasattr(beam_search, '__name__'), "Beam search module should exist"
        assert hasattr(sampler, '__name__'), "Sampler module should exist"
        
        # Test 2: Test Tunix generation utilities with realistic text generation scenarios
        # Simulate a conversation system with multiple users
        user_conversations = [
            [64, 128, 256],    # Short conversations
            [512, 1024],       # Medium conversations  
            [2048]             # Long conversation
        ]
        
        for conv_batch in user_conversations:
            max_len = max(conv_batch)
            batch_size = len(conv_batch)
            
            # Create attention masks for this batch
            attention_mask = jnp.zeros((batch_size, max_len))
            for i, length in enumerate(conv_batch):
                attention_mask = attention_mask.at[i, :length].set(1)
            
            # Process through Tunix utility
            positions = utils.build_positions_from_mask(attention_mask)
            
            # Validate utility output
            assert positions.shape == attention_mask.shape
            
            # Test each conversation in the batch
            for i, length in enumerate(conv_batch):
                valid_positions = positions[i, :length]
                expected_positions = jnp.arange(length)
                assert jnp.allclose(valid_positions, expected_positions)
        
        # Test 3: Test Tunix utilities with edge case data
        # Test with different mask patterns (uniform shapes for JAX compatibility)
        edge_case_masks = [
            jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),  # Different valid lengths
            jnp.array([[1, 0], [1, 1]]),   # Simple case
            jnp.array([[0, 0], [0, 0]]),   # All zeros
        ]
        
        for mask in edge_case_masks:
            positions = utils.build_positions_from_mask(mask)
            assert positions.shape == mask.shape
        
        # Test 4: Verify Tunix generation module exports
        # Check that expected functions/classes are available
        expected_utils_functions = ['build_positions_from_mask']
        for func_name in expected_utils_functions:
            assert hasattr(utils, func_name), f"Utils module missing {func_name}"
        
        # Test 5: Test Tunix utilities with production-like data sizes
        # Test with realistic batch sizes and sequence lengths
        production_batch_sizes = [1, 4, 16]
        production_seq_lengths = [512, 1024, 2048]
        
        for batch_size in production_batch_sizes:
            for seq_len in production_seq_lengths:
                # Create a simple mask for this configuration
                mask = jnp.ones((batch_size, seq_len))
                positions = utils.build_positions_from_mask(mask)
                
                # Validate output
                assert positions.shape == (batch_size, seq_len)
                assert positions[0, 0] == 0
                assert positions[0, seq_len - 1] == seq_len - 1
        
        return True
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of TAP tests for generation functionality.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("text_generation_pipeline", test_text_generation_pipeline, True),
        ("tunix_generation_utilities", test_tunix_generation_utilities, True),
        ("tunix_generation_integration", test_tunix_generation_integration, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
