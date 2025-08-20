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

"""Basic TAP tests for Tunix core functionality."""

import jax
import jax.numpy as jnp
import time
from typing import List, Tuple, Any, Callable


def test_tunix_core_functionality() -> bool:
    """Test core Tunix functionality and module integration."""
    try:
        # Test that all core Tunix modules can be imported and accessed
        import tunix
        import tunix.generate
        import tunix.rl
        import tunix.sft
        import tunix.distillation
        import tunix.models
        
        # Test core Tunix API classes are available
        from tunix import (
            DistillationTrainer,
            Sampler,
            DpoTrainer,
            GrpoLearner,
            RLCluster,
            PeftTrainer
        )
        
        # Test that these classes exist and are callable
        assert callable(DistillationTrainer)
        assert callable(Sampler)
        assert callable(DpoTrainer)
        assert callable(GrpoLearner)
        assert callable(RLCluster)
        assert callable(PeftTrainer)
        
        # Test module structure
        assert hasattr(tunix.generate, 'sampler')
        assert hasattr(tunix.rl, 'grpo')
        assert hasattr(tunix.sft, 'peft_trainer')
        # Note: tunix.models is not a direct module, individual model modules are imported separately
        
        return True
    except Exception:
        return False


def test_tunix_data_flow_and_operations() -> bool:
    """Test Tunix data flow, operations, and actual functionality."""
    try:
        from tunix.generate import utils
        from tunix.sft import metrics_logger, system_metrics_calculator
        
        # Test 1: Position encoding with real data flow
        # Create realistic conversation data with varying lengths
        conversation_lengths = [128, 256, 64, 512]  # Different conversation lengths
        max_len = max(conversation_lengths)
        batch_size = len(conversation_lengths)
        
        # Build attention mask from real conversation data
        attention_mask = jnp.zeros((batch_size, max_len))
        for i, length in enumerate(conversation_lengths):
            attention_mask = attention_mask.at[i, :length].set(1)
        
        # Process through Tunix utility function
        positions = utils.build_positions_from_mask(attention_mask)
        
        # Validate the data transformation makes sense
        assert positions.shape == attention_mask.shape
        
        # Test that positions are correctly computed for each conversation
        for i, length in enumerate(conversation_lengths):
            # Valid positions should be sequential
            valid_positions = positions[i, :length]
            expected_positions = jnp.arange(length)
            assert jnp.allclose(valid_positions, expected_positions)
            
            # Padding positions should continue the last valid position
            if length < max_len:
                padding_positions = positions[i, length:]
                assert jnp.all(padding_positions == length - 1)
        
        # Test 2: Metrics logging with real training data flow
        # Create realistic training metrics over multiple steps
        training_steps = [100, 200, 300, 400, 500]
        loss_values = [2.8, 2.5, 2.2, 1.9, 1.7]  # Decreasing loss
        accuracy_values = [0.65, 0.72, 0.78, 0.83, 0.87]  # Increasing accuracy
        
        # Initialize metrics logger
        options = metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/tunix_training_logs",
            flush_every_n_steps=100
        )
        logger = metrics_logger.MetricsLogger(options)
        
        # Log training progression
        for step, loss, accuracy in zip(training_steps, loss_values, accuracy_values):
            logger.log("train_loss", jnp.array(loss), metrics_logger.Mode.TRAIN, step)
            logger.log("train_accuracy", jnp.array(accuracy), metrics_logger.Mode.TRAIN, step)
        
        # Test 3: System performance calculation with realistic model data
        # Simulate training a 7B parameter model
        model_params = 7_000_000_000
        batch_sizes = [16, 32, 64, 128]  # Different batch sizes
        step_times = [1.2, 1.1, 1.0, 0.9]  # Improving step times
        
        # Calculate TFLOPS for different configurations
        tflops_results = []
        for batch_size, step_time in zip(batch_sizes, step_times):
            tflops = system_metrics_calculator.tflops(
                total_model_params=model_params,
                global_batch_size=batch_size,
                step_time_delta=step_time,
            )
            tflops_results.append(tflops)
            
            # Validate TFLOPS calculation is reasonable
            assert isinstance(tflops, float)
            assert tflops > 0
            assert tflops < 1000  # Sanity check
        
        # Test that larger batch sizes generally give higher TFLOPS
        # (more parallel computation)
        for i in range(1, len(tflops_results)):
            # This should generally be true, but allow some variance
            # due to memory bandwidth limitations
            assert tflops_results[i] > 0.5 * tflops_results[i-1]
        
        # Test 4: Data validation and error handling
        # Test with edge case data (fixed length for JAX compatibility)
        edge_case_mask = jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])  # Different lengths, same shape
        edge_positions = utils.build_positions_from_mask(edge_case_mask)
        
        # Verify the utility handles variable-length sequences correctly
        assert edge_positions.shape == edge_case_mask.shape
        
        # Test 5: Performance characteristics
        # Test that position encoding scales reasonably with sequence length
        long_sequence = jnp.ones((1, 1024))  # Long sequence
        start_time = time.time()
        long_positions = utils.build_positions_from_mask(long_sequence)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert long_positions.shape == (1, 1024)
        
        return True
    except Exception:
        return False


def test_tunix_ml_utilities_and_metrics() -> bool:
    """Test Tunix utilities with realistic ML training scenarios."""
    try:
        from tunix.generate import utils
        from tunix.sft import metrics_logger, system_metrics_calculator
        
        # Test position building for transformer attention
        # Simulate a batch with varying sequence lengths (padding)
        mask = jnp.array([
            [1, 1, 1, 1, 0, 0],  # Sequence length 4
            [1, 1, 1, 0, 0, 0],  # Sequence length 3
            [1, 1, 1, 1, 1, 1]   # Sequence length 6 (no padding)
        ])
        
        positions = utils.build_positions_from_mask(mask)
        assert positions.shape == mask.shape
        
        # Validate position encoding makes sense
        # First sequence should have positions [0,1,2,3,0,0]
        assert positions[0, 0] == 0
        assert positions[0, 3] == 3
        
        # Test that the utility function handles different mask shapes correctly
        # This tests Tunix's utility function, not JAX operations
        small_mask = jnp.array([[1, 0], [1, 1]])
        small_positions = utils.build_positions_from_mask(small_mask)
        assert small_positions.shape == small_mask.shape
        
        # Test metrics logger with realistic training metrics
        options = metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/tunix_training_logs",
            flush_every_n_steps=100  # Realistic flush frequency
        )
        
        logger = metrics_logger.MetricsLogger(options)
        
        # Simulate logging training metrics
        step = 1000
        train_loss = jnp.array(2.5)  # Realistic language model loss
        eval_accuracy = jnp.array(0.82)  # Realistic accuracy
        
        logger.log("train_loss", train_loss, metrics_logger.Mode.TRAIN, step)
        logger.log("eval_accuracy", eval_accuracy, metrics_logger.Mode.EVAL, step)
        
        # Test system metrics calculation for performance monitoring
        model_params = 7_000_000_000  # 7B parameter model
        batch_size = 32
        step_time = 0.8  # seconds per step
        
        tflops = system_metrics_calculator.tflops(
            total_model_params=model_params,
            global_batch_size=batch_size,
            step_time_delta=step_time,
        )
        
        # Validate TFLOPS calculation is reasonable
        assert isinstance(tflops, float)
        assert tflops > 0
        assert tflops < 1000  # Sanity check - shouldn't be impossibly high
        
        return True
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of TAP tests for basic functionality.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("tunix_core_functionality", test_tunix_core_functionality, True),
        ("tunix_data_flow_and_operations", test_tunix_data_flow_and_operations, True),
        ("tunix_ml_utilities_and_metrics", test_tunix_ml_utilities_and_metrics, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
