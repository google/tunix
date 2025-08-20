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

"""TAP tests for Tunix SFT (Supervised Fine-Tuning) functionality."""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Any, Callable


def test_sft_core_imports() -> bool:
    """Test that all core SFT modules can be imported."""
    try:
        from tunix.sft import (
            checkpoint_manager, hooks, inflight_throttler, metrics_logger,
            peft_trainer, profiler, progress_bar, system_metrics_calculator
        )
        
        return True
    except ImportError:
        return False


def test_sft_configurations() -> bool:
    """Test SFT configuration classes and setup."""
    try:
        from tunix.sft.peft_trainer import TrainingConfig
        from tunix.sft.metrics_logger import MetricsLoggerOptions
        from tunix.sft.checkpoint_manager import CheckpointManager
        from tunix.sft.profiler import ProfilerOptions
        import orbax.checkpoint as ocp
        
        # Test PEFT training config
        peft_config = TrainingConfig(
            eval_every_n_steps=100,
            max_steps=1000,
            gradient_accumulation_steps=4,
        )
        assert peft_config.max_steps == 1000
        
        # Test metrics logger config
        metrics_config = MetricsLoggerOptions(
            log_dir="/tmp/test_logs",
            flush_every_n_steps=10,
        )
        assert metrics_config.flush_every_n_steps == 10
        
        # Test profiler config
        profiler_config = ProfilerOptions(
            log_dir="/tmp/profiler",
            skip_first_n_steps=10,
            profiler_steps=50,
        )
        assert profiler_config.log_dir == "/tmp/profiler"
        
        return True
    except Exception:
        return False


def test_sft_functionality() -> bool:
    """Test SFT functionality and operations."""
    try:
        from tunix.sft import system_metrics_calculator
        from tunix.sft import metrics_logger
        from tunix.sft import hooks
        
        # Test system metrics calculation
        tflops = system_metrics_calculator.tflops(
            total_model_params=1_000_000_000,
            global_batch_size=32,
            step_time_delta=0.5,
        )
        assert isinstance(tflops, float)
        assert tflops >= 0.0
        
        # Test metrics logger operations
        options = metrics_logger.MetricsLoggerOptions(
            log_dir="/tmp/test_logs",
            flush_every_n_steps=10,
        )
        logger = metrics_logger.MetricsLogger(options)
        logger.log("loss", jnp.array(0.5), metrics_logger.Mode.TRAIN, 1)
        
        # Test hooks interface
        assert hasattr(hooks.TrainingHooks, '__abstractmethods__')
        assert hasattr(hooks.DataHooks, '__abstractmethods__')
        
        return True
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of TAP tests for SFT functionality.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("sft_core_imports", test_sft_core_imports, True),
        ("sft_configurations", test_sft_configurations, True),
        ("sft_functionality", test_sft_functionality, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
