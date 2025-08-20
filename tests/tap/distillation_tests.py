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

"""TAP tests for Tunix distillation functionality."""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Any, Callable


def test_distillation_core_imports() -> bool:
    """Test that all core distillation modules can be imported."""
    try:
        from tunix.distillation import distillation_trainer
        from tunix.distillation.feature_extraction import (
            pooling, projection, sowed_module
        )
        from tunix.distillation.strategies import (
            base_strategy, logit, attention, feature_pooling, feature_projection
        )
        
        return True
    except ImportError:
        return False


def test_distillation_strategies_and_interface() -> bool:
    """Test distillation strategies and base interface."""
    try:
        from tunix.distillation.strategies import base_strategy
        from tunix.distillation.strategies import logit
        from tunix.distillation.strategies import feature_pooling
        from tunix.distillation.strategies import feature_projection
        
        # Test that BaseStrategy is abstract
        assert hasattr(base_strategy.BaseStrategy, '__abstractmethods__')
        
        # Test that strategy classes can be imported
        assert hasattr(logit, 'LogitStrategy')
        assert hasattr(feature_pooling, 'FeaturePoolingStrategy')
        assert hasattr(feature_projection, 'FeatureProjectionStrategy')
        
        return True
    except Exception:
        return False


def test_distillation_feature_extraction() -> bool:
    """Test distillation feature extraction functionality."""
    try:
        from tunix.distillation.feature_extraction import sowed_module
        
        # Test that SowedModule can be imported
        assert hasattr(sowed_module, 'SowedModule')
        
        # Test basic loss computation (import test)
        from tunix.distillation.strategies import logit
        assert hasattr(logit, 'LogitStrategy')
        
        # Test distillation trainer
        from tunix.distillation import distillation_trainer
        assert hasattr(distillation_trainer, 'DistillationTrainer')
        
        return True
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of TAP tests for distillation functionality.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("distillation_core_imports", test_distillation_core_imports, True),
        ("distillation_strategies_and_interface", test_distillation_strategies_and_interface, True),
        ("distillation_feature_extraction", test_distillation_feature_extraction, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
