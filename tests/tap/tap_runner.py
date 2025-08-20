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

"""TAP (Test Anything Protocol) test runner for Tunix."""

import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple


class TAPTestRunner:
    """Test runner that outputs results in TAP (Test Anything Protocol) format.
    
    TAP is a simple text-based protocol for communicating test results.
    Each test produces a line of output in the format:
    ok/not ok <test_number> <test_name> # <optional_diagnostic>
    """
    
    def __init__(self, plan: Optional[int] = None):
        """Initialize the TAP test runner.
        
        Args:
            plan: Total number of tests to run. If None, plan will be output after all tests.
        """
        self.test_count = 0
        self.passed = 0
        self.failed = 0
        self.plan = plan
        self.start_time = time.time()
        
        # Output TAP version
        print("TAP version 13")
        
        # Output plan if provided
        if plan is not None:
            print(f"1..{plan}")
    
    def run_test(self, test_name: str, test_func: Callable[[], Any], 
                 expected_result: Any = True) -> bool:
        """Run a single test and output TAP format result.
        
        Args:
            test_name: Name of the test
            test_func: Function to execute for the test
            expected_result: Expected result (default: True for success)
            
        Returns:
            True if test passed, False otherwise
        """
        self.test_count += 1
        
        try:
            result = test_func()
            passed = result == expected_result
            
            if passed:
                self.passed += 1
                print(f"ok {self.test_count} {test_name}")
            else:
                self.failed += 1
                print(f"not ok {self.test_count} {test_name} # Expected {expected_result}, got {result}")
                
        except Exception as e:
            self.failed += 1
            error_msg = str(e).replace('\n', ' ')
            print(f"not ok {self.test_count} {test_name} # {error_msg}")
            if hasattr(e, '__traceback__'):
                traceback.print_exception(type(e), e, e.__traceback__)
        
        return passed
    
    def run_test_suite(self, test_suite: List[Tuple[str, Callable[[], Any], Any]]) -> Dict[str, int]:
        """Run a suite of tests.
        
        Args:
            test_suite: List of (test_name, test_func, expected_result) tuples
            
        Returns:
            Dictionary with test statistics
        """
        for test_name, test_func, expected_result in test_suite:
            self.run_test(test_name, test_func, expected_result)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get test statistics.
        
        Returns:
            Dictionary with test statistics
        """
        return {
            'total': self.test_count,
            'passed': self.passed,
            'failed': self.failed
        }
    
    def finish(self):
        """Finish the test run and output final statistics."""
        if self.plan is None:
            print(f"1..{self.test_count}")
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        stats = self.get_statistics()
        print(f"# Test run completed in {duration:.2f}s")
        print(f"# Total: {stats['total']}, Passed: {stats['passed']}, Failed: {stats['failed']}")
        
        if stats['failed'] > 0:
            sys.exit(1)


def run_tap_tests(test_modules: List[str]) -> None:
    """Run TAP tests from specified modules.
    
    Args:
        test_modules: List of module names to import and run tests from
    """
    runner = TAPTestRunner()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=['get_tap_tests'])
            if hasattr(module, 'get_tap_tests'):
                test_suite = module.get_tap_tests()
                runner.run_test_suite(test_suite)
            else:
                runner.run_test(f"import_{module_name}", lambda: True, True)
        except ImportError as e:
            runner.run_test(f"import_{module_name}", lambda: False, True)
            print(f"# Failed to import {module_name}: {e}")
    
    runner.finish()


if __name__ == "__main__":
    # Example usage
    test_modules = [
        "tests.tap.basic_tests",
        "tests.tap.model_tests", 
        "tests.tap.rl_tests",
        "tests.tap.sft_tests",
        "tests.tap.distillation_tests"
    ]
    run_tap_tests(test_modules)
