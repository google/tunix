#!/usr/bin/env python3
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

"""Demo script showing how to use the TAP test infrastructure."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.tap.tap_runner import TAPTestRunner


def demo_basic_usage():
    """Demonstrate basic TAP test usage."""
    print("# Demo: Basic TAP Test Usage")
    print("# ==========================")
    
    # Create a TAP test runner
    runner = TAPTestRunner()
    
    # Define some simple tests
    def test_addition():
        return 2 + 2 == 4
    
    def test_multiplication():
        return 3 * 4 == 12
    
    def test_failing_test():
        return 1 == 2  # This will fail
    
    # Run individual tests
    runner.run_test("addition", test_addition, True)
    runner.run_test("multiplication", test_multiplication, True)
    runner.run_test("failing_test", test_failing_test, True)
    
    # Finish and show statistics
    runner.finish()
    print()


def demo_test_suite():
    """Demonstrate running a test suite."""
    print("# Demo: Test Suite Execution")
    print("# ==========================")
    
    # Create a test suite
    test_suite = [
        ("test_1", lambda: True, True),
        ("test_2", lambda: True, True),
        ("test_3", lambda: False, True),  # This will fail
        ("test_4", lambda: "hello" == "hello", True),
        ("test_5", lambda: len([1, 2, 3]) == 3, True),
    ]
    
    # Run the test suite
    runner = TAPTestRunner()
    stats = runner.run_test_suite(test_suite)
    runner.finish()
    
    print(f"# Test suite completed with {stats['passed']} passed and {stats['failed']} failed")
    print()


def demo_error_handling():
    """Demonstrate error handling in tests."""
    print("# Demo: Error Handling")
    print("# ====================")
    
    runner = TAPTestRunner()
    
    # Test that raises an exception
    def test_with_exception():
        return 1 / 0  # This will raise ZeroDivisionError
    
    # Test that returns wrong type
    def test_wrong_type():
        return "not a boolean"
    
    runner.run_test("exception_test", test_with_exception, True)
    runner.run_test("wrong_type_test", test_wrong_type, True)
    runner.finish()
    print()


def demo_with_plan():
    """Demonstrate using TAP with a predefined plan."""
    print("# Demo: TAP with Plan")
    print("# ===================")
    
    # Create runner with a plan
    runner = TAPTestRunner(plan=3)
    
    # Run exactly 3 tests
    runner.run_test("planned_test_1", lambda: True, True)
    runner.run_test("planned_test_2", lambda: True, True)
    runner.run_test("planned_test_3", lambda: True, True)
    
    runner.finish()
    print()


def main():
    """Run all demos."""
    print("TAP Test Infrastructure Demo")
    print("============================")
    print()
    
    demo_basic_usage()
    demo_test_suite()
    demo_error_handling()
    demo_with_plan()
    
    print("# Demo completed!")
    print("# To run actual Tunix tests, use:")
    print("#   python tests/tap/run_all_tap_tests.py")
    print("#   python tests/tap/simple_tests.py")


if __name__ == "__main__":
    main()
