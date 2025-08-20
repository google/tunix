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

"""Test script to verify TAP test infrastructure."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.tap.tap_runner import TAPTestRunner


def test_tap_runner_creation():
    """Test that TAP runner can be created."""
    try:
        runner = TAPTestRunner()
        return True
    except Exception:
        return False


def test_tap_runner_with_plan():
    """Test that TAP runner can be created with a plan."""
    try:
        runner = TAPTestRunner(plan=5)
        return True
    except Exception:
        return False


def test_simple_test_execution():
    """Test that a simple test can be executed."""
    try:
        runner = TAPTestRunner()
        result = runner.run_test("simple_test", lambda: True, True)
        return result
    except Exception:
        return False


def test_failing_test_execution():
    """Test that a failing test is handled correctly."""
    try:
        runner = TAPTestRunner()
        result = runner.run_test("failing_test", lambda: False, True)
        return not result  # Should return False for failing test
    except Exception:
        return False


def test_exception_handling():
    """Test that exceptions in tests are handled correctly."""
    try:
        runner = TAPTestRunner()
        result = runner.run_test("exception_test", lambda: 1/0, True)
        return not result  # Should return False for exception
    except Exception:
        return False


def test_test_suite_execution():
    """Test that a test suite can be executed."""
    try:
        runner = TAPTestRunner()
        test_suite = [
            ("test1", lambda: True, True),
            ("test2", lambda: True, True),
            ("test3", lambda: False, True),
        ]
        stats = runner.run_test_suite(test_suite)
        return stats['total'] == 3 and stats['passed'] == 2 and stats['failed'] == 1
    except Exception:
        return False


def get_tap_tests():
    """Return list of TAP tests for infrastructure verification."""
    return [
        ("tap_runner_creation", test_tap_runner_creation, True),
        ("tap_runner_with_plan", test_tap_runner_with_plan, True),
        ("simple_test_execution", test_simple_test_execution, True),
        ("failing_test_execution", test_failing_test_execution, True),
        ("exception_handling", test_exception_handling, True),
        ("test_suite_execution", test_test_suite_execution, True),
    ]


if __name__ == "__main__":
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
