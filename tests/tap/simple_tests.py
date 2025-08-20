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

"""Simple TAP tests that don't require external dependencies."""

import os
import sys
from typing import List, Tuple, Any, Callable


def test_python_version() -> bool:
    """Test that Python version is 3.10 or higher."""
    try:
        version = sys.version_info
        return version.major == 3 and version.minor >= 10
    except Exception:
        return False


def test_os_availability() -> bool:
    """Test that OS module is available."""
    try:
        return os.name in ['posix', 'nt', 'java']
    except Exception:
        return False


def test_sys_availability() -> bool:
    """Test that sys module is available."""
    try:
        return hasattr(sys, 'version')
    except Exception:
        return False


def test_basic_math() -> bool:
    """Test basic mathematical operations."""
    try:
        result = 2 + 2
        return result == 4
    except Exception:
        return False


def test_string_operations() -> bool:
    """Test basic string operations."""
    try:
        test_str = "hello world"
        upper_str = test_str.upper()
        return upper_str == "HELLO WORLD"
    except Exception:
        return False


def test_list_operations() -> bool:
    """Test basic list operations."""
    try:
        test_list = [1, 2, 3, 4, 5]
        sum_result = sum(test_list)
        return sum_result == 15
    except Exception:
        return False


def test_dict_operations() -> bool:
    """Test basic dictionary operations."""
    try:
        test_dict = {"a": 1, "b": 2, "c": 3}
        keys = list(test_dict.keys())
        return len(keys) == 3 and "a" in keys
    except Exception:
        return False


def test_file_operations() -> bool:
    """Test basic file operations."""
    try:
        # Test that we can create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
            f.write("test")
            f.flush()
            # File should exist
            return os.path.exists(f.name)
    except Exception:
        return False


def test_import_system() -> bool:
    """Test that the import system works."""
    try:
        import math
        result = math.sqrt(4)
        return result == 2.0
    except Exception:
        return False


def test_exception_handling() -> bool:
    """Test that exception handling works."""
    try:
        try:
            # This should raise an exception
            result = 1 / 0
            return False  # Should not reach here
        except ZeroDivisionError:
            return True  # Exception was caught correctly
    except Exception:
        return False


def get_tap_tests() -> List[Tuple[str, Callable[[], Any], Any]]:
    """Return list of simple TAP tests.
    
    Returns:
        List of (test_name, test_func, expected_result) tuples
    """
    return [
        ("python_version", test_python_version, True),
        ("os_availability", test_os_availability, True),
        ("sys_availability", test_sys_availability, True),
        ("basic_math", test_basic_math, True),
        ("string_operations", test_string_operations, True),
        ("list_operations", test_list_operations, True),
        ("dict_operations", test_dict_operations, True),
        ("file_operations", test_file_operations, True),
        ("import_system", test_import_system, True),
        ("exception_handling", test_exception_handling, True),
    ]


if __name__ == "__main__":
    from tests.tap.tap_runner import TAPTestRunner
    
    runner = TAPTestRunner()
    test_suite = get_tap_tests()
    runner.run_test_suite(test_suite)
    runner.finish()
