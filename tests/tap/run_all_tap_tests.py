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

"""Main script to run all TAP tests for Tunix."""

import argparse
import sys
from typing import List

from tests.tap.tap_runner import TAPTestRunner, run_tap_tests


def main():
    """Main function to run TAP tests."""
    parser = argparse.ArgumentParser(description="Run TAP tests for Tunix")
    parser.add_argument(
        "--test-modules",
        nargs="+",
        default=[
            "tests.tap.basic_tests",
            "tests.tap.model_tests",
            "tests.tap.rl_tests",
            "tests.tap.sft_tests",
            "tests.tap.distillation_tests",
            "tests.tap.generation_tests",
        ],
        help="List of test modules to run",
    )
    parser.add_argument(
        "--plan",
        type=int,
        help="Total number of tests to run (TAP plan)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all available tests without running them",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"# Running TAP tests for Tunix")
        print(f"# Test modules: {args.test_modules}")
        print(f"# Plan: {args.plan}")
    
    if args.list_tests:
        print("# Available TAP test modules:")
        for module in args.test_modules:
            print(f"#   {module}")
        return 0
    
    try:
        run_tap_tests(args.test_modules)
        return 0
    except Exception as e:
        print(f"# Error running TAP tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
