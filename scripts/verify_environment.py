#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-time Python environment verifier for Tunix container images."""

import argparse
import importlib
import sys
import traceback


def check_module(module_name: str, check_tunix_version: bool = False) -> bool:
  """Imports a module and optionally asserts a non-dev Tunix version."""
  try:
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", "installed")
    print(f"  [OK] {module_name:<30} - v{version}")
    if (
        check_tunix_version
        and module_name == "tunix"
        and version == "0.0.0.dev0"
    ):
      print("  [FAIL] Tunix version is '0.0.0.dev0' (package metadata unset).")
      return False
    return True
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"  [FAIL] {module_name:<30} - ERROR: {exc}")
    traceback.print_exc(file=sys.stdout)
    return False


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Verify Python module imports and Tunix package metadata."
  )
  parser.add_argument(
      "--packages",
      nargs="+",
      required=True,
      help="List of Python modules to import and verify.",
  )
  parser.add_argument(
      "--check-tunix-version",
      action="store_true",
      help="Assert that tunix.__version__ is not '0.0.0.dev0'.",
  )
  args = parser.parse_args()

  print("\n========================================================")
  print(" Tunix Container Environment Verification")
  print(f" Python runtime: {sys.version.split()[0]}")
  print("========================================================\n")

  failed = [
      pkg
      for pkg in args.packages
      if not check_module(pkg, check_tunix_version=args.check_tunix_version)
  ]
  if failed:
    print(f"\nVerification FAILED (failed modules: {failed}).")
    sys.exit(1)

  print("\nAll requested modules verified successfully.")


if __name__ == "__main__":
  main()
