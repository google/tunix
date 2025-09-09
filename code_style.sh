#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Clean up Python codes using Pylint & Pyink
# Googlers: please run `pip install pylint --force; pip install pyink==23.10.0` in advance

set -e # Exit immediately if any command fails

# Get all python files that are modified or added.
# We include ipynb files as well, as pyink can format them.
FILES_TO_FORMAT=$(git diff --name-only --diff-filter=d HEAD -- "*.py" "*.ipynb" ; git ls-files --others --exclude-standard -- "*.py" "*.ipynb")

if [ -z "$FILES_TO_FORMAT" ]; then
  echo "No Python or Notebook files to format."
  exit 0
fi

LINE_LENGTH=80
if [ -f pylintrc ]; then
  LINE_LENGTH_FROM_FILE=$(grep -E "^max-line-length=" pylintrc | cut -d '=' -f 2)
  if [ -n "$LINE_LENGTH_FROM_FILE" ]; then
    LINE_LENGTH=$LINE_LENGTH_FROM_FILE
  fi
fi

# Check for --check flag
CHECK_ONLY_PYINK_FLAGS=""
if [[ "$1" == "--check" ]]; then
  CHECK_ONLY_PYINK_FLAGS="--check --diff --color"
fi

echo "Formatting files: ${FILES_TO_FORMAT}"
# The tools take a list of files, so no loop is needed.
pyink $FILES_TO_FORMAT ${CHECK_ONLY_PYINK_FLAGS} --pyink-indentation=2 --line-length=${LINE_LENGTH}

# The original script ran pylint, but without a configuration, it can be noisy.
# If you have a pylintrc, you can uncomment the following lines.
# echo "Linting files: ${FILES_TO_FORMAT}"
# pylint --disable C0114,R0401,R0917,W0201,W0613 $FILES_TO_FORMAT

echo "Successfully clean up all codes."
