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

#!/bin/bash

set -eo pipefail

BASE_DIR="${BASE_DIR:-${WORK_DIR:-/mnt/disks}}"
GITHUB_ROOT="${GITHUB_ROOT:-$BASE_DIR/github}"
REPO_ROOT="${REPO_ROOT:-$GITHUB_ROOT/.cache/.repos}"
LOG_ROOT="${LOG_ROOT:-$GITHUB_ROOT/.log}"

mkdir -p "$REPO_ROOT"
mkdir -p "$LOG_ROOT"

cd "$REPO_ROOT"

echo >"$LOG_ROOT/build-tpu-sync.log"

if [ ! -d "tpu-sync" ]; then
    echo "clone tpu-sync: $REPO_ROOT/tpu-sync"
    git clone https://github.com/google/tpu-sync.git \
      &>>"$LOG_ROOT/build-tpu-sync.log" || { tail "$LOG_ROOT/build-tpu-sync.log"; echo "clone tpu-sync fail."; exit 1; }
fi

cd "$REPO_ROOT/tpu-sync"
echo "tpu-sync: $(pwd)"

if ! command -v bazel &> /dev/null; then
    echo "install bazel: /usr/local/bin/bazel"
    sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/8.6.0/bazel-8.6.0-linux-x86_64
    sudo chmod +x /usr/local/bin/bazel
fi
echo "bazel: $(command -v bazel)"

if [ -z "$BAZEL_OUTPUT_BASE" ]; then
    export BAZEL_OUTPUT_BASE="$GITHUB_ROOT/.cache/bazel/output"
fi
if [[ "$BAZEL_OUTPUT_BASE" =~ /.cache/bazel/ ]]; then
    rm -rf "$BAZEL_OUTPUT_BASE"
fi
echo "BAZEL_OUTPUT_BASE: $BAZEL_OUTPUT_BASE"

if [[ "$1" == "-e" ]]; then
  shift
  ./build.sh jax \
    &>>"$LOG_ROOT/build-tpu-sync.log" || { tail "$LOG_ROOT/build-tpu-sync.log"; echo "build fail."; exit 1; }
else
  ./build.sh jax //ci/wheel:raiden_jax_wheel \
    &>>"$LOG_ROOT/build-tpu-sync.log" || { tail "$LOG_ROOT/build-tpu-sync.log"; echo "build fail."; exit 1; }
fi

if [[ "$1" == "--test" ]]; then
  ./run_tests.sh jax
fi

echo "Done."
