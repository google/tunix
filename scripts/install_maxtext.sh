#!/bin/bash

# Copyright 2026 Google LLC
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

# Installs MaxText (the `MaxTextTrainingEngine` trainer backend) and
# `maxtext_vllm_adapter` (the vLLM plugin that registers `MaxTextForCausalLM`,
# so the rollout runs MaxText's model code too).
#
# WHICH REF
# ---------
# MaxText and Tunix are separate repositories on separate release cadences, and
# the Trellis trainer engine currently lands on MaxText main several times a
# week. So the ref is a knob, not a constant:
#
#   MAXTEXT_REF=main         (default) test against MaxText HEAD. Catches
#                            cross-repo breakage on the day it lands, at the
#                            cost of a CI signal that can go red for reasons
#                            outside this repo.
#   MAXTEXT_REF=<sha|tag>    pin. Use this once the engine stabilises and a
#                            nightly tag exists.
#   MAXTEXT_REF=pinned       use requirements/maxtext_requirements.txt verbatim,
#   (or the empty string)    i.e. the commit already pinned in that file.
#
# The CI workflow plumbs this through a `maxtext_ref` workflow input so a pin
# can be applied without touching this script.
#
# PIN PROTECTION
# --------------
# MaxText's dependency closure overlaps the container's: it can drag jax,
# jaxlib, libtpu, numpy or torch off the versions vLLM and tpu-inference were
# built against. The trainer and the rollout share this interpreter, so a silent
# bump here surfaces much later as an opaque TPU runtime or Numba failure.
# Versions of those packages are therefore captured before the install and
# restored after it, loudly.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

MAXTEXT_REPO=${MAXTEXT_REPO:-"https://github.com/AI-Hypercomputer/maxtext.git"}
MAXTEXT_REF=${MAXTEXT_REF-main}
MAXTEXT_REQUIREMENTS=${MAXTEXT_REQUIREMENTS:-"${ROOT_DIR}/requirements/maxtext_requirements.txt"}

# Packages whose versions the TPU runtime, vLLM and tpu-inference agree on.
# MaxText must fit around them, not the other way round.
PROTECTED_PACKAGES=(jax jaxlib libtpu libtpu-nightly numpy torch vllm tpu-inference)

if [[ ! -f "${MAXTEXT_REQUIREMENTS}" ]]; then
  echo "Error: requirements file not found: ${MAXTEXT_REQUIREMENTS}" >&2
  exit 1
fi

WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

REQ_FILE="${WORK_DIR}/maxtext_requirements.txt"
if [[ "${MAXTEXT_REF}" == "pinned" ]]; then
  MAXTEXT_REF=""
fi
if [[ -n "${MAXTEXT_REF}" ]]; then
  # Rewrite every `git+<url>@<ref>` to the requested repo and ref, leaving the
  # `#subdirectory=` fragment (which selects the vLLM adapter) intact.
  sed -E "s%git\+[^[:space:]#]+%git+${MAXTEXT_REPO}@${MAXTEXT_REF}%" \
    "${MAXTEXT_REQUIREMENTS}" > "${REQ_FILE}"
  echo "Installing MaxText from ${MAXTEXT_REPO}@${MAXTEXT_REF}"
else
  cp "${MAXTEXT_REQUIREMENTS}" "${REQ_FILE}"
  echo "Installing MaxText from the commit pinned in ${MAXTEXT_REQUIREMENTS}"
fi
grep -E '^(maxtext|maxtext-vllm-adapter) ' "${REQ_FILE}" || true

record_versions() {
  python3 - "$@" <<'PY'
import importlib.metadata as md
import sys

for name in sys.argv[1:]:
  try:
    print(f"{name} {md.version(name)}")
  except md.PackageNotFoundError:
    pass
PY
}

BEFORE="${WORK_DIR}/before.txt"
AFTER="${WORK_DIR}/after.txt"
record_versions "${PROTECTED_PACKAGES[@]}" > "${BEFORE}"
echo "Protected package versions before the install:"
sed 's/^/  /' "${BEFORE}"

python3 -m pip install -r "${REQ_FILE}"

record_versions "${PROTECTED_PACKAGES[@]}" > "${AFTER}"

# Restore anything the install moved. `--no-deps` so the restore cannot itself
# start a second round of resolution; only packages that were present before are
# touched, so additive installs (e.g. a torch MaxText genuinely needs and the
# image lacked) are left alone.
RESTORED=0
while read -r name version; do
  after_version=$(awk -v n="${name}" '$1 == n {print $2}' "${AFTER}")
  if [[ -n "${after_version}" && "${after_version}" != "${version}" ]]; then
    echo "MaxText moved ${name} ${version} -> ${after_version}; restoring ${version}."
    # </dev/null: this loop's stdin is the version list, and pip would eat it.
    python3 -m pip install --force-reinstall --no-deps "${name}==${version}" </dev/null
    RESTORED=1
  fi
done < "${BEFORE}"
if [[ "${RESTORED}" == "0" ]]; then
  echo "No protected package versions changed."
fi

# Both halves of the stack import from this interpreter, so assert both here
# rather than discovering it inside a backgrounded trainer or rollout process,
# where the traceback lands in a log file nobody reads until the job times out.
# JAX_PLATFORMS=cpu keeps this from claiming the TPU chips the test needs.
JAX_PLATFORMS=cpu python3 - <<'PY'
import importlib.metadata as md

import jax

print(f"jax {jax.__version__}")
for name in ("jaxlib", "libtpu", "numpy", "vllm", "tpu-inference", "maxtext"):
  try:
    print(f"{name} {md.version(name)}")
  except md.PackageNotFoundError:
    print(f"{name} MISSING")

# Trainer side: the engine Tunix's maxtext backend instantiates.
from maxtext.configs import pyconfig  # noqa: F401
from maxtext.training_engine import maxtext_engine  # noqa: F401

# Rollout side: the vLLM general plugin that registers MaxTextForCausalLM. vLLM
# loads it by entry point, which swallows import errors into a warning, so a
# broken install would otherwise show up as the rollout quietly running the
# stock Qwen3 model and Raiden matching zero tensors by name.
import maxtext_vllm_adapter  # noqa: F401

assert hasattr(maxtext_vllm_adapter, "register")
print("MaxText trainer engine and vLLM adapter import cleanly.")
PY
