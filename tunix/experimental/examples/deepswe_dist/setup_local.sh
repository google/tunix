#!/usr/bin/env bash
# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run inside an existing Tunix + vLLM/TPU inference + MaxText environment.
set -euo pipefail
"${PYTHON:-python}" -m pip install \
  docker kubernetes gym swebench==3.0.2 fire simple-parsing unidiff
"${PYTHON:-python}" -m pip install --no-deps qwix==0.1.8 \
  'git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735'

# Same compatibility fixes as the repository's INSTALL_DEEPSWE_DEPS image.
# --no-deps above avoids R2E-Gym's old datasets pin replacing the Tunix stack.
"${PYTHON:-python}" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

root = Path(next(iter(find_spec("r2egym").submodule_search_locations)))
patches = [
    ("agenthub/utils/utils.py", "create_repo, upload_folder, HfFolder",
     "create_repo, upload_folder"),
    ("agenthub/runtime/docker.py",
     "self.commit = ParsedCommit(**json.loads(self.commit_json))",
     "self.commit = ParsedCommit(**(json.loads(self.commit_json) "
     "if isinstance(self.commit_json, str) else self.commit_json))"),
]
for relative, old, new in patches:
    path = root / relative
    content = path.read_text()
    if old in content:
        path.write_text(content.replace(old, new))
    elif new not in content:
        raise RuntimeError(f"Unexpected R2E-Gym source in {path}")
PY
