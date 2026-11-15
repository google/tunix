# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the exact original AIME tools in an explicitly configured container.

The image must already be installed and pinned by digest. It needs Python 3.11+
and SymPy, matching the original tool dependencies. There is no host-Python
fallback, network access, writable host mount or implicit image pull.
"""

import dataclasses
import json
from pathlib import Path
import re
import subprocess
import uuid

from examples.r2orch_math import source_adapter

_ENTRYPOINT = """import json, sys
sys.path.insert(0, '/opt/source')
from r2orch import agent_tools
request = json.load(sys.stdin)
classes = {'python_exec': agent_tools.PythonExecutionTool,
           'symbolic_compute': agent_tools.SymbolicComputeTool,
           'finite_search': agent_tools.FiniteSearchTool}
tool = classes[request['name']]()
sys.stdout.write(tool.execute(request['arguments']))
"""


@dataclasses.dataclass(frozen=True)
class ContainerSourceTool:
  """An original tool schema with a bounded, isolated execution transport."""

  name: str
  description: str
  parameters: dict
  source_root: str
  image: str
  timeout_seconds: float = 15.0
  max_output_chars: int = 16000
  stateful: bool = False

  def execute(self, arguments):
    payload = json.dumps({"name": self.name, "arguments": arguments})
    if len(payload) > 100000:
      raise ValueError("AIME tool arguments exceed the container input budget.")
    container = "tunix-r2orch-tool-" + uuid.uuid4().hex
    command = [
        "docker",
        "run",
        "-i",
        "--rm",
        "--pull=never",
        "--name",
        container,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=1g",
        "--memory-swap=1g",
        "--cpus=1",
        "--user=65534:65534",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=64m",
        "--workdir=/tmp",
        "--mount",
        f"type=bind,source={self.source_root},target=/opt/source,readonly",
        "--entrypoint=python",
        self.image,
        "-I",
        "-c",
        _ENTRYPOINT,
    ]
    try:
      completed = subprocess.run(
          command,
          input=payload,
          capture_output=True,
          text=True,
          timeout=self.timeout_seconds,
          check=False,
      )
    except subprocess.TimeoutExpired:
      # Killing only the Docker client does not stop the isolated workload.
      # Remove the exact invocation we own, then propagate an explicit failure.
      subprocess.run(
          ["docker", "rm", "--force", container],
          capture_output=True,
          timeout=5,
          check=False,
      )
      raise TimeoutError(
          "The isolated AIME tool exceeded its wall-clock budget."
      ) from None
    if completed.returncode != 0:
      raise RuntimeError(
          "The isolated AIME tool failed: " + completed.stderr[-2000:]
      )
    if len(completed.stdout) > self.max_output_chars:
      raise ValueError("The isolated AIME tool exceeded its output budget.")
    return completed.stdout


def original_aime_tools(source, *, image: str):
  """Return all three upstream AIME tool schemas using isolated execution."""
  if not re.fullmatch(
      r"(?:[A-Za-z0-9][A-Za-z0-9_./:-]*@)?sha256:[a-f0-9]{64}", image
  ):
    raise ValueError(
        "The tool image must be pinned by a complete sha256 digest."
    )
  module = source_adapter.load_source_module(source, "agent_tools")
  source_root = str(Path(module.__file__).parents[1])
  tools = {}
  for cls in (
      module.PythonExecutionTool,
      module.SymbolicComputeTool,
      module.FiniteSearchTool,
  ):
    original = cls()
    tools[original.name] = ContainerSourceTool(
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        source_root=source_root,
        image=image,
    )
  return tools
