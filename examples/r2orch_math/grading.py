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

"""Run the pinned original AIME scorer without importing its training stack.

The upstream math extraction contract is substantially broader than its
separate strict AIME submission parser. Keeping this explicit source dependency
avoids silently changing the benchmark reward while porting the trainer.
"""

import ast
import functools
from pathlib import Path
import subprocess
import types

from examples.r2orch_math.prepare_data import SOURCE_COMMIT


@functools.lru_cache(maxsize=4)
def load_original_aime_grader(source):
  """Load only the transitive definitions used by upstream _integer_score.

  Read the exact committed object, never an editable working-tree Python file.
  The source checkout is an explicit dependency supplied by the operator.
  No upstream tools, environment execution, or training entrypoint is imported.
  """
  source = Path(source).resolve()
  revision = subprocess.check_output(
      ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
  ).strip()
  if revision != SOURCE_COMMIT:
    raise ValueError(f"The original scorer requires revision {SOURCE_COMMIT}.")
  filename = "src/r2orch/benchmark_scoring.py"
  code = subprocess.check_output(
      ["git", "-C", str(source), "show", f"{SOURCE_COMMIT}:{filename}"],
      text=True,
  )
  tree = ast.parse(code, filename=filename)
  definitions = {}
  for node in tree.body:
    names = []
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
      names = [node.name]
    elif isinstance(node, ast.Assign):
      names = [
          n.id
          for target in node.targets
          for n in ast.walk(target)
          if isinstance(n, ast.Name)
      ]
    elif isinstance(node, ast.Import):
      names = [a.asname or a.name.split(".")[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom) and not node.level:
      names = [a.asname or a.name for a in node.names]
    for name in names:
      definitions[name] = node
  selected, pending = set(), ["_integer_score"]
  while pending:
    node = definitions.get(pending.pop())
    if node is None or id(node) in selected:
      continue
    selected.add(id(node))
    pending.extend(
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    )
  module = ast.Module(
      body=[
          ast.ImportFrom(
              module="__future__",
              names=[ast.alias(name="annotations")],
              level=0,
          )
      ]
      + [node for node in tree.body if id(node) in selected],
      type_ignores=[],
  )
  module = ast.fix_missing_locations(module)
  namespace = types.ModuleType("_tunix_pinned_r2orch_math_scorer")
  exec(compile(module, filename, "exec"), namespace.__dict__)  # pylint: disable=exec-used

  def grade(text, answer):
    try:
      return bool(namespace._integer_score(text, str(answer)))  # pylint: disable=protected-access
    except (ValueError, ArithmeticError, RecursionError):
      return False

  return grade
