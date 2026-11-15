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

"""Deterministic two-module repair tasks; no generated code is executed.

Files contain a deliberately bounded JSON expression DSL. Public specifications
describe each module. Hidden inputs and oracle execution remain on the grader
side. This task is a synthetic public adaptation, not an Avalon-M1 benchmark.
"""

import dataclasses
import hashlib
import json
import math
import random

FILES = ("module_a.json", "module_b.json")
OPS = ("identity", "scale", "shift", "square", "absolute")
VERSION = "modular-repair-v1"
SPLIT_FAMILIES = {
    "train": ("affine",),
    "dev": ("shift_scale",),
    "test": ("quadratic", "absolute_scale"),
}


def parse_module(value):
  if not isinstance(value, dict) or set(value) != {"op", "value"}:
    raise ValueError("A module contains exactly op and value.")
  if (
      value["op"] not in OPS
      or type(value["value"]) is not int
      or abs(value["value"]) > 100
  ):
    raise ValueError("Unsupported operation or bounded integer parameter.")
  return {"op": value["op"], "value": value["value"]}


def execute(modules, x):
  if set(modules) != set(FILES):
    raise ValueError("A submission must contain exactly both module files.")
  value = float(x)
  for name in FILES:
    module = parse_module(modules[name])
    op, arg = module["op"], module["value"]
    if op == "scale":
      value *= arg
    elif op == "shift":
      value += arg
    elif op == "square":
      value *= value
    elif op == "absolute":
      value = abs(value)
    if not math.isfinite(value):
      raise ValueError("Nonfinite DSL result.")
  return value


@dataclasses.dataclass(frozen=True)
class RepairTask:
  task_id: str
  split: str
  family: str
  requirements: tuple[tuple[str, int], ...]
  hidden_inputs: tuple[int, ...]

  def oracle_modules(self):
    """Private grader representation; never included in an agent observation."""
    return {
        name: {"op": spec[0], "value": spec[1]}
        for name, spec in zip(FILES, self.requirements, strict=True)
    }

  def public(self):
    descriptions = {
        "identity": "Return the input unchanged.",
        "scale": "Multiply the input by {value}.",
        "shift": "Add {value} to the input.",
        "square": "Return the square of the input.",
        "absolute": "Return the nonnegative magnitude of the input.",
    }
    return {
        "task_id": self.task_id,
        "description": (
            "Repair both JSON modules. Apply module_a followed by module_b to"
            " each input."
        ),
        "requirements": {
            name: descriptions[spec[0]].format(value=spec[1])
            for name, spec in zip(FILES, self.requirements, strict=True)
        },
        "public_examples": [
            {"input": x, "output": execute(self.oracle_modules(), x)}
            for x in (-2, 0, 3)
        ],
        "files": list(FILES),
        "dsl": (
            "identity(x)=x; scale(x,v)=x*v; shift(x,v)=x+v; square(x)=x*x;"
            " absolute(x)=abs(x)"
        ),
    }

  def grade(self, final_text):
    try:
      submitted = json.loads(final_text)
      target = self.oracle_modules()
      return float(
          all(
              execute(submitted, x) == execute(target, x)
              for x in self.hidden_inputs
          )
      )
    except (ValueError, TypeError, KeyError, OverflowError):
      return 0.0


def tasks(split, *, count=8, seed=42):
  if split not in SPLIT_FAMILIES or count < 1:
    raise ValueError("Use a named split and a positive task count.")
  output = []
  offset = {"train": 0, "dev": 100000, "test": 200000}[split]
  for index in range(count):
    rng = random.Random(seed + offset + index)
    family = SPLIT_FAMILIES[split][index % len(SPLIT_FAMILIES[split])]
    a, b = rng.randint(2, 7), rng.choice((-5, -3, 3, 5))
    specs = {
        "affine": (("scale", a), ("shift", b)),
        "shift_scale": (("shift", b), ("scale", a)),
        "quadratic": (("square", 0), ("shift", b)),
        "absolute_scale": (("absolute", 0), ("scale", a)),
    }
    inputs = tuple(
        rng.sample([x for x in range(-30, 31) if x not in (-2, 0, 3)], 16)
    )
    output.append(
        RepairTask(
            f"{split}/{family}/{index}", split, family, specs[family], inputs
        )
    )
  return output


def manifest(task_list, seed):
  serialized = json.dumps(
      [dataclasses.asdict(t) for t in task_list], sort_keys=True
  )
  return {
      "generator": VERSION,
      "seed": seed,
      "tasks": len(task_list),
      "families": sorted({t.family for t in task_list}),
      "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
  }
