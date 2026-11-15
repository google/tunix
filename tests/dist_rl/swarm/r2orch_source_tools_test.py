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

"""Container transport boundaries; these do not claim Docker execution evidence."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from examples.r2orch_math import source_tools


def tool():
  return source_tools.ContainerSourceTool(
      name="python_exec",
      description="Python",
      parameters={},
      source_root="/tmp/pinned-source/src",
      image="sha256:" + "a" * 64,
  )


def test_tool_requires_pinned_image_before_source_loading():
  with pytest.raises(ValueError, match="digest"):
    source_tools.original_aime_tools("unused", image="python:latest")


def test_tool_transmits_stdin_and_has_no_writable_host_or_network(monkeypatch):
  observed = []

  def run(argv, **kwargs):
    observed.append((argv, kwargs))
    return SimpleNamespace(returncode=0, stdout='{"stdout":"2"}', stderr="")

  monkeypatch.setattr(subprocess, "run", run)
  result = tool().execute({"code": "print(1+1)", "purpose": "compute"})
  assert json.loads(result)["stdout"] == "2"
  argv, kwargs = observed[0]
  assert argv[:3] == ["docker", "run", "-i"]
  assert (
      "--pull=never" in argv
      and "--network=none" in argv
      and "--read-only" in argv
  )
  assert "--cap-drop=ALL" in argv and "--user=65534:65534" in argv
  assert argv[argv.index("--mount") + 1].endswith("target=/opt/source,readonly")
  assert argv[-2] == "-c" and argv[-1] == source_tools._ENTRYPOINT
  assert json.loads(kwargs["input"])["arguments"]["code"] == "print(1+1)"
  assert kwargs["timeout"] == 15.0


def test_timed_out_container_is_removed_not_only_client_killed(monkeypatch):
  observed = []

  def run(argv, **kwargs):
    observed.append(argv)
    if argv[1] == "run":
      raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(subprocess, "run", run)
  with pytest.raises(TimeoutError, match="wall-clock"):
    tool().execute({"code": "while True: pass", "purpose": "compute"})
  name = observed[0][observed[0].index("--name") + 1]
  assert observed[1] == ["docker", "rm", "--force", name]


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "a" * 16001)])
def test_failed_or_oversized_tool_output_is_not_a_success(
    monkeypatch, returncode, stdout
):
  monkeypatch.setattr(
      subprocess,
      "run",
      lambda *a, **k: SimpleNamespace(
          returncode=returncode, stdout=stdout, stderr="container error"
      ),
  )
  with pytest.raises((ValueError, RuntimeError)):
    tool().execute({"code": "print(2)", "purpose": "compute"})


def test_real_isolated_original_tools_when_explicitly_configured():
  source = os.environ.get("TUNIX_R2ORCH_SOURCE")
  image = os.environ.get("TUNIX_R2ORCH_TOOL_IMAGE")
  if not source or not image:
    pytest.skip(
        "Set source and immutable tool image to run actual isolated tools."
    )
  tools = source_tools.original_aime_tools(source, image=image)
  python = json.loads(
      tools["python_exec"].execute({"code": "print(1+1)", "purpose": "compute"})
  )
  assert python["status"] == "ok" and python["stdout"].strip() == "2"
  symbolic = json.loads(
      tools["symbolic_compute"].execute(
          {"operation": "evaluate", "expressions": ["1+1"]}
      )
  )
  assert symbolic["status"] == "ok" and symbolic["result"] == ["2"]
  finite = json.loads(
      tools["finite_search"].execute({
          "variables": [{"name": "x", "min": 0, "max": 3}],
          "condition": "x*x == 4",
          "return_expressions": ["x"],
      })
  )
  assert finite["status"] == "ok" and finite["result"]["matches"] == [[2]]
