# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Twinned log sites (spec Validation 1.10).

`testdata/events_twins_golden.json` is the golden copy of every twinned site's
log call at the base commit (logger, level, format string and the source of
each argument). The tests check that

  1.10.1 each converted `log_and_emit()` call has the base call's logger,
         level, format string and arguments (so its text line is the same,
         byte for byte, for the same values), and that the text line of each
         site reached by the existing unit tests (driven below with events on)
         is `fmt % args` with the site's own file and function in the record;
  1.10.2 the event's fields are the text arguments by name (`names=`), every
         field is in the event's schema, and each emitted event equals the
         record's arguments;
  1.10.3 no log call with a twinned marker is left outside `log_and_emit()`,
         and the count per marker equals the golden count.

The driven run (`test_1_10_code_paths`) runs existing test modules in a
subprocess with this module loaded as a pytest plugin (`-p`); the plugin hooks
below do nothing unless `EVENTS_TWIN_CAPTURE` is set.
"""

import ast
import collections
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from tunix.experimental import events

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN = json.loads((HERE / "testdata" / "events_twins_golden.json")
                    .read_text(encoding="utf-8"))
SITES = GOLDEN["sites"]
MARKERS = (
    "WEIGHT_SYNC_PHASE", "WEIGHT_SYNC_PROFILE", "[rollout-ft]", "Packed ",
    "TRAINER_STEP_TIMING", "[pipeline] STAGGER", "[pipeline] DISPATCH",
    "[pipeline] ARRIVE", "Appended checkpoint manifest",
    "[DEBUG_INFERENCE][Turn]", "[DEBUG_INFERENCE][Sampler]",
    "[DEBUG_INFERENCE][Trajectory]", "stage_batch RPC returned",
)
LEVEL_NAMES = ("debug", "info", "warning", "error", "exception", "critical")
# Existing unit tests that reach twinned sites (run with events on).
DRIVEN = (
    "tests/experimental/weight_sync/weight_sync_coordinator_test.py",
    "tests/experimental/worker/remote_execution_test.py",
    "tests/experimental/orchestrator/orchestrator_test.py",
    "tests/experimental/orchestrator/distributed_rl_engine_test.py",
    "tests/experimental/orchestrator/distributed_rl_engine_staging_test.py",
    "tests/experimental/orchestrator/rl_program_test.py",
    "tests/experimental/worker/trainer_worker_test.py",
)


def _marker(fmt: str) -> str | None:
  for m in MARKERS:
    if fmt.startswith(m):
      return m
  return None


def _kw(call: ast.Call, name: str):
  for k in call.keywords:
    if k.arg == name:
      return k.value
  return None


def converted_calls(path: str) -> list[dict]:
  """Every `events.log_and_emit(...)` call in `path`, as golden-shaped dicts."""
  src = (REPO / path).read_text(encoding="utf-8")
  out = []
  for node in ast.walk(ast.parse(src)):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_and_emit"):
      continue
    logger, level, fmt, ev = node.args[:4]
    targs = _kw(node, "text_args")
    names = _kw(node, "names")
    out.append({
        "file": path,
        "logger": ast.unparse(logger),
        "level": level.attr if isinstance(level, ast.Attribute) else "?",
        "fmt": ast.literal_eval(fmt),
        "ev": ast.literal_eval(ev),
        "args": [ast.unparse(a) for a in (targs.elts if targs else [])],
        "names": [ast.literal_eval(n) for n in (names.elts if names else [])],
        "fields": sorted(k.arg for k in node.keywords
                         if k.arg not in ("text_args", "names")
                         and k.arg is not None),
        "extra": any(k.arg is None for k in node.keywords),
        "consts": {k.arg: ast.literal_eval(k.value) for k in node.keywords
                   if k.arg not in ("text_args", "names", None)},
        "lineno": node.lineno,
    })
  return out


def _key(site: dict) -> str:
  return json.dumps({k: site[k] for k in ("file", "logger", "level", "fmt",
                                          "ev", "args", "names", "fields",
                                          "extra", "consts")}, sort_keys=True)


class StaticTwinTest(unittest.TestCase):

  def test_1_10_1_calls_match_the_base_calls(self):
    files = sorted({s["file"] for s in SITES})
    current = [c for f in files for c in converted_calls(f)]
    want = collections.Counter(_key(s) for s in SITES)
    got = collections.Counter(_key(c) for c in current)
    self.assertEqual(sum(want.values()), 44)
    missing = want - got
    extra = got - want
    self.assertEqual((list(missing), list(extra)), ([], []))

  def test_1_10_2_fields_are_the_text_arguments_by_name(self):
    for s in SITES:
      with self.subTest(file=s["file"], fmt=s["fmt"][:50]):
        self.assertEqual(len(s["names"]), len(s["args"]))
        named = [n for n in s["names"] if n is not None]
        self.assertEqual(len(named), len(set(named)))
        self.assertFalse(set(named) & set(s["fields"]))
        schema = events.SCHEMAS[s["ev"]]
        for n in named + s["fields"]:
          self.assertIsNotNone(schema.kind(n), (s["ev"], n))
        # The text's conversion count equals the argument count.
        self.assertEqual(s["fmt"].replace("%%", "").count("%"),
                         len(s["args"]))

  def test_1_10_3_no_marker_site_left_unconverted(self):
    files = sorted({s["file"] for s in SITES})
    leftover = []
    per_marker = collections.Counter()
    for f in files:
      tree = ast.parse((REPO / f).read_text(encoding="utf-8"))
      for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args
                and isinstance(node.func, ast.Attribute)):
          continue
        first = node.args[0]
        if node.func.attr == "log_and_emit" and len(node.args) >= 3:
          first = node.args[2]
        if not isinstance(first, (ast.Constant, ast.JoinedStr, ast.BinOp)):
          continue
        try:
          text = ast.literal_eval(first)
        except ValueError:
          text = ast.unparse(first)
        if not isinstance(text, str) or _marker(text) is None:
          continue
        if node.func.attr == "log_and_emit":
          per_marker[_marker(text)] += 1
        elif node.func.attr in LEVEL_NAMES:
          leftover.append(f"{f}:{node.lineno}: {text[:60]}")
    self.assertEqual(leftover, [])
    golden = collections.Counter(_marker(s["fmt"]) for s in SITES)
    self.assertEqual(per_marker, golden)
    # Per-marker counts at this base. TRAINER_STEP_TIMING and
    # "stage_batch RPC returned" have no site here yet; they are converted
    # when the step timer and the staging queue land.
    self.assertEqual(golden["WEIGHT_SYNC_PHASE"], 14)
    self.assertEqual(golden["[rollout-ft]"], 20)
    self.assertEqual(golden["Packed "], 2)
    self.assertEqual(golden["TRAINER_STEP_TIMING"], 0)

  def test_1_1_4_lint_over_the_tree_with_call_sites(self):
    from tunix.experimental import events_test  # pylint: disable=g-import-not-at-top
    self.assertEqual(events_test.lint_tree(REPO / "tunix"), [])


# ---------------------------------------------------------------------------
# The driven run: plugin hooks (active only under EVENTS_TWIN_CAPTURE).

_CAPTURED: list = []


class _Tap(logging.Handler):

  def emit(self, record):
    st = events._state  # pylint: disable=protected-access
    _CAPTURED.append((record, len(st.q) if st is not None else -1))


def pytest_configure(config):
  del config
  if not os.environ.get("EVENTS_TWIN_CAPTURE"):
    return
  from absl import logging as absl_logging  # pylint: disable=g-import-not-at-top
  events.configure("twins", "proc", "pytest", None, "full",
                   local_dir=tempfile.mkdtemp(prefix="twins_"),
                   queue_size=10**7, start_thread=False)
  absl_logging.set_verbosity(absl_logging.DEBUG)
  root = logging.getLogger()
  root.addHandler(_Tap(level=logging.DEBUG))
  root.setLevel(logging.DEBUG)


def _site_index():
  idx = {}
  for s in SITES:
    idx.setdefault((os.path.basename(s["file"]), s["fmt"]), []).append(s)
  return idx


def _same(a, b) -> bool:
  if a is b:
    return True
  try:
    return bool(a == b)
  except Exception:  # pylint: disable=broad-except
    return False


def pytest_sessionfinish(session, exitstatus):
  del session
  out = os.environ.get("EVENTS_TWIN_CAPTURE")
  if not out:
    return
  st = events._state  # pylint: disable=protected-access
  q = list(st.q) if st is not None else []
  idx = _site_index()
  hits, problems = collections.Counter(), []
  for record, at in _CAPTURED:
    sites = idx.get((record.filename, record.msg))
    if not sites:
      continue
    s = sites[0]
    site = f"{s['file']}:{s['fmt'][:48]}"
    hits[site] += 1
    args = record.args if isinstance(record.args, tuple) else (record.args,)
    if record.funcName == "log_and_emit" or record.filename == "events.py":
      problems.append(f"{site}: the record names events.py")
    try:
      text = record.getMessage()
    except Exception as e:  # pylint: disable=broad-except
      problems.append(f"{site}: text does not format: {e!r}")
      continue
    if text != s["fmt"] % args:
      problems.append(f"{site}: text differs")
    # The event this call emitted: the first later item of that type whose
    # named fields are the record's arguments.
    match = None
    for item in q[max(at, 0):max(at, 0) + 200]:
      if item.get(" ev") != s["ev"]:
        continue
      if all(_same(item.get(n), a) for n, a in zip(s["names"], args)
             if n is not None):
        match = item
        break
    if match is None:
      problems.append(f"{site}: no event with the text's arguments")
      continue
    fields = {k: v for k, v in match.items() if not k.startswith(" ")}
    try:
      events._validate_strict(s["ev"], fields)  # pylint: disable=protected-access
    except events.SchemaError as e:
      problems.append(f"{site}: schema: {e}")
  with open(out, "w", encoding="utf-8") as f:
    json.dump({"exitstatus": int(exitstatus), "records": len(_CAPTURED),
               "events": len(q), "hits": dict(hits),
               "problems": sorted(set(problems))}, f, indent=1)


class CodePathTwinTest(unittest.TestCase):
  """Runs DRIVEN with events on; prints `EVENTS_TWINS {json}` for the README."""

  def test_1_10_code_paths(self):
    total_hits, all_problems, per_module = collections.Counter(), [], {}
    for mod in DRIVEN:
      if not (REPO / mod).exists():
        per_module[mod] = "absent"
        continue
      with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
      env = dict(os.environ, EVENTS_TWIN_CAPTURE=out,
                 PYTHONPATH=os.pathsep.join(
                     [str(REPO)] + os.environ.get("PYTHONPATH", "").split(
                         os.pathsep)))
      env.pop("TUNIX_EVENTS", None)
      proc = subprocess.run(
          [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "-p", "tunix.experimental.events_twins_test", "-x" if False else
           "-q", str(REPO / mod)], cwd=str(REPO), env=env,
          capture_output=True, text=True, timeout=1200)
      try:
        res = json.load(open(out, encoding="utf-8"))
      except (OSError, ValueError):
        self.fail(f"{mod}: no capture (rc={proc.returncode})\n"
                  f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
      per_module[mod] = {"rc": proc.returncode, "hits": sum(
          res["hits"].values()), "sites": len(res["hits"])}
      total_hits.update(res["hits"])
      all_problems += [f"{mod}: {p}" for p in res["problems"]]
    covered = set(total_hits)
    all_sites = {f"{s['file']}:{s['fmt'][:48]}" for s in SITES}
    summary = {"modules": per_module, "sites_total": len(all_sites),
               "sites_covered": len(covered & all_sites),
               "records_checked": sum(total_hits.values()),
               "uncovered": sorted(all_sites - covered),
               "problems": all_problems}
    print("EVENTS_TWINS " + json.dumps(summary), flush=True)
    self.assertEqual(all_problems, [])
    self.assertGreater(len(covered), 0)


if __name__ == "__main__":
  unittest.main()
