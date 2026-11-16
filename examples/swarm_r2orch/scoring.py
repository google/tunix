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

"""Multi-domain R2Orch verifiers and normalized Worker token-cost accounting."""

import ast
import builtins
import dataclasses
import decimal
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import uuid

from examples.swarm_math import drmas_reward

REFERENCE_TOKENS: dict[str, tuple[int, int]] = {
    "aime": (14000, 8400),
    "math_hard": (14000, 8400),
    "hotpotqa": (6800, 430),
    "musique": (6800, 430),
    "medqa": (4500, 920),
    "mbppplus": (1650, 290),
}
DEFAULT_REFERENCE_TOKENS = (5000, 1000)
COST_WEIGHT_IN = 0.25
COST_WEIGHT_OUT = 1.0

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*(.+)", re.IGNORECASE
)
_CHOICE_CUES = (
    re.compile(
        r"(?i:final\s+answer|answer)[*_\s]*(?i:is)?[*_\s]*[:：=]?\s*[*_(\[]*([A-J])(?![A-Za-z])"
    ),
    re.compile(
        r"(?i:option|choice)[*_\s]*(?i:is)?[*_\s]*[:：=]?\s*[*_(\[]*([A-J])(?![A-Za-z])"
    ),
)
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,})([^`]*)$")
_ASSERTION_CALL_RE = re.compile(r"assertion\(\s*([A-Za-z_]\w*)\(\*inp\)")
_DRIVER_CALLS = frozenset({
    "print",
    "input",
    "exit",
    "quit",
    "sys.exit",
    "os._exit",
    "unittest.main",
    "pytest.main",
})
_ONE_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


@dataclasses.dataclass(frozen=True)
class R2OrchTask:
  """Uniform benchmark task across the reasoning, QA, MCQ, and coding suites."""

  task_id: str
  dataset: str
  verifier: str
  prompt: str
  answers: tuple[str, ...] = ()
  options: tuple[str, ...] = ()
  code_spec: dict[str, object] = dataclasses.field(default_factory=dict)
  curriculum: dict[str, object] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    object.__setattr__(self, "answers", tuple(str(x) for x in self.answers))
    object.__setattr__(self, "options", tuple(str(x) for x in self.options))


def compute_worker_cost(token_in: int, token_out: int, dataset: str) -> float:
  """Normalized R2Orch Worker token cost `(0.25 * in + 1.0 * out) / ref_cost`."""
  ref_in, ref_out = REFERENCE_TOKENS.get(dataset, DEFAULT_REFERENCE_TOKENS)
  ref_cost = ref_in * COST_WEIGHT_IN + ref_out * COST_WEIGHT_OUT
  return (token_in * COST_WEIGHT_IN + token_out * COST_WEIGHT_OUT) / ref_cost


def _parse_decimal(raw: str) -> decimal.Decimal | None:
  cleaned = raw.strip().strip("$*`").rstrip(".。")
  if not _NUMBER_RE.fullmatch(cleaned):
    return None
  try:
    return decimal.Decimal(cleaned.replace(",", ""))
  except decimal.InvalidOperation:
    return None


def extract_math_scalar(output: str) -> decimal.Decimal | None:
  """Extracts an explicit boxed, `####`, or `Answer:` scalar without guessing."""
  text = output.strip()
  if not text:
    return None
  direct = _parse_decimal(text)
  if direct is not None:
    return direct
  prose = re.sub(r"```.*?(?:```|\Z)", "", text, flags=re.DOTALL)
  boxed = _BOXED_RE.findall(prose)
  if boxed:
    return _parse_decimal(boxed[-1])
  hashes = re.findall(r"(?m)^\s*####\s*([^\n]+)", prose)
  if hashes:
    return _parse_decimal(hashes[-1])
  cues = list(_ANSWER_CUE_RE.finditer(prose))
  if not cues:
    return None
  tail = cues[-1].group(1).splitlines()[0]
  match = _NUMBER_RE.search(tail)
  return _parse_decimal(match.group(0)) if match else None


def _normalize_qa(text: str) -> str:
  lowered = re.sub(r"[^\w\s]", "", text.lower())
  without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
  return " ".join(without_articles.split())


def extract_short_answer(output: str) -> str:
  """Extracts the concise answer span from `Answer:` cue or final line."""
  lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
  if not lines:
    return ""
  cues = list(_ANSWER_CUE_RE.finditer(output))
  span = cues[-1].group(1).splitlines()[0] if cues else lines[-1]
  span = re.sub(r"[*_`]+", "", span).strip().strip("\"'").rstrip(".").strip()
  if span or not cues:
    return span
  tail_lines = [
      re.sub(r"[*_`]+", "", l).strip().strip("\"'").rstrip(".").strip()
      for l in output[cues[-1].end() :].splitlines()
  ]
  return next((l for l in tail_lines if l), "")


def score_qa_exact_match(output: str, answers: tuple[str, ...]) -> float:
  """SQuAD/HotpotQA normalized exact match with yes/no first-word handling."""
  pred = _normalize_qa(extract_short_answer(output))
  if not pred:
    return 0.0
  for target in answers:
    gold = _normalize_qa(target)
    if gold in ("yes", "no"):
      if pred.split()[0] == gold:
        return 1.0
    elif pred == gold:
      return 1.0
  return 0.0


def score_letter_choice(
    output: str, answer: str, options: tuple[str, ...]
) -> float:
  """Extracts a capital option letter in `options` and compares with `answer`."""
  allowed = set(options) if options else {"A", "B", "C", "D"}
  for pattern in _CHOICE_CUES:
    matches = [m for m in pattern.findall(output) if m in allowed]
    if matches:
      return float(matches[-1] == answer)
  bare = re.fullmatch(r"\s*[*_(\[]*\s*([A-J])\s*[)\].:*_]*\s*", output)
  if bare and bare.group(1) in allowed:
    return float(bare.group(1) == answer)
  return 0.0


def _infer_entry_point(code_spec: dict[str, object]) -> str | None:
  plus_test = str(code_spec.get("plus_test") or "")
  found = _ASSERTION_CALL_RE.findall(plus_test)
  if found:
    return found[-1]
  raw_tests = code_spec.get("test_list") or []
  if isinstance(raw_tests, str):
    try:
      raw_tests = ast.literal_eval(raw_tests)
    except (SyntaxError, ValueError):
      raw_tests = []
  if not raw_tests:
    return None
  try:
    tree = ast.parse(str(raw_tests[0]))
  except SyntaxError:
    return None
  builtin_names = frozenset(dir(builtins))
  for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id not in builtin_names
    ):
      return node.func.id
  return None


def _extract_fenced_blocks(output: str) -> list[str]:
  blocks: list[str] = []
  current_ticks = 0
  buf: list[str] = []
  for line in output.splitlines():
    match = _FENCE_LINE_RE.match(line)
    if match and current_ticks == 0:
      current_ticks = len(match.group(1))
      buf = []
    elif match and len(match.group(1)) >= current_ticks and not match.group(2).strip():
      source = textwrap.dedent("\n".join(buf)).strip()
      if source:
        blocks.append(source)
      current_ticks = 0
    elif current_ticks > 0:
      buf.append(line)
  if current_ticks > 0 and buf:
    blocks.append(textwrap.dedent("\n".join(buf)).strip())
  return blocks


def _is_driver_stmt(stmt: ast.stmt, defined: set[str]) -> bool:
  if isinstance(stmt, ast.Assert):
    return True
  if isinstance(stmt, ast.If) and ast.unparse(stmt.test).replace('"', "'") in (
      "__name__ == '__main__'",
      "'__main__' == __name__",
  ):
    return True
  if not isinstance(stmt, ast.Expr):
    return False
  for child in ast.walk(stmt):
    if isinstance(child, ast.Call):
      name = ast.unparse(child.func)
      if name in _DRIVER_CALLS or name in defined:
        return True
  return False


def extract_python_candidate(output: str, entry_point: str | None) -> str:
  """Selects the last parseable code block defining `entry_point` and strips drivers."""
  candidates = _extract_fenced_blocks(output) or [output.strip()]
  chosen = candidates[-1]
  for block in reversed(candidates):
    try:
      tree = ast.parse(block)
    except SyntaxError:
      continue
    names = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if entry_point is None or entry_point in names:
      chosen = block
      break
  try:
    tree = ast.parse(chosen)
  except SyntaxError:
    return chosen
  defined = {
      n.name
      for n in tree.body
      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
  }
  kept = [stmt for stmt in tree.body if not _is_driver_stmt(stmt, defined)]
  return "\n".join(ast.unparse(stmt) for stmt in kept) if kept else chosen


def score_mbppplus(
    output: str,
    code_spec: dict[str, object],
    *,
    cpu_seconds: int = 20,
    python_bin: str = sys.executable,
) -> float:
  """Runs extracted candidate against `plus_test` in a resource-limited child."""
  plus_test = str(code_spec.get("plus_test") or "")
  if not plus_test:
    return 0.0
  source = extract_python_candidate(output, _infer_entry_point(code_spec))
  if not source:
    return 0.0
  try:
    ast.parse(source)
  except SyntaxError:
    return 0.0
  raw_imports = code_spec.get("test_imports") or []
  imports = "\n".join(str(x) for x in raw_imports)
  sentinel = f"mbppplus-ok-{uuid.uuid4().hex}"
  rlimit_header = (
      "import resource as _r\n"
      f"_r.setrlimit(_r.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
      "_r.setrlimit(_r.RLIMIT_AS, (4 << 30, 4 << 30))\n"
      "_r.setrlimit(_r.RLIMIT_FSIZE, (8 << 20, 8 << 20))\n"
      if sys.platform.startswith("linux")
      else ""
  )
  program = (
      f"{rlimit_header}{source}\n\n{imports}\n{plus_test}\n"
      f"print({sentinel!r})\n"
  )
  with tempfile.TemporaryDirectory(prefix="r2orch-mbpp-") as tmp_dir:
    script = Path(tmp_dir) / "check.py"
    script.write_text(program, encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        **_ONE_THREAD_ENV,
    }
    try:
      proc = subprocess.run(
          [python_bin, "-I", str(script)],
          cwd=tmp_dir,
          env=env,
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE,
          stderr=subprocess.DEVNULL,
          timeout=cpu_seconds + 5,
          check=False,
      )
    except (OSError, subprocess.SubprocessError):
      return 0.0
  lines = proc.stdout.decode("utf-8", "replace").splitlines()
  return float(proc.returncode == 0 and lines[-1:] == [sentinel])


def score_math_answer(
    output: str, answer: str, *, allow_symbolic: bool
) -> float:
  """Grades exact scalar or symbolic mathematical equivalence."""
  if allow_symbolic:
    _, ok = drmas_reward.RewardMathFn(drmas_reward.RewardConfig())(
        {"ground_truth": answer}, output
    )
    if ok:
      return 1.0
  pred = extract_math_scalar(output)
  gold = _parse_decimal(answer)
  return float(pred is not None and gold is not None and pred == gold)


def score_task(task: R2OrchTask, output: str) -> float:
  """Grades `output` in {0.0, 1.0} using `task.verifier`."""
  if not output.strip():
    return 0.0
  if task.verifier in ("integer_answer", "math_symbolic"):
    if not task.answers:
      return 0.0
    return score_math_answer(
        output, task.answers[0], allow_symbolic=task.verifier == "math_symbolic"
    )
  if task.verifier == "qa_exact_match":
    return score_qa_exact_match(output, task.answers)
  if task.verifier == "letter_choice":
    if not task.answers:
      return 0.0
    return score_letter_choice(output, task.answers[0], task.options)
  if task.verifier == "mbppplus_tests":
    return score_mbppplus(output, task.code_spec)
  raise ValueError(f"Unsupported verifier: {task.verifier!r}")
