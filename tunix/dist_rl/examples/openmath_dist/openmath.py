# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenMathInstruct-2 components for distributed GRPO training.

Matches the AI-Hypercomputer/tpu-recipes Qwen3-0.6B OpenMathInstruct-2 GRPO
recipe (`maxtext.trainers.post_train.rl.train_rl` + `utils_rl.py`):
* Prompt formatting: `gsm8k_rl.json` (`<reasoning>`/`<answer>`) rendered
  through the model tokenizer's `apply_chat_template(...,
  add_generation_prompt=True)`.
* Length filter: prompts exceeding `max_prompt_length` tokens are dropped after
  deterministic Grain shuffling (`seed=42`).
* Reward stack: exact format match (`+0.1`) plus exact/LaTeX symbolic
  correctness (`+1.0`) isolated in persistent worker subprocesses with per-call
  timeout protection against hung SymPy expressions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import importlib.util
import json
from multiprocessing import connection
from multiprocessing import process as mp_process
import os
import queue
import re
import threading
from typing import TypedDict

from absl import logging
import datasets
import grain
import transformers
from tunix.dist_rl.common import registry
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.utils import math_utils

import multiprocessing

OPENMATH_ENV_NAME = "openmathenv"
OPENMATH_AGENT_NAME = "openmathagent"

DEFAULT_DATA_FILES = (
    "hf://datasets/nvidia/OpenMathInstruct-2"
    "@469216e3f46f4dacf476b382e192485ea51a143e/data/train_1M-*.parquet"
)

REASONING_START_TOKEN = "<reasoning>"
REASONING_END_TOKEN = "</reasoning>"
SOLUTION_START_TOKEN = "<answer>"
SOLUTION_END_TOKEN = "</answer>"

REWARD_EXACT_FORMAT_MATCH = 0.1
REWARD_EXACT_ANSWER = 1.0
FALLBACK_ANSWER = "-1000000"
_WORKER_STARTUP_TIMEOUT_S = 60.0

SYSTEM_PROMPT = (
    "You are given a problem. Think about the problem and provide your "
    f"reasoning. Place it between {REASONING_START_TOKEN} and "
    f"{REASONING_END_TOKEN}. Then, provide the final answer (i.e., just one "
    f"numerical value) between {SOLUTION_START_TOKEN} and "
    f"{SOLUTION_END_TOKEN}."
)
USER_TEMPLATE = (
    "<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n"
    "<start_of_turn>model"
)

# Atomic skip groups `(?>.*?X)` match MaxText `utils_rl.get_match_format_regex`
# so long completions repeating unclosed tag literals cannot catastrophically
# backtrack.
_FORMAT_REGEX = re.compile(
    rf"{re.escape(REASONING_START_TOKEN)}"
    rf"(?>.*?{re.escape(REASONING_END_TOKEN)})"
    rf"(?>.*?{re.escape(SOLUTION_START_TOKEN)})(.+?)"
    rf"{re.escape(SOLUTION_END_TOKEN)}",
    flags=re.MULTILINE | re.DOTALL,
)
_ANSWER_FALLBACK_REGEX = re.compile(
    rf"{re.escape(SOLUTION_START_TOKEN)}(.+?){re.escape(SOLUTION_END_TOKEN)}",
    flags=re.MULTILINE | re.DOTALL,
)
_BOXED_PREFIX = "\\boxed"
_BRACE_REGEX = re.compile(r"[{}]")
_PERMISSIVE_BOXED_REGEX = re.compile(
    r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?"
)
_LATEX_FRAC_REGEX = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_N_REGEX = re.compile(r"\\sqrt\[([^\[\]{}]+)\]\{([^{}]+)\}")
_LATEX_SQRT_REGEX = re.compile(r"\\sqrt\{([^{}]+)\}")
_LATEX_POW_REGEX = re.compile(r"\^\{([^{}]+)\}")
_MCQ_OPTION_REGEX = re.compile(
    r"""
    (?:
        \(?([A-E])\)?
        |
        \\(?:text|mathrm|textbf)\{([A-E])\}
    )
    [\s\.\:\}\$]*\s*
    (.*?)
    (?=
        \s*(?:\\quad|\\qquad|\\hspace|\n|\r)\s*(?:\(?|\\(?:text|mathrm|textbf))
        | $
    )
    """,
    flags=re.DOTALL | re.VERBOSE,
)

_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("\\\\", "\\"),
    ("\\tfrac", "\\frac"),
    ("\\dfrac", "\\frac"),
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" or ", ","),
    (" and ", ","),
    ("million", "*10^6"),
    ("billion", "*10^9"),
    ("trillion", "*10^12"),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
)
_UNITS: tuple[str, ...] = (
    "yard",
    "foot",
    "feet",
    "mile",
    "day",
    "week",
    "month",
    "year",
    "hour",
    "minute",
    "second",
    "centimeter",
    "meter",
    "cm",
    "mm",
    "km",
    "inch",
    "degree",
    "pound",
    "cent",
    "mph",
)
_REMOVED_EXPRESSIONS: tuple[str, ...] = (
    "\\left",
    "\\right",
    "\\!",
    "square",
    "ways",
    "integers",
    "dollars",
    "units",
    "\\ldots",
    "sue",
    "points",
    "digits",
    "gm",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
)
_ESCAPE_FIXES: tuple[tuple[str, str, str], ...] = (
    ("\f", "rac", r"\frac"),
    ("\n", "ewline", r"\newline"),
    ("\n", "e", r"\ne"),
    ("\t", "heta", r"\theta"),
    ("\t", "an", r"\tan"),
    ("\t", "o", r"\to"),
    ("\t", "imes", r"\times"),
    ("\t", "ext", r"\text"),
    ("\t", "extbf", r"\textbf"),
    ("\t", "extit", r"\textit"),
    ("\r", "ightarrow", r"\rightarrow"),
    ("\b", "eta", r"\beta"),
    ("\b", "ar", r"\bar"),
    ("\b", "inom", r"\binom"),
    ("\b", "oxed", r"\boxed"),
    ("\a", "lpha", r"\alpha"),
    ("\a", "pprox", r"\approx"),
    ("\v", "ec", r"\vec"),
)
_LATEX_COMMANDS: tuple[str, ...] = (
    "frac",
    "sqrt",
    "pi",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "sum",
    "int",
    "infty",
    "cdot",
    "times",
    "div",
    "pm",
    "mp",
    "leq",
    "geq",
    "neq",
    "approx",
    "equiv",
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "exp",
    "lim",
    "to",
    "rightarrow",
    "leftarrow",
    "Rightarrow",
    "Leftarrow",
    "overline",
    "underline",
    "hat",
    "bar",
    "vec",
    "dot",
    "ddot",
    "mathbb",
    "mathbf",
    "mathrm",
    "text",
    "textbf",
    "textit",
    "boxed",
    "left",
    "right",
    "choose",
    "binom",
)


class OpenMathEnvConfig(TypedDict):
  """Environment configuration attached to each OpenMath task record."""

  prompt: str
  question: str
  answer: str


class OpenMathMetadata(TypedDict):
  """Task metadata carried on each OpenMath orchestrator record."""

  question: str
  answer: str
  env_config: OpenMathEnvConfig


class OpenMathTaskRecord(TypedDict):
  """Task record returned by `create_dataset`."""

  prompt: str
  metadata: OpenMathMetadata


class ScoreInfo(TypedDict):
  """Detailed reward breakdown returned by `score_openmath_completion`."""

  format_correct: bool
  answer_correct: bool
  extracted_answer: str
  gold_answer: str


def boxed(expr: str) -> str:
  r"""Wraps `expr` in `\boxed{...}` when not already boxed."""
  return expr if expr.startswith("\\boxed{") else f"\\boxed{{{expr}}}"


def _unbox_for_sympy(expr: str) -> str:
  """Converts a boxed LaTeX expression into SymPy-readable infix notation."""
  inner = _last_balanced_boxed(expr)
  text = inner if inner is not None else expr
  text = (
      text.replace("\\cdot", "*")
      .replace("\\times", "*")
      .replace("\\div", "/")
      .replace("\\left", "")
      .replace("\\right", "")
  )
  while True:
    updated = _LATEX_SQRT_N_REGEX.sub(r"((\2)**(1/(\1)))", text)
    updated = _LATEX_SQRT_REGEX.sub(r"sqrt(\1)", updated)
    updated = _LATEX_FRAC_REGEX.sub(r"((\1)/(\2))", updated)
    updated = _LATEX_POW_REGEX.sub(r"**(\1)", updated)
    if updated == text:
      break
    text = updated
  return text.replace("{", "(").replace("}", ")").replace("\\", "").strip()


def normalize_final_answer(final_answer: str) -> str:
  """Normalizes a math answer string (mirrors MaxText `utils_rl.py`)."""
  ans = final_answer.split("=")[-1]
  ans = re.sub(r"([0-9]) +([0-9])", r"\1+\2", ans)
  for before, after in _SUBSTITUTIONS:
    ans = ans.replace(before, after)
  for unit in _UNITS:
    ans = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", ans)
  for expr in _REMOVED_EXPRESSIONS:
    ans = ans.replace(expr, "")
  ans = re.sub(
      r".*?(\d+)?\s*\$\s*(\d+)?\s*(\\frac\{.*?\}\{.*?\}|\d+/\d+)\s*\$.*",
      lambda m: f"${w}{m.group(3)}$"
      if (w := (m.group(1) or m.group(2)))
      else f"${m.group(3)}$",
      ans,
  )
  for wrapper in ("text", "textbf", "overline"):
    ans = re.sub(rf"(\\{wrapper}\{{)(.*?)(\}})", r"\2", ans)
  ans = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", ans)
  ans = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", ans)
  ans = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", ans)
  ans = ans.replace("$", "")
  if ans.startswith("."):
    ans = "0" + ans
  ans = ans.replace("{.", "{0.")
  if len(ans) >= 2 and ans[0] == "{" and ans[-1] == "}":
    ans = ans[1:-1]
  try:
    val = float(ans)
    if abs(val - round(val)) < 1e-7:
      ans = str(int(round(val)))
  except (ValueError, OverflowError):
    pass
  if ans.replace(",", "").isdigit():
    ans = ans.replace(",", "")
  return ans.strip()


def preprocess_math_string(text: str) -> str:
  """Normalizes math strings and repairs unescaped LaTeX command names."""
  out = normalize_final_answer(text)
  for escape_char, suffix, latex_cmd in _ESCAPE_FIXES:
    if escape_char in out:
      out = out.replace(escape_char + suffix, latex_cmd)
  for cmd in _LATEX_COMMANDS:
    out = re.sub(rf"(?<!\\)\b{cmd}\b", rf"\\{cmd}", out)
  return out


def _last_balanced_boxed(content: str) -> str | None:
  r"""Extracts the content of the last balanced `\boxed{...}` in linear time."""
  best: str | None = None
  stack: list[int] = []
  for match in _BRACE_REGEX.finditer(content):
    idx = match.start()
    if content[idx] == "{":
      stack.append(idx)
    elif stack:
      open_idx = stack.pop()
      prefix_start = open_idx - len(_BOXED_PREFIX)
      if prefix_start >= 0 and content.startswith(_BOXED_PREFIX, prefix_start):
        best = content[open_idx + 1 : idx].strip()
  return best


def extract_answer(response: str) -> str:
  r"""Extracts the candidate answer from `<answer>...</answer>` or `\boxed{}`."""
  answer_matches = _ANSWER_FALLBACK_REGEX.findall(response)
  scope = answer_matches[-1] if answer_matches else response
  boxed_ans = _last_balanced_boxed(scope)
  if boxed_ans is not None:
    return boxed_ans
  permissive = _PERMISSIVE_BOXED_REGEX.search(scope)
  if permissive is not None:
    return permissive.group(1).strip()
  if answer_matches:
    return answer_matches[-1].strip()
  return FALLBACK_ANSWER


def _grade_symbolic_pair(norm_golds: Sequence[str], norm_guess: str) -> bool:
  """Grades symbolic math equivalence inside the isolated worker process."""
  if importlib.util.find_spec("math_verify") is not None:
    mv = importlib.import_module("math_verify")
    mv_parser = importlib.import_module("math_verify.parser")
    configs = (
        mv_parser.ExprExtractionConfig(),
        mv_parser.LatexExtractionConfig(),
    )
    parsed_guess = mv.parse(norm_guess, extraction_config=configs)
    if not parsed_guess:
      return False
    for gold in norm_golds:
      parsed_gold = mv.parse(gold, extraction_config=configs)
      if parsed_gold and mv.verify(parsed_gold, parsed_guess):
        return True
    return False

  guess_sym = _unbox_for_sympy(norm_guess)
  return any(
      math_utils.are_equal_under_sympy(_unbox_for_sympy(gold), guess_sym)
      for gold in norm_golds
  )


def _math_verify_worker_loop(conn: connection.Connection) -> None:
  """Persistent subprocess loop isolating `math_verify` / `sympy`."""
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["TPU_VISIBLE_DEVICES"] = ""
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  try:
    _ = _grade_symbolic_pair(["\\boxed{1}"], "\\boxed{1}")
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  conn.send(True)

  while True:
    try:
      msg = conn.recv()
    except EOFError:
      return
    if msg is None:
      return
    norm_golds, norm_guess = msg
    try:
      matched = _grade_symbolic_pair(norm_golds, norm_guess)
    except Exception:  # pylint: disable=broad-exception-caught
      matched = False
    conn.send(matched)


def _terminate_worker(
    proc: mp_process.BaseProcess, conn: connection.Connection
) -> None:
  proc.kill()
  proc.join(timeout=1.0)
  conn.close()


def _wait_for_worker_reply(
    conn: connection.Connection, timeout_s: float
) -> bool:
  """Waits up to `timeout_s` seconds for a worker result on `conn`."""
  return conn.poll(timeout_s)


class _MathVerifyRunner:
  """Pool of persistent workers that kills/replaces hung SymPy calls."""

  def __init__(self, max_workers: int = 4) -> None:
    ctx = multiprocessing.get_context("forkserver")
    self._ctx = ctx
    self._slots: queue.Queue[
        tuple[mp_process.BaseProcess, connection.Connection] | None
    ] = queue.Queue()
    for _ in range(max_workers):
      self._slots.put(None)

  def _spawn_worker(
      self,
  ) -> tuple[mp_process.BaseProcess, connection.Connection]:
    """Starts a fresh verifier worker and waits until its imports are warm."""
    parent_conn, child_conn = self._ctx.Pipe(duplex=True)
    proc = self._ctx.Process(
        target=_math_verify_worker_loop, args=(child_conn,), daemon=True
    )
    proc.start()
    child_conn.close()
    if not parent_conn.poll(_WORKER_STARTUP_TIMEOUT_S):
      _terminate_worker(proc, parent_conn)
      raise TimeoutError("Timed out starting math verifier worker subprocess.")
    _ = parent_conn.recv()
    return proc, parent_conn

  def verify_symbolic(
      self,
      norm_golds: Sequence[str],
      norm_guess: str,
      timeout_s: float = 10.0,
  ) -> bool:
    """Checks whether `norm_guess` matches any `norm_golds` within `timeout_s`."""
    slot = self._slots.get()
    replacement: tuple[mp_process.BaseProcess, connection.Connection] | None = (
        None
    )
    try:
      if slot is None or not slot[0].is_alive():
        if slot is not None:
          slot[1].close()
        slot = self._spawn_worker()
      proc, conn = slot
      try:
        conn.send((list(norm_golds), norm_guess))
        if _wait_for_worker_reply(conn, timeout_s):
          result = bool(conn.recv())
          replacement = (proc, conn)
          return result
      except (EOFError, OSError):
        pass
      _terminate_worker(proc, conn)
      return False
    finally:
      self._slots.put(replacement)


_VERIFIER_LOCK = threading.Lock()
_VERIFIER: _MathVerifyRunner | None = None


def _get_verifier() -> _MathVerifyRunner:
  global _VERIFIER
  with _VERIFIER_LOCK:
    if _VERIFIER is None:
      _VERIFIER = _MathVerifyRunner(max_workers=4)
    return _VERIFIER


def decode_acceptable_answers(raw_answer: str | Sequence[str]) -> Sequence[str]:
  """Decodes serialized JSON acceptable-answer lists or plain answer strings."""
  if isinstance(raw_answer, str):
    stripped = raw_answer.strip()
    if stripped.startswith("["):
      try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
          return list(dict.fromkeys(str(item) for item in parsed))
      except json.JSONDecodeError:
        pass
    return [raw_answer]
  return list(dict.fromkeys(str(item) for item in raw_answer))


def _is_answer_correct(
    guess: str, acceptable: Sequence[str], verify_timeout_s: float
) -> bool:
  """Checks exact, normalized, or symbolic answer equivalence."""
  if guess == FALLBACK_ANSWER or not acceptable:
    return False
  stripped_guess = guess.strip()
  if any(
      guess == gold or stripped_guess == gold.strip() for gold in acceptable
  ):
    return True
  norm_guess = boxed(preprocess_math_string(guess))
  norm_golds = [boxed(preprocess_math_string(gold)) for gold in acceptable]
  if norm_guess in norm_golds:
    return True
  if verify_timeout_s <= 0.0:
    return False
  return _get_verifier().verify_symbolic(
      norm_golds, norm_guess, timeout_s=verify_timeout_s
  )


def score_openmath_completion(
    completion: str,
    gold_answer: str | Sequence[str],
    *,
    verify_timeout_s: float = 10.0,
) -> tuple[float, ScoreInfo]:
  """Scores a completion with MaxText's exact format (`+0.1`) + math (`+1.0`)."""
  format_ok = _FORMAT_REGEX.search(completion) is not None
  format_score = REWARD_EXACT_FORMAT_MATCH if format_ok else 0.0

  acceptable = decode_acceptable_answers(gold_answer)
  guess = extract_answer(completion)
  answer_ok = _is_answer_correct(guess, acceptable, verify_timeout_s)

  answer_score = REWARD_EXACT_ANSWER if answer_ok else 0.0
  primary_gold = acceptable[0] if acceptable else ""
  return format_score + answer_score, {
      "format_correct": format_ok,
      "answer_correct": answer_ok,
      "extracted_answer": guess,
      "gold_answer": primary_gold,
  }


def process_mcq(question: str, answer: str) -> Sequence[str]:
  """Extracts both letter and value acceptable answers for MCQ items."""
  options: dict[str, str] = {}
  for match in _MCQ_OPTION_REGEX.findall(question):
    letter = match[0] or match[1]
    clean_val = re.sub(
        r"\\q?quad|\\textbf\{|[~$]", "", match[2].strip()
    ).strip()
    options[letter] = clean_val

  acceptable = [answer]
  if answer in options:
    acceptable.append(options[answer])
    return acceptable
  norm_ans = normalize_final_answer(answer)
  for letter, value in options.items():
    if normalize_final_answer(value) == norm_ans:
      acceptable.append(letter)
      break
  return acceptable


def build_prompt(
    question: str, tokenizer: transformers.PreTrainedTokenizerBase
) -> str:
  """Formats `question` using `gsm8k_rl.json` + tokenizer chat template."""
  user_content = USER_TEMPLATE.format(
      system_prompt=SYSTEM_PROMPT, question=question
  )
  return str(
      tokenizer.apply_chat_template(
          [{"role": "user", "content": user_content}],
          tokenize=False,
          add_generation_prompt=True,
      )
  )


def _as_str(value: object) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return str(value)


def _extract_question_and_answer(row: Mapping[str, object]) -> tuple[str, str]:
  """Extracts the raw problem statement and expected answer from `row`."""
  question: str | None = None
  for key in ("problem", "prompt", "question"):
    if key in row and row[key] is not None:
      question = _as_str(row[key])
      break
  if question is None:
    raise KeyError(
        f"OpenMath example is missing problem/prompt/question keys: {list(row)}"
    )

  answer: str | None = None
  for key in ("expected_answer", "answer", "solution"):
    if key in row and row[key] is not None:
      answer = _as_str(row[key])
      break
  if answer is None:
    raise KeyError(
        "OpenMath example is missing expected_answer/answer/solution:"
        f" {list(row)}"
    )
  return question, answer


def create_dataset(
    *,
    data_files: str = DEFAULT_DATA_FILES,
    split: str = "train_1M",
    shuffle: bool = True,
    seed: int = 42,
    max_prompt_length: int = 4096,
    tokenizer_path: str = "Qwen/Qwen3-0.6B",
    max_examples: int = 8192,
) -> grain.MapDataset:
  """Loads, shuffles, formats, and length-filters OpenMathInstruct-2 records."""
  logging.info(
      "Loading OpenMathInstruct-2 parquet data_files=%s split=%s shuffle=%s"
      " seed=%d max_prompt_length=%d max_examples=%d.",
      data_files,
      split,
      shuffle,
      seed,
      max_prompt_length,
      max_examples,
  )
  hf_ds = datasets.load_dataset(
      "parquet", data_files={split: data_files}, split=split
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  total = len(hf_ds)
  index_ds: grain.MapDataset = grain.MapDataset.range(total)
  if shuffle:
    index_ds = index_ds.shuffle(seed=seed)

  records: list[OpenMathTaskRecord] = []
  scanned = 0
  for pos in range(total):
    scanned = pos + 1
    row_idx = index_ds[pos]
    assert row_idx is not None
    row = hf_ds[int(row_idx)]
    if not isinstance(row, Mapping):
      raise TypeError(f"Expected mapping row at index {pos}, got {type(row)}.")
    question, raw_answer = _extract_question_and_answer(row)
    prompt = build_prompt(question, tokenizer)
    if len(tokenizer.tokenize(prompt)) > max_prompt_length:
      continue
    question_type = (
        _as_str(row["question_type"])
        if "question_type" in row and row["question_type"] is not None
        else "default"
    )
    acceptable = (
        list(process_mcq(question, raw_answer))
        if question_type == "MCQ"
        else [raw_answer]
    )
    serialized_answer = json.dumps(acceptable)
    records.append({
        "prompt": prompt,
        "metadata": {
            "question": question,
            "answer": serialized_answer,
            "env_config": {
                "prompt": prompt,
                "question": question,
                "answer": serialized_answer,
            },
        },
    })
    if max_examples > 0 and len(records) >= max_examples:
      break

  if not records:
    raise ValueError(
        "No OpenMath examples remained within"
        f" max_prompt_length={max_prompt_length}."
    )
  logging.info(
      "Prepared %d OpenMath task records (scanned %d/%d rows).",
      len(records),
      scanned,
      total,
  )
  return grain.MapDataset.source(records)


@registry.register_env(OPENMATH_ENV_NAME)
class OpenMathEnv(base_environment.BaseTaskEnv):
  """Single-turn OpenMathInstruct-2 environment."""

  def __init__(
      self,
      prompt: str = "",
      question: str = "",
      answer: str = "",
      prompt_id: str = "",
      group_index: int = 0,
      policy_version: int = 0,
      max_steps: int = 1,
      verify_timeout_s: float = 10.0,
      reward_fn: Callable[..., object] | None = None,
      final_reward_fn: Callable[[], float] | None = None,
      **extra_kwargs: object,
  ) -> None:
    """Initializes the OpenMath single-turn environment."""
    self._verify_timeout_s = float(verify_timeout_s)
    super().__init__(
        task={
            "prompts": prompt,
            "question": question,
            "answer": answer,
            "policy_version": policy_version,
        },
        reward_fn=reward_fn,
        final_reward_fn=final_reward_fn,
        max_steps=max_steps,
        prompt_id=prompt_id,
        group_index=group_index,
        **extra_kwargs,
    )

  def _initial_observation(self) -> dict[str, str]:
    return {"prompts": str(self.task["prompts"])}

  def _step_impl(
      self, action: agent_types.Action | str
  ) -> base_environment.EnvStepResult:
    completion = (
        action.action if isinstance(action, agent_types.Action) else str(action)
    )
    gold = str(self.task["answer"])
    reward, info = score_openmath_completion(
        str(completion), gold, verify_timeout_s=self._verify_timeout_s
    )
    return base_environment.EnvStepResult(
        observation={
            "answer": str(completion),
            "gold_answer": info["gold_answer"],
        },
        reward=reward,
        done=True,
        info={
            **info,
            "correct": info["answer_correct"],
        },
    )


@registry.register_agent(OPENMATH_AGENT_NAME)
class OpenMathAgent(base_agent.ConversationAgentBase):
  """Forwards generated model text as the OpenMath environment action."""

  name = OPENMATH_AGENT_NAME

  def __init__(self) -> None:
    """Initializes the stateless single-turn OpenMath agent."""
    super().__init__("")

  def update_from_model(
      self, response: str, **kwargs: object
  ) -> agent_types.Action:
    """Records `response` on the trajectory and returns it as an `Action`."""
    del kwargs
    action = agent_types.Action(action=response)
    self.trajectory.steps.append(
        agent_types.Step(
            model_response=response,
            thought="",
            action=action,
        )
    )
    self.chat_completions.append({"role": "assistant", "content": response})
    return action
