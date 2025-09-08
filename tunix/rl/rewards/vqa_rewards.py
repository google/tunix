"""Lightweight reward functions for VLM (VQA / caption-like) GRPO."""

from __future__ import annotations
from typing import Iterable, List, Sequence
import math
import re

# Optional: if your dataset provides multiple acceptable answers per item,
# pass a list for 'answer' instead of a string and enable 'ANSWERS_AS_LIST=True'.
ANSWERS_AS_LIST = True  # set False if you always pass a single string

# ---------
# Utilities
# ---------

_WS = re.compile(r"\s+")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _normalize_text(s: str) -> str:
  """Lowercase, strip, squeeze spaces, strip punctuation at edges."""
  s = s.strip().lower()
  s = _WS.sub(" ", s)
  # lightweight punctuation trim at edges (keeps math signs inside tokens)
  return s.strip(" \t\n\r.,;:!?\"'`")

def _extract_first_number(s: str) -> float | None:
  m = _NUM.search(s)
  if not m:
    return None
  try:
    return float(m.group(0))
  except Exception:
    return None

def _any_equiv(cand: str, answers: Sequence[str]) -> bool:
  cn = _normalize_text(cand)
  return any(cn == _normalize_text(a) for a in answers)

def _closest_num_error(cand: str, answers: Sequence[str]) -> float | None:
  """Returns min absolute error to any numeric ground-truth; None if none numeric."""
  cnum = _extract_first_number(cand)
  if cnum is None:
    return None
  errs = []
  for a in answers:
    anum = _extract_first_number(a)
    if anum is not None:
      errs.append(abs(cnum - anum))
  return min(errs) if errs else None


# ---------------
# Reward helpers
# ---------------

def match_format_exact(prompts: List[str], completions: List[str], **kargs) -> List[float]:
  """+1 if completion uses <reasoning>...</reasoning> and <answer>...</answer> blocks."""
  reasoning = re.compile(r"<\s*reasoning\s*>.+?<\s*/\s*reasoning\s*>", re.S | re.I)
  answer = re.compile(r"<\s*answer\s*>.+?<\s*/\s*answer\s*>", re.S | re.I)
  out = []
  for c in completions:
    ok = 1.0 if (reasoning.search(c) and answer.search(c)) else 0.0
    out.append(ok)
  return out

def match_format_soft(prompts: List[str], completions: List[str], **kargs) -> List[float]:
  """Partial credit for including tags at least once; penalize duplication."""
  tags = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
  out = []
  for c in completions:
    score = 0.0
    cl = c.lower()
    for t in tags:
      n = cl.count(t)
      score += 0.5 if n == 1 else (-0.25 if n > 1 else -0.25)
    out.append(score)
  return out

def exact_match(prompts: List[str], completions: List[str], *, answer, **kargs) -> List[float]:
  """Case/space-normalized string exact match against ground truth."""
  # answer can be List[str] or str. For List[str], give credit if any matches.
  answers_per_item: Iterable[Sequence[str]]
  if isinstance(answer, (list, tuple)) and answer and isinstance(answer[0], (list, tuple)):
    # Already batched like List[List[str]]
    answers_per_item = answer  # type: ignore
  else:
    # Single answer per item; wrap
    answers_per_item = [[a] for a in (answer if isinstance(answer, list) else [answer]*len(completions))]

  out = []
  for c, golds in zip(completions, answers_per_item):
    ok = 1.0 if _any_equiv(c, golds) else 0.0
    out.append(ok)
  return out

def fuzzy_contains(prompts: List[str], completions: List[str], *, answer, **kargs) -> List[float]:
  """+0.5 if normalized gold string is a substring of completion (or vice versa)."""
  answers_per_item: Iterable[Sequence[str]]
  if isinstance(answer, (list, tuple)) and answer and isinstance(answer[0], (list, tuple)):
    answers_per_item = answer  # type: ignore
  else:
    answers_per_item = [[a] for a in (answer if isinstance(answer, list) else [answer]*len(completions))]

  out = []
  for c, golds in zip(completions, answers_per_item):
    cn = _normalize_text(c)
    hit = any(gn in cn or cn in gn for gn in (_normalize_text(g) for g in golds if g is not None))
    out.append(0.5 if hit else 0.0)
  return out

def numeric_tolerance(
    prompts: List[str],
    completions: List[str],
    *,
    answer,
    atol: float = 0.0,
    rtol: float = 0.0,
    partial_credit: float = 0.25,
    **kargs,
) -> List[float]:
  """Numeric reward: +1 if within tolerance to any numeric truth; partial if close.

  Args:
    atol: absolute tolerance for full credit
    rtol: relative tolerance for full credit
    partial_credit: credit if within 2x tolerances (soft)
  """
  answers_per_item: Iterable[Sequence[str]]
  if isinstance(answer, (list, tuple)) and answer and isinstance(answer[0], (list, tuple)):
    answers_per_item = answer  # type: ignore
  else:
    answers_per_item = [[a] for a in (answer if isinstance(answer, list) else [answer]*len(completions))]

  out = []
  for c, golds in zip(completions, answers_per_item):
    best_err = _closest_num_error(c, golds)
    if best_err is None:
      out.append(0.0)
      continue
    # Build tolerance based on first numeric gold we find
    tol_abs = math.inf
    tol_rel = math.inf
    for g in golds:
      gv = _extract_first_number(g) if g is not None else None
      if gv is not None:
        tol_abs = min(tol_abs, atol)
        tol_rel = min(tol_rel, rtol * abs(gv))
    tol = min(tol_abs, tol_rel) if (tol_abs != math.inf or tol_rel != math.inf) else 0.0

    if best_err <= tol:
      out.append(1.0)
    elif best_err <= max(2 * atol, 2 * tol_rel):
      out.append(partial_credit)
    else:
      out.append(0.0)
  return out

def brevity_penalty(prompts: List[str], completions: List[str], max_tokens: int = 64, **kargs) -> List[float]:
  """Small penalty if output looks excessively long (discourage rambling)."""
  out = []
  for c in completions:
    toks = _WS.split(_normalize_text(c))
    out.append(0.0 if len(toks) <= max_tokens else -0.25)
  return out