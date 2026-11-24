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
"""Wide-WikiQA, Multi-Hop QA, and BrowseComp-Plus task builders & grader."""

from __future__ import annotations

import collections
import json
from pathlib import Path
import random
import re
import string
from typing import Any

from tunix.dist_rl.swarm import contracts

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
  cleaned = _THINK_RE.sub("", str(text or ""))
  lowered = cleaned.lower().translate(_PUNC_TABLE)
  return " ".join(_ARTICLES_RE.sub(" ", lowered).split())


def token_f1(prediction: str, truth: str) -> float:
  pred_toks = normalize_answer(prediction).split()
  gold_toks = normalize_answer(truth).split()
  if not pred_toks or not gold_toks:
    return float(pred_toks == gold_toks)
  common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0
  prec = num_same / len(pred_toks)
  rec = num_same / len(gold_toks)
  return 2.0 * prec * rec / (prec + rec)


def best_qa_match(prediction: str, aliases: list[str]) -> tuple[float, float]:
  """Returns (best_em, best_f1) across acceptable answer aliases."""
  norm_pred = normalize_answer(prediction)
  best_em = 0.0
  best_f1 = 0.0
  for alias in aliases:
    norm_gold = normalize_answer(alias)
    if not norm_gold:
      continue
    em = 1.0 if (norm_gold == norm_pred or norm_gold in norm_pred) else 0.0
    f1 = max(em, token_f1(prediction, alias))
    best_em = max(best_em, em)
    best_f1 = max(best_f1, f1)
  return best_em, best_f1


def _extract_numbered_items(text: str, n_parts: int) -> dict[int, str]:
  """Extracts `(1) ...` / `1. ...` / `| 1 | ...` slots for exact item-level F1."""
  if n_parts <= 1:
    return {}
  cleaned = _THINK_RE.sub("", text or "")
  pattern = re.compile(r"(?:^|\n)\s*(?:\((\d+)\)|(\d+)[\.\):]|(?:\|\s*(\d+)\s*\|))\s*")
  matches = list(pattern.finditer(cleaned))
  out: dict[int, str] = {}
  for idx, m in enumerate(matches):
    num_str = m.group(1) or m.group(2) or m.group(3)
    slot = int(num_str) - 1
    if 0 <= slot < n_parts:
      start = m.end()
      end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
      out[slot] = cleaned[start:end].strip()
  return out


def _extract_answers(row: dict[str, Any]) -> list[str]:
  raw: Any = None
  for key in ("target_answers", "answers", "answer", "reference"):
    val = row.get(key)
    if val is not None and val != "" and val != []:
      raw = val
      break
  if isinstance(raw, list):
    return [str(x).strip() for x in raw if str(x).strip()]
  if isinstance(raw, dict):
    vals = raw.get("answers", raw.get("answer", []))
    return [
        str(x).strip()
        for x in (vals if isinstance(vals, list) else [vals])
        if str(x).strip()
    ]
  s = str(raw or "").strip()
  return [s] if s else []


def _clean_question(raw_prompt: str) -> str:
  text = str(raw_prompt or "").strip()
  for marker in ("\nPassages:", "\nContext:", "\nDocuments:"):
    idx = text.find(marker)
    if idx != -1:
      text = text[:idx].strip()
  q_idx = text.find("Question:")
  if q_idx != -1:
    text = text[q_idx + len("Question:") :].strip()
  return text


def _load_qa_rows(
    jsonl_path: Path, dataset_filter: set[str] | None = None
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  if not jsonl_path.is_file():
    return rows
  for line in jsonl_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    rec = json.loads(line)
    ds = str(rec.get("dataset", rec.get("metadata", {}).get("dataset", ""))).lower()
    if dataset_filter is not None and ds not in dataset_filter:
      continue
    prompt = _clean_question(str(rec.get("prompt", rec.get("question", ""))))
    answers = _extract_answers(rec)
    if prompt and answers:
      rows.append({
          "id": str(rec.get("task_id", rec.get("id", f"{ds}-{len(rows)}"))),
          "dataset": ds or jsonl_path.stem,
          "question": prompt,
          "answers": answers,
      })
  return rows


def _load_browsecomp_rows(browsecomp_dir: Path | None) -> list[dict[str, Any]]:
  if browsecomp_dir is None:
    return []
  q_path = browsecomp_dir / "queries.jsonl"
  if not q_path.is_file():
    return []
  rows: list[dict[str, Any]] = []
  for line in q_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    rec = json.loads(line)
    q = str(rec.get("query", "")).strip()
    ans = str(rec.get("answer", "")).strip()
    if q and ans:
      rows.append({
          "id": str(rec.get("query_id", f"bc-{len(rows)}")),
          "dataset": "browsecomp_plus",
          "question": q,
          "answers": [ans],
      })
  return rows


def _make_wide_item(
    items: list[dict[str, Any]], task_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
  lines = [f"({i + 1}) {it['question']}" for i, it in enumerate(items)]
  question = (
      f"Answer ALL {len(items)} independent factual questions below. "
      f"Present your final response with labeled lines (1) through ({len(items)}):\n"
      + "\n".join(lines)
  )
  pub = {
      "id": task_id,
      "public": {
          "question": question,
          "kind": "wide_wikiqa",
          "corpus": "wiki18",
          "n_parts": len(items),
      },
  }
  priv = {"kind": "wide_wikiqa", "sub_answers": [it["answers"] for it in items]}
  return pub, priv


def _pack_split(
    fact_rows: list[dict[str, Any]],
    mh_rows: list[dict[str, Any]],
    bc_rows: list[dict[str, Any]],
    *,
    n_wide: int,
    n_mh: int,
    n_bc: int,
    prefix: str,
    rng: random.Random,
    wide_min_width: int = 2,
    wide_max_width: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  pub_list: list[dict[str, Any]] = []
  priv_map: dict[str, Any] = {}
  lo = max(2, int(wide_min_width))
  hi = max(lo, int(wide_max_width))
  span = hi - lo + 1
  if fact_rows and n_wide > 0:
    cursor = 0
    for i in range(n_wide):
      width = lo + (i % span)
      picks = [fact_rows[(cursor + k) % len(fact_rows)] for k in range(width)]
      cursor += width
      pub, priv = _make_wide_item(picks, f"{prefix}-wide-{i:04d}")
      pub_list.append(pub)
      priv_map[pub["id"]] = priv
  if mh_rows and n_mh > 0:
    for i in range(min(n_mh, len(mh_rows))):
      row = mh_rows[i]
      tid = f"{prefix}-mh-{i:04d}"
      pub_list.append({
          "id": tid,
          "public": {
              "question": row["question"],
              "kind": "multihop_qa",
              "corpus": "wiki18",
              "n_parts": 1,
          },
      })
      priv_map[tid] = {"kind": "multihop_qa", "sub_answers": [row["answers"]]}
  if bc_rows and n_bc > 0:
    for i in range(min(n_bc, len(bc_rows))):
      row = bc_rows[i]
      tid = f"{prefix}-bc-{row['id']}"
      pub_list.append({
          "id": tid,
          "public": {
              "question": row["question"],
              "kind": "browsecomp_plus",
              "corpus": "browsecomp",
              "n_parts": 1,
          },
      })
      priv_map[tid] = {
          "kind": "browsecomp_plus",
          "sub_answers": [row["answers"]],
      }
  rng.shuffle(pub_list)
  return pub_list, priv_map


def build_parl_splits(
    upstream_root: Path,
    *,
    browsecomp_dir: Path | None = None,
    seed: int = 42,
    n_wide_train: int = 900,
    n_multihop_train: int = 900,
    n_browsecomp_train: int = 0,
    n_wide_eval: int = 48,
    n_multihop_eval: int = 48,
    n_browsecomp_eval: int = 0,
    wide_min_width: int = 2,
    wide_max_width: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
  """Builds disjoint train and heldout PARL tasks across Wide, Multi-Hop, and BrowseComp-Plus."""
  rng = random.Random(seed)
  train_pool = upstream_root / "data/r2orch/train/train_pool.jsonl"
  held_hotpot = upstream_root / "data/r2orch/heldout/hotpotqa.jsonl"
  held_nq = upstream_root / "data/r2orch/heldout/nq_open.jsonl"
  iid_nq = upstream_root / "data/r2orch/test/iid/nq_open.jsonl"
  ood_trivia = upstream_root / "data/r2orch/test/ood/triviaqa.jsonl"
  ood_musique = upstream_root / "data/r2orch/test/ood/musique.jsonl"

  nq_train = _load_qa_rows(train_pool, {"nq_open", "triviaqa"})
  mh_train = _load_qa_rows(train_pool, {"hotpotqa", "musique"})
  nq_eval = _load_qa_rows(held_nq) + _load_qa_rows(iid_nq) + _load_qa_rows(ood_trivia)
  mh_eval = _load_qa_rows(held_hotpot) + _load_qa_rows(ood_musique)
  bc_all = _load_browsecomp_rows(browsecomp_dir)

  rng.shuffle(nq_train)
  rng.shuffle(mh_train)
  rng.shuffle(nq_eval)
  rng.shuffle(mh_eval)
  rng.shuffle(bc_all)

  n_bc_ev = min(int(n_browsecomp_eval), len(bc_all))
  bc_eval = bc_all[:n_bc_ev]
  bc_train = bc_all[n_bc_ev:]

  train_pub, train_priv = _pack_split(
      nq_train,
      mh_train,
      bc_train,
      n_wide=n_wide_train,
      n_mh=n_multihop_train,
      n_bc=n_browsecomp_train,
      prefix="tr",
      rng=rng,
      wide_min_width=wide_min_width,
      wide_max_width=wide_max_width,
  )
  eval_pub, eval_priv = _pack_split(
      nq_eval,
      mh_eval,
      bc_eval,
      n_wide=n_wide_eval,
      n_mh=n_multihop_eval,
      n_bc=n_bc_ev,
      prefix="ev",
      rng=rng,
      wide_min_width=wide_min_width,
      wide_max_width=wide_max_width,
  )
  if not train_pub or not eval_pub:
    raise ValueError(
        f"Empty PARL split under {upstream_root}: "
        f"train={len(train_pub)} eval={len(eval_pub)}"
    )
  return train_pub, train_priv, eval_pub, eval_priv


def grade_parl_episode(
    episode: contracts.EpisodeRecord,
    private_example: dict[str, Any],
) -> contracts.ScoreRecord:
  """Private grader computing Wide item-level F1/EM, Multi-Hop, and BrowseComp-Plus scores."""
  pred = _THINK_RE.sub("", str(episode.public_output or "")).strip()
  sub_answers: list[list[str]] = list(private_example["sub_answers"])
  if not pred or not sub_answers:
    return contracts.ScoreRecord(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        reward=0.0,
        components={"r_perf": 0.0, "em": 0.0, "f1": 0.0},
    )
  slots = _extract_numbered_items(pred, len(sub_answers))
  ems: list[float] = []
  f1s: list[float] = []
  for idx, aliases in enumerate(sub_answers):
    slot_pred = slots.get(idx, pred)
    em_slot, f1_slot = best_qa_match(slot_pred, aliases)
    em_full, _ = best_qa_match(pred, aliases)
    ems.append(max(em_slot, em_full))
    f1s.append(max(f1_slot, em_full))
  mean_em = sum(ems) / len(ems)
  mean_f1 = sum(f1s) / len(f1s)
  r_perf = 0.5 * mean_em + 0.5 * mean_f1
  return contracts.ScoreRecord(
      episode_id=episode.episode_id,
      task_id=episode.task_id,
      reward=r_perf,
      components={"r_perf": r_perf, "em": mean_em, "f1": mean_f1},
  )
