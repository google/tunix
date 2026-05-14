#!/usr/bin/env python3
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

"""Dataset loader for the NL2SQL GRPO example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import grain


SYSTEM_PROMPT = (
    "You translate natural language questions into SQLite SQL. "
    "Only output a single SQL SELECT statement. "
    "Do not include explanations, comments, or markdown."
)

SCHEMA_TEXT = """Tables:
customers(customer_id, name, city)
products(product_id, name, price)
orders(order_id, customer_id, product_id, order_date, quantity)"""

USER_TEMPLATE = """Schema:
{schema}

Example:
Question: How many customers are there?
SQL: SELECT COUNT(*) FROM customers;

Question: {question}
SQL:"""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  records: list[dict[str, Any]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    records.append(json.loads(line))
  return records


def create_dataset(data_path: str | None = None) -> grain.MapDataset:
  """Loads the JSONL dataset and formats prompts for chat templating."""
  example_dir = Path(__file__).resolve().parent
  dataset_path = Path(data_path) if data_path else example_dir / "nl2sql_data.jsonl"

  if not dataset_path.exists():
    raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

  data = _load_jsonl(dataset_path)

  def _process_example(example: dict[str, Any]) -> dict[str, Any]:
    question = example["question"]
    prompt_text = USER_TEMPLATE.format(schema=SCHEMA_TEXT, question=question)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "id": example["id"],
        "question": question,
        "gold_sql": example["gold_sql"],
        "gold_result": example["gold_result"],
    }

  return grain.MapDataset.source(data).map(_process_example)

