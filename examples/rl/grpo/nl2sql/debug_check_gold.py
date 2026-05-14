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

"""Sanity-check gold SQL against the example SQLite database."""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = ROOT / "example.sqlite"
DATA_PATH = ROOT / "nl2sql_data.jsonl"


def run_sql(sql: str) -> List[List[object]]:
  with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return [list(row) for row in rows]


def main() -> None:
  with DATA_PATH.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      rid = obj["id"]
      gold_sql = obj["gold_sql"]
      gold_result = obj["gold_result"]

      result = run_sql(gold_sql)
      if result != gold_result:
        print(f"[MISMATCH] {rid}")
        print("  SQL:", gold_sql)
        print("  DB result:", result)
        print("  gold_result:", gold_result)
      else:
        print(f"[OK] {rid}")


if __name__ == "__main__":
  main()

