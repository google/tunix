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

"""Builds the SQLite database for the NL2SQL GRPO example."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> None:
  example_dir = Path(__file__).resolve().parent
  schema_path = example_dir / "schema.sql"
  db_path = example_dir / "example.sqlite"

  if not schema_path.exists():
    raise FileNotFoundError(f"Missing schema file: {schema_path}")

  if db_path.exists():
    db_path.unlink()

  schema_sql = schema_path.read_text(encoding="utf-8")
  with sqlite3.connect(db_path) as conn:
    conn.executescript(schema_sql)
    conn.commit()

  print(f"SQLite DB created at {db_path}")


if __name__ == "__main__":
  main()

