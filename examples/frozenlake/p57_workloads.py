"""Deterministic materialized FrozenLake recipes for the P57 TIM study."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class Recipe:
  """One preregistered workload envelope."""

  name: str
  min_grid_side: int
  max_grid_side: int
  max_turns: int
  context_hard_cap: int
  frozen_probability: float | None
  eligible: bool

  def grid_sides(self) -> tuple[int, ...]:
    return tuple(range(self.min_grid_side, self.max_grid_side + 1))

  def path_envelope(self, side: int) -> tuple[int, int]:
    if side not in self.grid_sides():
      raise ValueError(f"grid side {side} is outside recipe {self.name}")
    if self.name == "l0":
      return 1, min(self.max_turns, 2 * (side - 1))
    return (
        max(4, side - 1),
        min(self.max_turns, side + 5, 2 * (side - 1)),
    )


RECIPES = {
    recipe.name: recipe
    for recipe in (
        # Materialized legacy-envelope anchor; never eligible for selection.
        Recipe("l0", 2, 9, 5, 6_144, None, False),
        # Easy-Mix explicitly includes 10x10 (user correction, 2026-08-21).
        Recipe("m10", 5, 10, 10, 8_192, 0.82, True),
        Recipe("m15", 5, 12, 15, 12_288, 0.82, True),
        Recipe("m20", 5, 15, 20, 16_384, 0.82, True),
    )
}

# Every P57 prompt produces one complete DP8 row group.  This is shared by
# calibration, training, and isolated evaluation: the evaluation rescore maps
# this caller-global row axis over DP8, so a smaller count is not admissible.
GENERATIONS_PER_PROMPT = 8

# The original P45 workload predates the materialized P57 recipes.  Preserve
# its exact generator namespaces here so a stale or hand-edited parquet cannot
# silently become a different paired workload.
P45_GENERATOR_SEEDS = {"train": 42, "eval": 123}

# Calibration and final paired-study data are deliberately disjoint.
_SPLIT_SEEDS = {
    "calibration": {"train": 57_000_000, "eval": 57_100_000},
    "selection": {"train": 57_200_000, "eval": 57_300_000},
    "main": {"train": 57_400_000, "eval": 57_500_000},
}

# Immutable primary-study dataset identities.  These literals make a source
# change to either generator visible before any expensive run starts.
PRIMARY_DATASET_SHA256 = {
    ("p45", "legacy", "train", 10_000): (
        "ddc96fd9ae4e807d8aa8e800795aa743e423ffe4f936f681596460d28e670487"
    ),
    ("p45", "legacy", "eval", 100): (
        "b10add7f31b2cc9931c65b4cc59780004fd3d52a4fce9d20ed565c87df44b580"
    ),
    ("m15", "main", "train", 10_000): (
        "ff1e659b80a0c9bd640e616972a523132f4a333ef174b1a0b13b202958a30e43"
    ),
    ("m15", "main", "eval", 100): (
        "8edb61cb995b4abe8d3f90b32e961be74b8b74ab46120e0d43513ea26d324089"
    ),
}


def recipe(name: str) -> Recipe:
  try:
    return RECIPES[name]
  except KeyError as exc:
    raise ValueError(
        f"unknown P57 workload recipe {name!r}; expected {tuple(RECIPES)}"
    ) from exc


# Compatibility names local to the uncommitted P57 branch.
candidate = recipe
CANDIDATES = RECIPES


def validate_split(name: str) -> str:
  if name not in _SPLIT_SEEDS:
    raise ValueError(
        f"unknown P57 data split {name!r}; expected {tuple(_SPLIT_SEEDS)}"
    )
  return name


def _positions(desc: Iterable[str], symbol: str) -> list[tuple[int, int]]:
  return [
      (row, column)
      for row, line in enumerate(desc)
      for column, value in enumerate(line)
      if value == symbol
  ]


def shortest_path_length(desc: Iterable[str]) -> int | None:
  """Returns the shortest safe S-to-G path length, or None when unreachable."""
  rows = tuple(str(line) for line in desc)
  if not rows or any(len(line) != len(rows) for line in rows):
    raise ValueError("FrozenLake map must be a nonempty square")
  starts = _positions(rows, "S")
  goals = set(_positions(rows, "G"))
  if len(starts) != 1 or len(goals) != 1:
    raise ValueError("FrozenLake map must contain exactly one S and one G")
  queue = deque([(starts[0], 0)])
  seen = {starts[0]}
  while queue:
    (row, column), distance = queue.popleft()
    if (row, column) in goals:
      return distance
    for drow, dcolumn in ((1, 0), (-1, 0), (0, 1), (0, -1)):
      next_position = (row + drow, column + dcolumn)
      next_row, next_column = next_position
      if not (0 <= next_row < len(rows) and 0 <= next_column < len(rows)):
        continue
      if next_position in seen or rows[next_row][next_column] == "H":
        continue
      seen.add(next_position)
      queue.append((next_position, distance + 1))
  return None


def _map_sha256(desc: Iterable[str]) -> str:
  payload = ("\n".join(str(line) for line in desc) + "\n").encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def _row_parameters(
    spec: Recipe, index: int, seed: int
) -> tuple[int, float]:
  sides = spec.grid_sides()
  side = sides[index % len(sides)]
  if spec.frozen_probability is None:
    # Historical envelope p in [0.60, 0.85], without claiming map identity.
    probability = round(random.Random(seed ^ 0x57A11CE).uniform(0.60, 0.85), 6)
  else:
    probability = spec.frozen_probability
  return side, probability


@lru_cache(maxsize=None)
def _endpoint_choices(
    side: int, minimum: int, maximum: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
  return tuple(
      ((start_row, start_column), (goal_row, goal_column))
      for start_row in range(side)
      for start_column in range(side)
      for goal_row in range(side)
      for goal_column in range(side)
      if minimum
      <= abs(start_row - goal_row) + abs(start_column - goal_column)
      <= maximum
  )


def _sample_endpoints(
    rng: random.Random, side: int, minimum: int, maximum: int
) -> tuple[tuple[int, int], tuple[int, int]]:
  choices = _endpoint_choices(side, minimum, maximum)
  if not choices:
    raise ValueError(
        f"no endpoint pair for side={side} path=[{minimum},{maximum}]"
    )
  return rng.choice(choices)


def _monotone_path(
    rng: random.Random,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
  row, column = start
  steps = []
  steps.extend([(1 if goal[0] > row else -1, 0)] * abs(goal[0] - row))
  steps.extend([(0, 1 if goal[1] > column else -1)] * abs(goal[1] - column))
  rng.shuffle(steps)
  path = [(row, column)]
  for drow, dcolumn in steps:
    row += drow
    column += dcolumn
    path.append((row, column))
  assert path[-1] == goal
  return tuple(path)


def _generate_map(
    spec: Recipe, *, index: int, seed: int
) -> tuple[tuple[str, ...], int, float]:
  """Constructs a random map with a guaranteed shortest-path envelope."""
  side, probability = _row_parameters(spec, index, seed)
  minimum, maximum = spec.path_envelope(side)
  rng = random.Random(seed)
  start, goal = _sample_endpoints(rng, side, minimum, maximum)
  path = _monotone_path(rng, start, goal)
  board = [
      ["F" if rng.random() < probability else "H" for _ in range(side)]
      for _ in range(side)
  ]
  for row, column in path:
    board[row][column] = "F"
  board[start[0]][start[1]] = "S"
  board[goal[0]][goal[1]] = "G"
  desc = tuple("".join(row) for row in board)
  distance = shortest_path_length(desc)
  expected_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
  if distance != expected_distance or not (minimum <= distance <= maximum):
    raise AssertionError(
        "P57 constructive materializer violated its path proof: "
        f"recipe={spec.name} side={side} distance={distance} "
        f"expected={expected_distance} envelope=({minimum},{maximum})"
    )
  return desc, side, probability


def _materialize_role(
    recipe_name: str,
    split: str,
    role: str,
    count: int,
    *,
    seen_map_shas: set[str] | None = None,
) -> list[dict[str, Any]]:
  spec = recipe(recipe_name)
  validate_split(split)
  if role not in ("train", "eval"):
    raise ValueError("P57 dataset role must be train or eval")
  if count <= 0:
    raise ValueError("P57 dataset count must be positive")
  assigned = set() if seen_map_shas is None else seen_map_shas
  records = []
  for index in range(count):
    collision = 0
    while True:
      seed = _SPLIT_SEEDS[split][role] + index + collision * 100_000_000
      desc, side, probability = _generate_map(spec, index=index, seed=seed)
      map_sha = _map_sha256(desc)
      if map_sha not in assigned:
        assigned.add(map_sha)
        break
      collision += 1
    distance = shortest_path_length(desc)
    assert distance is not None
    records.append({
        "env_name": "frozenlake",
        "seed": seed,
        "size": side,
        "p": probability,
        "is_slippery": False,
        "desc_json": json.dumps(desc, separators=(",", ":")),
        "shortest_path": distance,
        "map_sha256": map_sha,
        "p57_candidate": recipe_name,
        "p57_data_split": split,
        "p57_role": role,
        "p57_index": index,
    })
  attest_records(records, recipe_name, split, role, expected_count=count)
  return records


def materialize_records(
    recipe_name: str, split: str, role: str, count: int
) -> list[dict[str, Any]]:
  return _materialize_role(recipe_name, split, role, count)


def materialize_dataset_pair(
    recipe_name: str,
    split: str,
    *,
    train_count: int,
    eval_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Returns one split after de-duplicating every preceding namespace."""
  validate_split(split)
  seen: set[str] = set()
  selected = None
  for namespace in _SPLIT_SEEDS:
    train = _materialize_role(
        recipe_name, namespace, "train", train_count, seen_map_shas=seen
    )
    evaluation = _materialize_role(
        recipe_name, namespace, "eval", eval_count, seen_map_shas=seen
    )
    if namespace == split:
      selected = (train, evaluation)
      break
  assert selected is not None
  return selected


def attest_p45_records(
    records: Iterable[Mapping[str, Any]],
    role: str,
    *,
    expected_count: int,
) -> str:
  """Validates one legacy P45 split and returns its canonical SHA-256."""
  try:
    generator_seed = P45_GENERATOR_SEEDS[role]
  except KeyError as exc:
    raise ValueError("P45 dataset role must be train or eval") from exc
  rows = [
      {key: _plain(value) for key, value in dict(raw).items()}
      for raw in records
  ]
  if len(rows) != expected_count:
    raise ValueError(
        f"P45 dataset row count drifted: {len(rows)} != {expected_count}"
    )
  expected_rows = materialize_p45_records(role, expected_count)
  canonical_rows = []
  for index, (row, expected) in enumerate(zip(rows, expected_rows)):
    if row != expected:
      wrong = {
          key: row.get(key)
          for key, value in expected.items()
          if row.get(key) != value
      }
      extra = sorted(set(row) - set(expected))
      raise ValueError(
          "P45 dataset row drifted: "
          f"role={role} index={index} wrong={wrong} extra={extra}"
      )
    canonical_rows.append(expected)
  payload = json.dumps(
      {
          "schema": "p57-p45-generator-dataset-v1",
          "role": role,
          "generator_seed": generator_seed,
          "rows": canonical_rows,
      },
      sort_keys=True,
      separators=(",", ":"),
  ).encode("utf-8")
  digest = hashlib.sha256(payload).hexdigest()
  registered = PRIMARY_DATASET_SHA256.get(
      ("p45", "legacy", role, expected_count)
  )
  if registered is not None and digest != registered:
    raise ValueError(
        f"P45 registered dataset SHA drifted: {digest} != {registered}"
    )
  return digest


def materialize_p45_records(role: str, count: int) -> list[dict[str, Any]]:
  """Reconstructs the original seed-42/123 P45 parameter rows."""
  try:
    generator_seed = P45_GENERATOR_SEEDS[role]
  except KeyError as exc:
    raise ValueError("P45 dataset role must be train or eval") from exc
  if count <= 0:
    raise ValueError("P45 dataset count must be positive")
  rng = np.random.RandomState(generator_seed)
  seeds = rng.randint(0, 100_000, size=count)
  sizes = rng.randint(2, 10, size=count)
  probabilities = rng.uniform(0.60, 0.85, size=count)
  return [
      {
          "env_name": "frozenlake",
          "seed": int(seed),
          "size": int(side),
          "p": float(probability),
      }
      for seed, side, probability in zip(seeds, sizes, probabilities)
  ]


def _plain(value: Any) -> Any:
  return value.item() if hasattr(value, "item") else value


def attest_records(
    records: Iterable[Mapping[str, Any]],
    recipe_name: str,
    split: str,
    role: str,
    *,
    expected_count: int,
) -> str:
  """Validates rows and returns a canonical dataset SHA-256."""
  spec = recipe(recipe_name)
  validate_split(split)
  rows = list(records)
  if len(rows) != expected_count:
    raise ValueError(
        f"P57 dataset row count drifted: {len(rows)} != {expected_count}"
    )
  canonical_rows = []
  for expected_index, raw in enumerate(rows):
    row = {key: _plain(value) for key, value in dict(raw).items()}
    try:
      desc = tuple(json.loads(str(row["desc_json"])))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
      raise ValueError("P57 dataset row has invalid desc_json") from exc
    seed = int(row["seed"])
    expected_side, expected_probability = _row_parameters(
        spec, expected_index, seed
    )
    distance = shortest_path_length(desc)
    expected = {
        "env_name": "frozenlake",
        "size": expected_side,
        "p": expected_probability,
        "is_slippery": False,
        "shortest_path": distance,
        "map_sha256": _map_sha256(desc),
        "p57_candidate": recipe_name,
        "p57_data_split": split,
        "p57_role": role,
        "p57_index": expected_index,
    }
    wrong = {
        key: row.get(key)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if wrong:
      raise ValueError(f"P57 dataset row contract drifted: {wrong}")
    minimum, maximum = spec.path_envelope(expected_side)
    if distance is None or not (minimum <= distance <= maximum):
      raise ValueError(
          f"P57 shortest-path envelope drifted: index={expected_index} "
          f"distance={distance} envelope=({minimum},{maximum})"
      )
    canonical_rows.append({
        **expected,
        "seed": seed,
        "desc_json": json.dumps(desc, separators=(",", ":")),
    })
  seeds = [row["seed"] for row in canonical_rows]
  map_shas = [row["map_sha256"] for row in canonical_rows]
  if len(set(seeds)) != len(seeds):
    raise ValueError("P57 dataset contains duplicate generation seeds")
  if len(set(map_shas)) != len(map_shas):
    raise ValueError("P57 dataset contains duplicate materialized maps")
  payload = json.dumps(
      {
          "schema": "p57-frozenlake-materialized-dataset-v2",
          "recipe": asdict(spec),
          "split": split,
          "role": role,
          "rows": canonical_rows,
      },
      sort_keys=True,
      separators=(",", ":"),
  ).encode("utf-8")
  digest = hashlib.sha256(payload).hexdigest()
  registered = PRIMARY_DATASET_SHA256.get(
      (recipe_name, split, role, expected_count)
  )
  if registered is not None and digest != registered:
    raise ValueError(
        "P57 registered dataset SHA drifted: "
        f"{recipe_name}/{split}/{role} {digest} != {registered}"
    )
  return digest
