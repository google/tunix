"""Validate the selected default training distribution, without changing it."""

from pathlib import Path

import numpy as np

from examples.frozenlake.gemma4_tim import recipe


def validate_training_data(root: Path) -> dict:
  import pandas as pd
  frame = pd.read_parquet(root / "train.parquet")
  if len(frame) != 10000 or set(frame.columns) != {"env_name", "seed", "size", "p"}:
    raise recipe.RecipeError("training data is not the default 10000-row FrozenLake schema")
  # Same RandomState sequence as examples/frozenlake/data.py, without mutating
  # the process-global NumPy RNG. This is validation, not a new generator.
  rng = np.random.RandomState(42)
  expected = {"seed": rng.randint(0, 100000, size=10000),
              "size": rng.randint(2, 10, size=10000),
              "p": rng.uniform(.6, .85, size=10000)}
  if not (frame["env_name"] == "frozenlake").all():
    raise recipe.RecipeError("training data has another environment")
  for name, values in expected.items():
    actual = frame[name].to_numpy()
    if actual.dtype != values.dtype or not np.array_equal(actual, values):
      raise recipe.RecipeError("training data differs from the default seed42 generator: " + name)
  return {"recipe": "examples/frozenlake/data.py:create_dataset defaults",
          "train_rows": 10000, "seed": 42, "grid_sizes": "2..9",
          "p_interval": [.6, .85], "semantic_rows": "EXACT",
          "test_split": "payload-hashed-not-used-by-five-update-carrier"}
