"""Real default generator and neighboring-workload rejection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from examples.frozenlake import data
from examples.frozenlake.gemma4_tim import data_contract, recipe


class DataTest(unittest.TestCase):
  def test_actual_default_generator_and_m15_or_order_drift(self):
    state = np.random.get_state()
    self.addCleanup(np.random.set_state, state)
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      data.create_dataset(data_dir=temp)
      self.assertEqual(data_contract.validate_training_data(root)["train_rows"], 10000)
      frame = pd.read_parquet(root / "train.parquet")
      for altered in (frame.iloc[::-1], frame.assign(size=12), frame.iloc[:9999]):
        altered.to_parquet(root / "train.parquet")
        with self.assertRaises(recipe.RecipeError):
          data_contract.validate_training_data(root)


if __name__ == "__main__":
  unittest.main()
