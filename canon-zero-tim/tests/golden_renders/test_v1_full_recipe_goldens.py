"""Exact golden renders for the production V1 full-recipe bundle."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    ROOT / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
)
BASE = ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


THREE = _load("v1_full_recipe_golden_three", SCRIPTS / "render_three_full_recipes.py")
TWO = _load(
    "v1_full_recipe_golden_two",
    SCRIPTS / "render_p67_frozenlake_two_full_recipes.py",
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _env(path: Path) -> dict[str, str]:
  document = THREE.yaml.safe_load(path.read_text(encoding="utf-8"))
  return THREE._env(document)  # pylint: disable=protected-access


class V1FullRecipeGoldensTest(unittest.TestCase):

  def _render_three(self, root: Path) -> tuple[Path, ...]:
    return THREE.render_three(
        source_commit="a" * 40,
        output_dir=root,
        gsm8k_run_id="g64a",
        p45_run_id="f45a",
        m15_run_id="m15a",
        campaign_root="v1hp-a",
        base_path=BASE,
    )

  def _render_two(self, root: Path) -> tuple[Path, ...]:
    return TWO.render_two(
        source_commit="b" * 40,
        output_dir=root,
        p45_run_id="p45p67a",
        m15_run_id="m15p67a",
        campaign_root="v1p67-a",
        base_path=BASE,
    )

  def test_three_full_manifests_match_exact_goldens_twice(self):
    expected = (
        "8b4fd423073bc1b8fc39a3a4bac60d418411392018c2e1bf7df37d0062ccc341",
        "86d1f6666c2425011be5484ebbe9c8d2240685793f4f09c167ab60b10cea40ce",
        "ef25b597442d1059a62ee91b42bec969804c14c0359fe50cea7140d8e94b2b51",
    )
    with tempfile.TemporaryDirectory() as tmp:
      first = self._render_three(Path(tmp) / "first")
      second = self._render_three(Path(tmp) / "second")
      self.assertEqual(tuple(map(_sha256, first)), expected)
      self.assertEqual(tuple(map(_sha256, second)), expected)
      for path in (*first, *second):
        self.assertEqual(_env(path)["CANON_DP_REDUCE_ONCE"], "1")

  def test_frozenlake_two_full_manifests_match_exact_goldens_twice(self):
    expected = (
        "c35b41922623642f920a5656d27dc04c010a31ae64599b7768376205c21a9998",
        "0e4c9fdfde5b2c3ac88116c0df3cd02370a47d1d70592033836c906687a16eea",
    )
    with tempfile.TemporaryDirectory() as tmp:
      first = self._render_two(Path(tmp) / "first")
      second = self._render_two(Path(tmp) / "second")
      self.assertEqual(tuple(map(_sha256, first)), expected)
      self.assertEqual(tuple(map(_sha256, second)), expected)
      for path in (*first, *second):
        self.assertEqual(_env(path)["CANON_DP_REDUCE_ONCE"], "1")


if __name__ == "__main__":
  unittest.main()
