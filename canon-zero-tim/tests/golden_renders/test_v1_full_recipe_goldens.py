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
        "c0c72e9b9f2203094488a51f108e5ba4d0337978567f87d21689189cbf9c1a1d",
        "f1b989dd35e1893dd11e55acd47071c792193eae4cfe90437047a754fb21c9a0",
        "7c3f92e5ff7310ed9d881f8c499d099eb7600edcd277f5b06fa78436c2cfd796",
    )
    with tempfile.TemporaryDirectory() as tmp:
      first = self._render_three(Path(tmp) / "first")
      second = self._render_three(Path(tmp) / "second")
      self.assertEqual(tuple(map(_sha256, first)), expected)
      self.assertEqual(tuple(map(_sha256, second)), expected)
      for path in (*first, *second):
        self.assertNotIn("CANON_DP_REDUCE_ONCE", _env(path))
        self.assertNotIn("CANON_P32_KEEP_TAPE", _env(path))

  def test_frozenlake_two_full_manifests_match_exact_goldens_twice(self):
    expected = (
        "1d831f517551201819c136a34be2cdd593a6dbd7ba64c02a96b6c4e016859680",
        "727d3cabf63b98c67e9a6351269625f3928dc34c0a9163bdb3967ca64e5daf49",
    )
    with tempfile.TemporaryDirectory() as tmp:
      first = self._render_two(Path(tmp) / "first")
      second = self._render_two(Path(tmp) / "second")
      self.assertEqual(tuple(map(_sha256, first)), expected)
      self.assertEqual(tuple(map(_sha256, second)), expected)
      for path in (*first, *second):
        self.assertNotIn("CANON_DP_REDUCE_ONCE", _env(path))
        self.assertNotIn("CANON_P32_KEEP_TAPE", _env(path))

  def test_image_receipt_and_profile_defaults_are_the_only_legacy_deltas(self):
    # The TiTO provenance change added CANON_CLIENT_IMAGE to FrozenLake.
    # Retain all old goldens as a reconstruction oracle: removing precisely
    # that independently checked receipt, then restoring only the stream/1
    # entries now owned by the profile, must recover every old byte hash.
    legacy = {
        "three": (
            "8b4fd423073bc1b8fc39a3a4bac60d418411392018c2e1bf7df37d0062ccc341",
            "86d1f6666c2425011be5484ebbe9c8d2240685793f4f09c167ab60b10cea40ce",
            "ef25b597442d1059a62ee91b42bec969804c14c0359fe50cea7140d8e94b2b51",
        ),
        "two": (
            "c35b41922623642f920a5656d27dc04c010a31ae64599b7768376205c21a9998",
            "0e4c9fdfde5b2c3ac88116c0df3cd02370a47d1d70592033836c906687a16eea",
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
      for label, render in (("three", self._render_three), ("two", self._render_two)):
        for path, expected in zip(render(Path(tmp) / label), legacy[label], strict=True):
          raw = path.read_text(encoding="utf-8")
          document = THREE.yaml.safe_load(raw)
          pod = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
          main = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
          receipts = [item for item in main["env"] if item["name"] == "CANON_CLIENT_IMAGE"]
          is_frozenlake = _env(path)["CANON_PROFILE_FILE"] != THREE._GSM8K_PROFILE
          self.assertEqual(receipts, [{"name": "CANON_CLIENT_IMAGE", "value": main["image"]}]
                           if is_frozenlake else [])
          main["env"] = [item for item in main["env"] if item["name"] != "CANON_CLIENT_IMAGE"]
          entries = main["env"]
          for name in ("CANON_P32_KEEP_TAPE", "CANON_DP_REDUCE_ONCE"):
            self.assertTrue(all(item["name"] != name for item in entries), name)
          position = 1 + max(
              index for index, item in enumerate(entries)
              if item["name"] in ("CANON_P71_SCAN", "CANON_P67_P66_VMA_P59_ONLY")
          )
          entries[position:position] = [
              {"name": "CANON_P32_KEEP_TAPE", "value": "stream"},
              {"name": "CANON_DP_REDUCE_ONCE", "value": "1"},
          ]
          header = "\n".join(raw.splitlines()[:2]) + "\n"
          reconstructed = header + THREE.yaml.safe_dump(document, sort_keys=False)
          self.assertEqual(hashlib.sha256(reconstructed.encode()).hexdigest(), expected)


if __name__ == "__main__":
  unittest.main()
