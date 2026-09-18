"""Source-relative imports for the renderer's shared priority contract."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


_RENDERER = (
    Path(__file__).resolve().parents[2] / "cluster/render_jobsets.py"
)
_IMPORT_PROBE = """
import importlib.util
from pathlib import Path
import sys
import types

source = Path(sys.argv[1])
shadow = types.ModuleType("render_deepswe_jobset")
shadow.PRIORITY_CLASS = "wrong-ambient-contract"
sys.modules[shadow.__name__] = shadow
path_before = list(sys.path)
spec = importlib.util.spec_from_file_location("p33_import_probe", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.p34 is not shadow
assert Path(module.p34.__file__).resolve() == source.with_name("render_deepswe_jobset.py").resolve()
assert module._PRIORITY_CLASS == module.p34.PRIORITY_CLASS == "medium"
assert sys.path == path_before
assert sys.modules[shadow.__name__] is shadow
print("P33_SOURCE_RELATIVE_IMPORT_PASS")
"""


class RendererImportTest(unittest.TestCase):

  def test_import_by_path_ignores_ambient_sibling_without_changing_search_path(self):
    with tempfile.TemporaryDirectory() as tmp:
      result = subprocess.run(
          [sys.executable, "-I", "-c", _IMPORT_PROBE, str(_RENDERER)],
          cwd=tmp,
          capture_output=True,
          text=True,
          check=False,
      )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("P33_SOURCE_RELATIVE_IMPORT_PASS", result.stdout)

  def test_missing_source_sibling_does_not_fall_back_to_ambient_module(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = Path(tmp) / _RENDERER.name
      shutil.copyfile(_RENDERER, source)
      result = subprocess.run(
          [sys.executable, "-I", "-c", _IMPORT_PROBE, str(source)],
          cwd=tmp,
          capture_output=True,
          text=True,
          check=False,
      )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("FileNotFoundError", result.stderr)
    self.assertIn("render_deepswe_jobset.py", result.stderr)
    self.assertNotIn("P33_SOURCE_RELATIVE_IMPORT_PASS", result.stdout)

  def test_cli_import_works_outside_the_source_directory(self):
    with tempfile.TemporaryDirectory() as tmp:
      result = subprocess.run(
          [sys.executable, str(_RENDERER), "--help"],
          cwd=tmp,
          capture_output=True,
          text=True,
          check=False,
      )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--source-commit", result.stdout)


if __name__ == "__main__":
  unittest.main()
