"""Regression contracts for Q4-to-R2E tool action normalization."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples" / "deepswe"))

from r2egym_action_compat import canonicalize_r2egym_action  # pylint: disable=wrong-import-position


class R2egymActionCompatTest(unittest.TestCase):

  def test_correct_file_editor_action_is_unchanged(self):
    action = (
        "<function=file_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>a.py</parameter>\n"
        "</function>"
    )
    self.assertEqual(canonicalize_r2egym_action(action), (action, 0))

  def test_observed_inline_form_is_narrowly_canonicalized(self):
    action = (
        "<function=file_editor>\n"
        "<parameter=command=view>\n"
        "<parameter=path=aiohttp/connector.py>\n"
        "<parameter=view_range=[1, 100]>\n"
        "</parameter>\n</parameter>\n</function>"
    )
    canonical, repairs = canonicalize_r2egym_action(action)
    self.assertEqual(repairs, 3)
    self.assertIn("<parameter=command>view</parameter>", canonical)
    self.assertIn(
        "<parameter=path>aiohttp/connector.py</parameter>", canonical
    )
    self.assertIn("<parameter=view_range>[1, 100]</parameter>", canonical)
    self.assertNotIn("<parameter=command=view>", canonical)

  def test_observed_search_inline_form_is_canonicalized(self):
    action = (
        "<function=search>"
        "<parameter=search_term=_resolve_host>"
        "<parameter=path=aiohttp/connector.py>"
        "</function>"
    )
    canonical, repairs = canonicalize_r2egym_action(action)
    self.assertEqual(repairs, 2)
    self.assertIn(
        "<parameter=search_term>_resolve_host</parameter>", canonical
    )
    self.assertIn(
        "<parameter=path>aiohttp/connector.py</parameter>", canonical
    )

  def test_file_editor_command_shorthand_is_canonicalized(self):
    action = (
        "<function=file_editor>"
        "<parameter=str_replace>"
        "<parameter=path>a.py</parameter>"
        "</function>"
    )
    canonical, repairs = canonicalize_r2egym_action(action)
    self.assertEqual(repairs, 1)
    self.assertIn(
        "<parameter=command>str_replace</parameter>", canonical
    )

  def test_unknown_tools_are_not_rewritten(self):
    action = (
        "<function=unknown_tool>"
        "<parameter=cmd=python -c 'x=1'>"
        "</function>"
    )
    self.assertEqual(canonicalize_r2egym_action(action), (action, 0))


if __name__ == "__main__":
  unittest.main()
