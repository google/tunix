"""Regression tests for the Qwen3-4B R2E-Gym action dialect."""

import unittest

from examples.deepswe.r2egym_action_compat import canonicalize_r2egym_action
from examples.deepswe.swe_agent import parse_xml_response


class R2EGymActionCompatTest(unittest.TestCase):

  def test_observed_editor_call_is_executable_after_repair(self):
    response = (
        "<function=file_editor>\n"
        "<parameter=command=view>\n"
        "<parameter=path=./scrapy/utils/template.py>\n"
        "</parameter>\n</parameter>\n</function>"
    )
    _, strict_action = parse_xml_response(response)
    self.assertNotIn("command", strict_action.parameters)

    _, repaired_action = parse_xml_response(
        response, action_compat_mode="q4_r2egym_xml_v2"
    )
    self.assertEqual(repaired_action.function_name, "file_editor")
    self.assertEqual(repaired_action.parameters["command"], "view")
    self.assertEqual(
        repaired_action.parameters["path"], "./scrapy/utils/template.py"
    )

  def test_correct_call_is_unchanged(self):
    action = (
        "<function=file_editor>"
        "<parameter=command>view</parameter>"
        "<parameter=path>a.py</parameter>"
        "</function>"
    )
    self.assertEqual(canonicalize_r2egym_action(action), (action, 0))


if __name__ == "__main__":
  unittest.main()
