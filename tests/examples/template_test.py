# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for examples/deepswe/template.py."""

import pytest
from examples.deepswe import template


def test_prompt_constants_exist():
  """Verify that all DeepSWE prompt constants are defined and non-empty."""
  assert template.SWE_SYSTEM_PROMPT_FN_CALL
  assert template.SWE_SYSTEM_PROMPT
  assert template.SWEAGENT_SYSTEM_PROMPT
  assert template.SWE_USER_PROMPT_FN_CALL
  assert template.SWE_USER_PROMPT
  assert template.SWEAGENT_USER_PROMPT


def test_get_system_prompt():
  """Verify get_system_prompt returns correct prompt for all scaffolds and modes."""
  assert (
      template.get_system_prompt("r2egym", use_fn_calling=False)
      == template.SWE_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("r2egym", use_fn_calling=True)
      == template.SWE_SYSTEM_PROMPT_FN_CALL
  )
  assert (
      template.get_system_prompt("sweagent", use_fn_calling=False)
      == template.SWEAGENT_SYSTEM_PROMPT
  )
  assert (
      template.get_system_prompt("sweagent", use_fn_calling=True)
      == template.SWEAGENT_SYSTEM_PROMPT
  )


def test_get_user_prompt_template():
  """Verify get_user_prompt_template returns correct prompt for all scaffolds."""
  assert (
      template.get_user_prompt_template("r2egym", use_fn_calling=False)
      == template.SWE_USER_PROMPT
  )
  assert (
      template.get_user_prompt_template("r2egym", use_fn_calling=True)
      == template.SWE_USER_PROMPT_FN_CALL
  )
  assert (
      template.get_user_prompt_template("sweagent", use_fn_calling=False)
      == template.SWEAGENT_USER_PROMPT
  )
