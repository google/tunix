# Copyright 2025 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import function_registry
from unittest import mock


# --- Dummy functions for testing ---
def dummy_func_a(x):
  return x + 1


def dummy_func_b(x, y):
  return x * y


class FunctionRegistryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize a registry with default categories for most tests
    self.registry = function_registry.FunctionRegistry()

  def test_default_categories_instance(self):
    self.assertCountEqual(
        self.registry.list_categories(),
        function_registry.FunctionRegistry.DEFAULT_ALLOWED_CATEGORIES,
    )

  def test_custom_categories_instance(self):
    custom_cats = ["cat1", "cat2"]
    # Test-specific instance for custom categories
    registry = function_registry.FunctionRegistry(
        allowed_categories=custom_cats
    )
    self.assertCountEqual(registry.list_categories(), custom_cats)

  def test_empty_categories_instance(self):
    # Test-specific instance for empty categories
    registry = function_registry.FunctionRegistry(allowed_categories=[])
    self.assertLen(registry.list_categories(), 3)

  @parameterized.named_parameters(
      dict(
          testcase_name="loss_fn_a",
          category="policy_loss_fn",
          name="func_a",
          func=dummy_func_a,
      ),
      dict(
          testcase_name="advantage_a",
          category="advantage_estimator",
          name="func_a",
          func=dummy_func_a,
      ),
  )
  def test_register_and_get_success_default(self, category, name, func):
    decorator = self.registry.register(category, name)
    registered_func = decorator(func)
    self.assertIs(registered_func, func)

    retrieved_func = self.registry.get(category, name)
    self.assertIs(retrieved_func, func)
    self.assertEqual(self.registry.list_functions(category), [name])

  def test_register_duplicate_name_fails_default(self):
    self.registry.register("policy_loss_fn", "my_func")(dummy_func_a)
    with self.assertRaisesRegex(
        ValueError,
        "'my_func' is already registered in category 'policy_loss_fn'",
    ):
      self.registry.register("policy_loss_fn", "my_func")(dummy_func_b)

  def test_custom_categories_behavior(self):
    custom_cats = ["custom1", "custom2"]
    # Test-specific instance for custom categories
    registry = function_registry.FunctionRegistry(
        allowed_categories=custom_cats
    )

    # Successful registration and get in custom
    registry.register("custom1", "func_a")(dummy_func_a)
    self.assertIs(registry.get("custom1", "func_a"), dummy_func_a)
    self.assertEqual(registry.list_functions("custom1"), ["func_a"])

    # Default categories should fail
    with self.assertRaisesRegex(
        ValueError, "Invalid category: 'policy_loss_fn'"
    ):
      registry.register("policy_loss_fn", "some_func")(dummy_func_a)

    with self.assertRaisesRegex(ValueError, "Invalid category: 'loss_agg'"):
      registry.register("loss_agg", "some_func")

    with self.assertRaisesRegex(
        LookupError, "No such category: 'advantage_estimator'"
    ):
      registry.list_functions("advantage_estimator")


class GlobalHelpersTest(absltest.TestCase):
  """Tests for the module-level helper functions."""

  def setUp(self):
    super().setUp()
    # ISOLATION: Replace the global 'default_registry' with a fresh instance
    # for every test. This prevents side effects between tests.
    self.original_registry = function_registry.default_registry
    function_registry.default_registry = function_registry.FunctionRegistry()

  def tearDown(self):
    super().tearDown()
    # Restore the original registry
    function_registry.default_registry = self.original_registry

  def test_policy_loss_fn_helpers(self):
    # Test the module-level decorator
    @function_registry.register_policy_loss_fn("global_loss")
    def my_loss():
      return "loss"

    # Test the module-level getter
    retrieved = function_registry.get_policy_loss_fn("global_loss")
    self.assertIs(retrieved, my_loss)

  def test_advantage_estimator_helpers(self):
    @function_registry.register_advantage_estimator("global_adv")
    def my_adv():
      return "adv"

    retrieved = function_registry.get_advantage_estimator("global_adv")
    self.assertIs(retrieved, my_adv)

  def test_reward_manager_registration_helper(self):
    # Note: We test get_reward_manager separately in LazyLoadingTest
    @function_registry.register_reward_manager("global_reward")
    def my_reward():
      return "reward"

    # We can check the registry directly to verify registration worked
    # without triggering the lazy loader yet.
    retrieved = function_registry.default_registry.get(
        function_registry._REWARD_MANAGER_CATEGORY, "global_reward"
    )
    self.assertIs(retrieved, my_reward)


class LazyLoadingTest(absltest.TestCase):
  """Tests specifically for the get_reward_manager lazy import logic."""

  def setUp(self):
    super().setUp()
    # 1. Reset the global lazy-load flag so we can test the trigger again.
    self.original_flag = function_registry._HAVE_REWARDS_LOADED
    function_registry._HAVE_REWARDS_LOADED = False

    # 2. Reset registry to clear previous tests
    self.original_registry = function_registry.default_registry
    function_registry.default_registry = function_registry.FunctionRegistry()

  def tearDown(self):
    super().tearDown()
    function_registry._HAVE_REWARDS_LOADED = self.original_flag
    function_registry.default_registry = self.original_registry

  @mock.patch.object(function_registry.importlib, "import_module")
  def test_get_reward_manager_triggers_import_once(self, mock_import):
    """Verifies that accessing a reward triggers the import exactly once."""

    # Register a dummy reward so the 'get' call succeeds after the import
    @function_registry.register_reward_manager("lazy_reward")
    class DummyReward:
      pass

    # --- Call 1: Should trigger import ---
    result = function_registry.get_reward_manager("lazy_reward")

    self.assertIs(result, DummyReward)
    # Verify we tried to import the specific file
    mock_import.assert_called_once_with("tunix.rl.reward_manager")

    # --- Call 2: Should NOT trigger import ---
    function_registry.get_reward_manager("lazy_reward")

    # Assert import was still only called once (call count didn't increase)
    mock_import.assert_called_once()

  def test_get_reward_manager_sets_flag_true(self):
    """Verifies the global flag is actually updated."""
    # We mock import_module so it doesn't crash trying to find a real file
    with mock.patch.object(function_registry.importlib, "import_module"):
      # Just calling it should flip the flag, even if lookup fails later
      try:
        function_registry.get_reward_manager("missing_reward")
      except LookupError:
        pass  # We expect a lookup error, but we care about the flag

    self.assertTrue(function_registry._HAVE_REWARDS_LOADED)


if __name__ == "__main__":
  absltest.main()
