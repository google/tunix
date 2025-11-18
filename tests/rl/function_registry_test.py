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

# from my_module import function_registry # Or include class def above


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
    self.assertLen(registry.list_categories(), 2)

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


if __name__ == "__main__":
  absltest.main()
