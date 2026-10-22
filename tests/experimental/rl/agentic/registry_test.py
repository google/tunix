# Copyright 2026 Google LLC
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

"""Unit tests for registry.py in experimental/rl/agentic."""

import os
import tempfile
from absl.testing import absltest
from tunix.experimental.rl.agentic import registry


class RegistryTest(absltest.TestCase):

  def test_register_with_custom_name(self):
    test_registry = registry.Registry("TestRegistry")

    @test_registry.register("custom_key")
    class DummyClass:
      pass

    self.assertIn("custom_key", test_registry)
    self.assertEqual(test_registry.get("custom_key"), DummyClass)

  def test_register_with_default_name(self):
    test_registry = registry.Registry("TestRegistry")

    @test_registry.register()
    class AutoNamedClass:
      pass

    self.assertIn("AutoNamedClass", test_registry)
    self.assertEqual(test_registry.get("AutoNamedClass"), AutoNamedClass)

  def test_duplicate_registration_raises(self):
    test_registry = registry.Registry("TestRegistry")

    @test_registry.register("dup_key")
    class ClassA:
      pass

    with self.assertRaisesRegex(KeyError, "already registered"):

      @test_registry.register("dup_key")
      class ClassB:
        pass

  def test_get_unregistered_key_raises(self):
    test_registry = registry.Registry("TestRegistry")

    with self.assertRaisesRegex(KeyError, "is not registered"):
      test_registry.get("nonexistent_key")

  def test_global_decorator_aliases(self):
    @registry.register_agent("test_agent")
    class TestAgent:
      pass

    @registry.register_env("test_env")
    class TestEnv:
      pass

    self.assertEqual(registry.AGENT_REGISTRY.get("test_agent"), TestAgent)
    self.assertEqual(registry.ENV_REGISTRY.get("test_env"), TestEnv)

  def test_auto_discover_directory_path(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "my_custom_env.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic import registry\n"
            "@registry.register_env('dir_discovered_env')\n"
            "class DirDiscoveredEnv:\n"
            "  pass\n"
        )
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("dir_discovered_env", registry.ENV_REGISTRY)

  def test_auto_discover_skips_unregistered_modules(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      decorated_file = os.path.join(tmp_dir, "decorated.py")
      with open(decorated_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic import registry\n"
            "@registry.register_agent('ast_agent')\n"
            "class ASTAgent:\n"
            "  pass\n"
        )
      undecorated_file = os.path.join(tmp_dir, "undecorated.py")
      with open(undecorated_file, "w") as f:
        f.write(
            "raise RuntimeError('Unregistered module should not be imported!')\n"
        )

      registry.auto_discover_modules(tmp_dir)
      self.assertIn("ast_agent", registry.AGENT_REGISTRY)

  def test_auto_discover_package_name(self):
    registry.auto_discover_modules("tunix.experimental.rl.agentic")
    self.assertIsNotNone(registry.AGENT_REGISTRY)
    self.assertIsNotNone(registry.ENV_REGISTRY)

  def test_auto_discover_nested_directory_and_relative_imports(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      sub_dir = os.path.join(tmp_dir, "subpkg")
      os.makedirs(sub_dir)

      with open(os.path.join(sub_dir, "helper.py"), "w") as f:
        f.write("BASE_VALUE = 42\n")

      with open(os.path.join(sub_dir, "rel_env.py"), "w") as f:
        f.write(
            "from . import helper\n"
            "from tunix.experimental.rl.agentic import registry\n"
            "@registry.register_env('rel_imported_env')\n"
            "class RelImportedEnv:\n"
            "  val = helper.BASE_VALUE\n"
        )

      registry.auto_discover_modules(tmp_dir)
      self.assertIn("rel_imported_env", registry.ENV_REGISTRY)
      env_cls = registry.ENV_REGISTRY.get("rel_imported_env")
      self.assertEqual(env_cls.val, 42)

  def test_registry_contains_and_keys(self):
    test_reg = registry.Registry("Custom")

    @test_reg.register("item1")
    class Item1:
      pass

    self.assertTrue(test_reg.contains("item1"))
    self.assertIn("item1", test_reg)
    self.assertFalse(test_reg.contains("item2"))
    self.assertEqual(test_reg.keys(), ["item1"])

  def test_has_registry_decorator_with_aliased_import(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "aliased_agent.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import register_agent"
            " as ra\n"
            "@ra('aliased_agent')\n"
            "class AliasedAgent:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("aliased_agent", registry.AGENT_REGISTRY)

  def test_has_registry_decorator_with_aliased_registry_instance(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "aliased_inst.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import AGENT_REGISTRY"
            " as ar\n"
            "@ar.register('aliased_inst_agent')\n"
            "class AliasedInstAgent:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("aliased_inst_agent", registry.AGENT_REGISTRY)

  def test_has_registry_decorator_with_local_variable_alias(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "local_alias.py")
      with open(py_file, "w") as f:
        f.write(
            "import tunix.experimental.rl.agentic.registry as reg\n"
            "my_dec = reg.register_agent\n"
            "@my_dec('local_alias_agent')\n"
            "class LocalAliasAgent:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("local_alias_agent", registry.AGENT_REGISTRY)

  def test_register_as_bare_decorator(self):
    test_registry = registry.Registry("TestRegistry")

    @test_registry.register
    class BareClass:
      pass

    self.assertIn("BareClass", test_registry)
    self.assertEqual(test_registry.get("BareClass"), BareClass)

    @registry.register_agent
    class GlobalBareAgent:
      pass

    self.assertIn("GlobalBareAgent", registry.AGENT_REGISTRY)
    self.assertEqual(
        registry.AGENT_REGISTRY.get("GlobalBareAgent"), GlobalBareAgent
    )

    @registry.register_env
    class GlobalBareEnv:
      pass

    self.assertIn("GlobalBareEnv", registry.ENV_REGISTRY)
    self.assertEqual(registry.ENV_REGISTRY.get("GlobalBareEnv"), GlobalBareEnv)

  def test_has_registry_decorator_with_bare_alias(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "bare_alias.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import register_agent"
            " as ra\n"
            "@ra\n"
            "class BareAliasedAgent:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("BareAliasedAgent", registry.AGENT_REGISTRY)

  def test_has_registry_decorator_stacked_multiple_decorators(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "stacked.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import register_agent"
            " as ra\n"
            "def dec1(cls): return cls\n"
            "def dec2(cls): return cls\n"
            "@dec1\n"
            "@ra('stacked_agent')\n"
            "@dec2\n"
            "class StackedAgent:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))
      registry.auto_discover_modules(tmp_dir)
      self.assertIn("stacked_agent", registry.AGENT_REGISTRY)

  def test_has_registry_decorator_ignores_function_and_method_decorators(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "funcs_only.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import register_agent"
            " as ra\n"
            "@ra('some_func')\n"
            "def some_function():\n"
            "  pass\n\n"
            "class UndecoratedClass:\n"
            "  @property\n"
            "  def prop(self):\n"
            "    return 1\n"
        )
      self.assertFalse(registry.has_registry_decorator(py_file))

  def test_has_registry_decorator_custom_registry_instance(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "custom_reg_file.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic import registry\n"
            "custom_reg = registry.Registry('CustomReg')\n"
            "@custom_reg.register('custom_registered_class')\n"
            "class CustomRegisteredClass:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))

  def test_has_registry_decorator_chained_module_path(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "chained.py")
      with open(py_file, "w") as f:
        f.write(
            "import tunix.experimental.rl.agentic.registry\n"
            "@tunix.experimental.rl.agentic.registry.register_env('chained_env')\n"
            "class ChainedEnv:\n"
            "  pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))

  def test_has_registry_decorator_multiple_classes_one_registered(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "multi_classes.py")
      with open(py_file, "w") as f:
        f.write(
            "from tunix.experimental.rl.agentic.registry import register_env\n"
            "class HelperA: pass\n"
            "@register_env('multi_env')\n"
            "class RegisteredB: pass\n"
            "class HelperC: pass\n"
        )
      self.assertTrue(registry.has_registry_decorator(py_file))

  def test_has_registry_decorator_handles_syntax_errors_gracefully(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "broken_syntax.py")
      with open(py_file, "w") as f:
        f.write("def broken( ::: invalid python syntax\n")
      self.assertFalse(registry.has_registry_decorator(py_file))

  def test_has_registry_decorator_handles_nonexistent_file(self):
    self.assertFalse(
        registry.has_registry_decorator("/nonexistent/path/to/file.py")
    )

  def test_has_registry_decorator_ignores_unrelated_registry_modules(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      py_file = os.path.join(tmp_dir, "unrelated_reg.py")
      with open(py_file, "w") as f:
        f.write(
            "import other_package.docker.registry as reg\n"
            "from some_lib.registry import SchemaValidator\n\n"
            "@reg.validate_container\n"
            "class ContainerConfig:\n"
            "  pass\n\n"
            "@SchemaValidator\n"
            "class ValidatedSchema:\n"
            "  pass\n"
        )
      self.assertFalse(registry.has_registry_decorator(py_file))


if __name__ == "__main__":
  absltest.main()


