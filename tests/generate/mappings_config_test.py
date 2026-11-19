"""Tests for tunix.generate.mappings.MappingConfig utilities."""

from absl.testing import absltest

from tunix.generate import mappings
from tunix.models.llama3 import model as model_lib
from flax import nnx

class MappingConfigTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    rng = nnx.Rngs(params=0)
    cls.model = model_lib.Llama3(model_lib.ModelConfig.llama3p2_1b(), rngs=rng)

  def test_from_object_with_none_errors_out(self):
    with self.assertRaisesRegex(
        AssertionError,
        "Either mapping_obj or model must be provided"
    ):
      cfg = mappings.MappingConfig.build()

  def test_from_object_with_mapping_config_instance(self):
    original = mappings.MappingConfig(
        to_hf_mappings={'foo': ('bar', ('tp',))},
        lora_to_hf_mappings={'lora': ('baz', ('fsdp',))},
        to_hf_hook_fns={'foo': lambda x: x},
        to_hf_transpose_keys={'foo': (1, 0)},
    )
    cfg = mappings.MappingConfig.build(original)
    self.assertIs(cfg, original)

  def test_from_object_with_dict_and_callables(self):
    mapping_obj = {
        'to_hf_mappings': lambda: {'a': ('b', ('tp',))},
        'lora_to_hf_mappings': lambda: {'c': ('d', ('fsdp',))},
        'to_hf_hook_fns': lambda: {'a': lambda x: x},
        'to_hf_transpose_keys': lambda: {'embedding': (1, 0)},
    }
    cfg = mappings.MappingConfig.build(mapping_obj)
    self.assertEqual(cfg.to_hf_mappings, {'a': ('b', ('tp',))})
    self.assertEqual(cfg.lora_to_hf_mappings, {'c': ('d', ('fsdp',))})
    self.assertEqual(
        cfg.to_hf_transpose_keys,
        {'embedding': (1, 0)},
    )
    self.assertIn('a', cfg.to_hf_hook_fns)

  def test_build_mapping_config_with_model_only(self):
    cfg = mappings.MappingConfig.build(self.model)
    self.assertTrue(
        cfg.to_hf_mappings['embedder.input_embedding'],
        (
          'model.embed.embedding',
          ('model', None),
        )
    )

    self.assertTrue(
        cfg.lora_to_hf_mappings['layers.*.mlp.gate_proj.kernel_lora_a'],
        (
          'model.layers.*.mlp.gate_proj.kernel_lora_a',
          (None, None),
        )
    )

    self.assertEqual(cfg.to_hf_transpose_keys, {'embedding': (1, 0)},)

  def test_build_mapping_config_with_overrides(self):
    override = {'embedder.input_embedding': (
          'fake.path.embedding',
          ('fake_dim', None),
      )}
    cfg = mappings.MappingConfig.build(
        {"to_hf_mappings":override,
        "to_hf_hook_fns":{'override': lambda x: x},
        }
    )
    self.assertEqual(cfg.to_hf_mappings, override)
    self.assertIn('override', cfg.to_hf_hook_fns)
    self.assertTrue(callable(cfg.to_hf_hook_fns['override']))


if __name__ == '__main__':
  absltest.main()
