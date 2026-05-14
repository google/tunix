from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from tunix.generate import param_mapping
from tunix.generate.param_mapping import runtime as param_mapping_runtime


class MockState:

  def __init__(self, params):
    self.params = params

  def flat_state(self):
    return [(tuple(k.split('.')), v) for k, v in self.params.items()]


class MockParam:

  def __init__(self, value):
    self.value = value


class ParamMappingHelpersTest(absltest.TestCase):

  def test_build_explicit_mapping_spec_builds_rule_and_flat_views(self):
    target_param = MockParam(jnp.zeros((2, 2), dtype=jnp.float16))
    unscanned_src_to_tgt_flat = {
        ('encoder.weight', 'decoder.weight'): (
            jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
            target_param,
        )
    }

    mapping_spec, resolved_src_flat, resolved_tgt_flat = (
        param_mapping.build_explicit_mapping_spec(
            unscanned_src_to_tgt_flat,
            key_mapping_hook_fns={'encoder.weight': lambda x: x + 1},
            transpose_keys={'encoder.weight': (1, 0)},
        )
    )

    self.assertEqual(mapping_spec.model_type, 'explicit')
    self.assertLen(mapping_spec.operation_rules, 1)
    rule = mapping_spec.operation_rules[0]
    self.assertEqual(rule.name, 'explicit_mapping')
    self.assertEqual(
        rule.source_patterns,
        (('__mapped__', 'encoder.weight', 'decoder.weight'),),
    )
    self.assertEqual(rule.target_patterns, (('decoder', 'weight'),))
    self.assertEqual(
        [transform.kind for transform in rule.transforms],
        ['transpose', 'hook', 'align_shape', 'cast_to_target'],
    )
    np.testing.assert_array_equal(
        resolved_src_flat[('__mapped__', 'encoder.weight', 'decoder.weight')],
        jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    )
    self.assertEqual(resolved_tgt_flat[('decoder', 'weight')].dtype, jnp.float16)

  def test_materialize_transfer_step_applies_transpose_and_cast(self):
    step = param_mapping.PlannedTransfer(
        rule_name='transpose_cast',
        source_keys=(('src',),),
        target_key=('dst',),
        transforms=(
            param_mapping.Transform.transpose((1, 0)),
            param_mapping.Transform.cast_to_target(),
        ),
    )
    src_flat = {('src',): jnp.array([[1.0, 2.0]], dtype=jnp.float32)}
    tgt_val = jnp.zeros((2, 1), dtype=jnp.float16)

    result = param_mapping.materialize_transfer_step(
        step,
        src_flat,
        tgt_val,
        scan_axis=0,
        value_cache={},
    )

    self.assertEqual(result.dtype, jnp.float16)
    np.testing.assert_array_equal(
        result,
        jnp.array([[1.0], [2.0]], dtype=jnp.float16),
    )

  def test_make_structural_mapping_spec_wraps_default_rules(self):
    spec = param_mapping.make_structural_mapping_spec()

    self.assertEqual(spec.model_type, 'structural')
    self.assertEqual(
        spec.operation_rules,
        param_mapping.default_structural_operation_rules(),
    )

  def test_resolve_transpose_axes_prefers_exact_key_and_supports_lora_regex(self):
    self.assertEqual(
        param_mapping_runtime._resolve_transpose_axes(
            'decoder.weight',
            {'weight': (1, 0), 'decoder.weight': (0, 1)},
            rollout_engine=None,
        ),
        (1, 0),
    )
    self.assertEqual(
        param_mapping_runtime._resolve_transpose_axes(
            'decoder.adapter.0.lora_a',
            {r'decoder\.adapter\..*\.lora_a': (0, 1)},
            rollout_engine='sglang_jax',
        ),
        (0, 1),
    )

  def test_build_flat_dict_groups_layer_targets_by_source_key(self):
    flat_state = [
        (('decoder', 'layers', '0', 'weight'), MockParam(jnp.array([1.0]))),
        (('decoder', 'layers', '1', 'weight'), MockParam(jnp.array([2.0]))),
    ]

    result = param_mapping.build_flat_dict(
        flat_state,
        {'encoder.weight': ('decoder.layers.*.weight', ('layer', None))},
    )

    self.assertIn('encoder.weight', result)
    grouped_values, grouped_paths, grouped_sharding = result['encoder.weight']
    self.assertEqual(
        grouped_paths,
        ['decoder.layers.0.weight', 'decoder.layers.1.weight'],
    )
    self.assertEqual(grouped_sharding, ('layer', None))
    self.assertLen(grouped_values, 2)

  def test_unroll_scanned_layers_slices_along_layer_axis(self):
    src_state = MockState(
        {
            'encoder.weight': MockParam(
                jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
            )
        }
    )
    src_to_tgt_map = {
        'encoder.weight': (
            [
                MockParam(jnp.zeros((2,), dtype=jnp.float32)),
                MockParam(jnp.zeros((2,), dtype=jnp.float32)),
            ],
            ['decoder.layers.0.weight', 'decoder.layers.1.weight'],
            ('layer', None),
        )
    }

    result = param_mapping.unroll_scanned_layers(src_state, src_to_tgt_map)

    np.testing.assert_array_equal(
        result[('encoder.weight', 'decoder.layers.0.weight')][0],
        jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result[('encoder.weight', 'decoder.layers.1.weight')][0],
        jnp.array([3.0, 4.0], dtype=jnp.float32),
    )

  def test_sync_tied_lm_head_if_needed_copies_embedding_when_needed(self):
    embedding = MockParam(jnp.array([1.0, 2.0], dtype=jnp.float32))
    lm_head = MockParam(jnp.zeros((2,), dtype=jnp.float32))
    tgt_flat_list = [
        (('token_embed', 'embedding'), embedding),
        (('output', 'lm_head'), lm_head),
    ]

    param_mapping.sync_tied_lm_head_if_needed(
        tgt_flat_list, transferred_target_keys=set()
    )

    np.testing.assert_array_equal(lm_head.value, embedding.value)

  def test_reshard_in_chunks_calls_reshard_fn_per_chunk(self):
    calls = []
    src_flat = {
        ('decoder', 'weight'): jnp.array([1.0, 2.0], dtype=jnp.float32),
        ('decoder', 'bias'): jnp.array([3.0, 4.0], dtype=jnp.float32),
    }
    spec_flat = {
        ('decoder', 'weight'): jnp.zeros((2,), dtype=jnp.float32),
        ('decoder', 'bias'): jnp.zeros((2,), dtype=jnp.float32),
    }

    def reshard_fn(source, target):
      del target
      calls.append(tuple(sorted(source['decoder'].keys())))
      return source

    result = param_mapping.reshard_in_chunks(
        dict(src_flat),
        spec_flat,
        reshard_fn,
        chunk_size=1,
    )

    self.assertLen(calls, 2)
    self.assertIn(('bias',), calls)
    self.assertIn(('weight',), calls)
    np.testing.assert_array_equal(
        result[('decoder', 'weight')],
        jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result[('decoder', 'bias')],
        jnp.array([3.0, 4.0], dtype=jnp.float32),
    )


if __name__ == '__main__':
  absltest.main()
