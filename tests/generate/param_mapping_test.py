from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
import numpy as np

from tunix.generate import param_mapping
from tunix.generate.param_mapping import runtime as param_mapping_runtime


class MockState:

    def __init__(self, params):
        self.params = params

    def flat_state(self):
        return [(tuple(k.split('.')), v) for k, v in self.params.items()]

    def from_flat_path(self, flat_path):
        new_params = {}
        for keys, param in flat_path:
            new_params['.'.join(keys)] = param.value
        return MockState(new_params)


class MockParam:

    def __init__(self, value):
        self.value = value


class ParamMappingTest(parameterized.TestCase):

  def test_public_api_exports_are_available(self):
    for public_name in param_mapping.__all__:
      self.assertTrue(hasattr(param_mapping, public_name), public_name)

  def test_prepared_transfer_defaults(self):
    prepared = param_mapping.PreparedTransfer(
        src_flat={('src',): jnp.array(1.0)},
        tgt_flat={('tgt',): jnp.array(0.0)},
    )

    self.assertEqual(prepared.scan_axis, 1)
    self.assertIsNone(prepared.transform_context)
    self.assertIsNone(prepared.operation_rules)

  def test_direct_transfer_converter_prepare_transfer(self):
    src_state = nnx.Dict(base=nnx.Dict(weight=nnx.Param(jnp.array([1.0, 2.0]))))
    dst_state = nnx.Dict(model=nnx.Dict(weight=nnx.Param(jnp.zeros((2,)))))

    converter = param_mapping.DirectTransferConverter(scan_axis=0)
    prepared = converter.prepare_transfer(src_state, dst_state)

    self.assertIsInstance(prepared, param_mapping.PreparedTransfer)
    self.assertEqual(prepared.scan_axis, 0)
    self.assertIn(("weight",), prepared.src_flat)
    self.assertIn(("weight",), prepared.tgt_flat)

  def test_mapping_program_projects_to_mapping_spec(self):
    operation_rule = param_mapping.OperationRule(
        name='explicit_q_proj',
        source_patterns=(('decoder', 'q_proj', 'weight'),),
        target_patterns=(('model', 'q_proj', 'weight'),),
        transforms=(param_mapping.Transform.copy(),),
    )

    program = param_mapping.MappingProgram(
        model_type='test_model',
        operation_rules=(operation_rule,),
    )

    spec = program.to_mapping_spec()

    self.assertEqual(spec.model_type, 'test_model')
    self.assertEqual(spec.operation_rules, (operation_rule,))
    self.assertEqual(program.operation_rules, (operation_rule,))

  def test_mapping_program_works_with_existing_planner(self):
    explicit_rule = param_mapping.OperationRule(
        name='explicit_override',
        source_patterns=(('decoder', 'custom_layer', 'weight'),),
        target_patterns=(('decoder', 'layers_1', 'mlp', 'weight'),),
        transforms=(
            param_mapping.Transform.align_shape(),
            param_mapping.Transform.cast_to_target(),
    ),
    )
    program = param_mapping.MappingProgram(
        model_type='test_model',
        operation_rules=(
            explicit_rule,
            *param_mapping.default_structural_operation_rules(),
    ),
    )
    src_flat = {
        ('decoder', 'layers', 'mlp', 'weight'): jnp.array([[1.0], [2.0]]),
        ('decoder', 'custom_layer', 'weight'): jnp.array([9.0], dtype=jnp.float32),
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'weight'): jnp.zeros((1,), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=program.to_mapping_spec().operation_rules,
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'explicit_override')
    self.assertEqual(plan[0].source_keys, (('decoder', 'custom_layer', 'weight'),))

  def test_operation_rule_represents_merge_and_split_concepts(self):
    merge_rule = param_mapping.OperationRule(
        name='merge_expert_gate_up_proj',
        source_patterns=(
            'mlp.experts.*.gate_proj.weight',
            'mlp.experts.*.up_proj.weight',
    ),
        target_patterns=('mlp.experts.gate_up_proj',),
        transforms=(
            param_mapping.Transform.merge_modulelist(dim=0),
            param_mapping.Transform.concatenate(dim=1),
    ),
    )
    split_rule = param_mapping.OperationRule(
        name='split_qkv',
        source_patterns=('attn.Wqkv',),
        target_patterns=(
            'self_attn.q_proj',
            'self_attn.k_proj',
            'self_attn.v_proj',
    ),
        transforms=(param_mapping.Transform.chunk(dim=0, chunks=3),),
    )
    self.assertEqual(split_rule.source_resolvers, ('pattern',))

    self.assertEqual(merge_rule.source_patterns[0], 'mlp.experts.*.gate_proj.weight')
    self.assertEqual(merge_rule.transforms[0].kind, 'merge_modulelist')
    self.assertEqual(merge_rule.transforms[1].args['dim'], 1)
    self.assertEqual(split_rule.target_patterns[2], 'self_attn.v_proj')
    self.assertEqual(split_rule.transforms[0].kind, 'chunk')
    self.assertEqual(split_rule.transforms[0].args['chunks'], 3)

  def test_build_transfer_plan_operation_rule_split(self):
    src_flat = {('attn', 'Wqkv'): jnp.arange(12.0, dtype=jnp.float32).reshape(6, 2)}
    tgt_flat = {
        ('self_attn', 'q_proj'): jnp.zeros((2, 2), dtype=jnp.float32),
        ('self_attn', 'k_proj'): jnp.zeros((2, 2), dtype=jnp.float32),
        ('self_attn', 'v_proj'): jnp.zeros((2, 2), dtype=jnp.float32),
    }
    operation_rules = (
        param_mapping.OperationRule(
            name='split_qkv',
            source_patterns=('attn.Wqkv',),
            target_patterns=(
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
            ),
            transforms=(param_mapping.Transform.chunk(dim=0, chunks=3),),
    ),
    )

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=operation_rules,
    )
    filtered_src_flat, _ = param_mapping.execute_transfer_plan(
        plan, src_flat, tgt_flat, scan_axis=0
    )

    self.assertLen(plan, 3)
    np.testing.assert_array_equal(
        filtered_src_flat[('self_attn', 'q_proj')],
        jnp.array([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        filtered_src_flat[('self_attn', 'k_proj')],
        jnp.array([[4.0, 5.0], [6.0, 7.0]], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        filtered_src_flat[('self_attn', 'v_proj')],
        jnp.array([[8.0, 9.0], [10.0, 11.0]], dtype=jnp.float32),
    )

  def test_build_transfer_plan_operation_rule_concatenate(self):
    src_flat = {
        ('mlp', 'gate_proj'): jnp.array([[1.0, 2.0]], dtype=jnp.float32),
        ('mlp', 'up_proj'): jnp.array([[3.0, 4.0]], dtype=jnp.float32),
    }
    tgt_flat = {('mlp', 'gate_up_proj'): jnp.zeros((1, 4), dtype=jnp.float32)}
    operation_rules = (
        param_mapping.OperationRule(
            name='merge_gate_up_proj',
            source_patterns=('mlp.gate_proj', 'mlp.up_proj'),
            target_patterns=('mlp.gate_up_proj',),
            transforms=(param_mapping.Transform.concatenate(dim=1),),
    ),
    )

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=operation_rules,
    )
    filtered_src_flat, _ = param_mapping.execute_transfer_plan(
        plan, src_flat, tgt_flat, scan_axis=0
    )

    self.assertLen(plan, 1)
    np.testing.assert_array_equal(
        filtered_src_flat[('mlp', 'gate_up_proj')],
        jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32),
    )

  def test_prepare_structural_transfer_flattens_and_fuses_inputs(self):
    src_state = nnx.Dict(
        base=nnx.Dict(
            layers=nnx.Dict(
                wi_0=nnx.Param(jnp.array([[1.0, 2.0]], dtype=jnp.float32)),
                wi_1=nnx.Param(jnp.array([[3.0, 4.0]], dtype=jnp.float32)),
            )
    )
    )
    dst_state = nnx.Dict(
        model=nnx.Dict(
            layers=nnx.Dict(
                wi=nnx.Param(jnp.zeros((1, 4), dtype=jnp.float32))
            )
    )
    )

    prepared = param_mapping.prepare_structural_transfer(src_state, dst_state)

    self.assertIn(('layers', 'wi'), prepared.src_flat)
    self.assertIn(('layers', 'wi'), prepared.tgt_flat)
    self.assertNotIn(('layers', 'wi_0'), prepared.src_flat)
    self.assertNotIn(('layers', 'wi_1'), prepared.src_flat)

  def test_transfer_state_with_converter_uses_shared_engine(self):
    class StructuralConverter(param_mapping.BaseConverter):

      def prepare_transfer(self, src_state, dst_state, **kwargs):
        return param_mapping.prepare_structural_transfer(
            src_state,
            dst_state,
            scan_axis=kwargs.get('scan_axis', 1),
        )

      def reshard_transfer(
          self,
          filtered_src_flat,
          filtered_tgt_flat,
          dst_state,
          reshard_fn,
          delete_dst_buffers=False,
          reshard_chunk_size=None,
          **kwargs,
      ):
        return param_mapping.reshard_transfer(
            filtered_src_flat=filtered_src_flat,
            filtered_tgt_flat=filtered_tgt_flat,
            dst_state=dst_state,
            reshard_fn=reshard_fn,
            delete_dst_buffers=delete_dst_buffers,
            reshard_chunk_size=reshard_chunk_size,
            **kwargs,
        )

      def finalize_transfer(self, dst_state, resharded_weights, **kwargs):
        del kwargs
        nnx.update(dst_state, resharded_weights)
        return dst_state

    src_state = nnx.Dict(
        base=nnx.Dict(
            decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array(1.0))))
    )
    )
    dst_state = nnx.Dict(
        model=nnx.Dict(
            decoder=nnx.Dict(layer0=nnx.Dict(weight=nnx.Param(jnp.array(0.0))))
    )
    )

    result = param_mapping.transfer_state_with_converter(
        src_state,
        dst_state['model'],
        StructuralConverter(),
        reshard_fn=lambda source, target: source,
    )

    self.assertIs(result, dst_state['model'])
    np.testing.assert_array_equal(
        dst_state['model']['decoder']['layer0']['weight'][...],
        jnp.array(1.0),
    )

  def test_transfer_state_with_converter_allows_custom_reshard_hook(self):
    calls = []

    class CustomReshardConverter(param_mapping.BaseConverter):

      def prepare_transfer(self, src_state, dst_state, **kwargs):
        return param_mapping.prepare_structural_transfer(src_state, dst_state)

      def reshard_transfer(
          self,
          filtered_src_flat,
          filtered_tgt_flat,
          dst_state,
          reshard_fn,
          delete_dst_buffers=False,
          reshard_chunk_size=None,
          **kwargs,
      ):
        calls.append((tuple(filtered_src_flat.keys()), tuple(filtered_tgt_flat.keys())))
        return param_mapping.reshard_transfer(
            filtered_src_flat=filtered_src_flat,
            filtered_tgt_flat=filtered_tgt_flat,
            dst_state=dst_state,
            reshard_fn=reshard_fn,
            delete_dst_buffers=delete_dst_buffers,
            reshard_chunk_size=reshard_chunk_size,
            **kwargs,
        )

      def finalize_transfer(self, dst_state, resharded_weights, **kwargs):
        del kwargs
        nnx.update(dst_state, resharded_weights)
        return dst_state

    src_state = nnx.Dict(decoder=nnx.Dict(weight=nnx.Param(jnp.array([1.0, 2.0]))))
    dst_state = nnx.Dict(decoder=nnx.Dict(weight=nnx.Param(jnp.array([0.0, 0.0]))))

    param_mapping.transfer_state_with_converter(
        src_state,
        dst_state,
        CustomReshardConverter(),
        reshard_fn=lambda source, target: source,
    )

    self.assertLen(calls, 1)
    self.assertEqual(calls[0][0], (('decoder', 'weight'),))
    self.assertEqual(calls[0][1], (('decoder', 'weight'),))
    np.testing.assert_array_equal(
        dst_state['decoder']['weight'][...],
        jnp.array([1.0, 2.0]),
    )

  def test_reshard_transfer_returns_unflattened_update_tree(self):
    filtered_src_flat = {('decoder', 'weight'): jnp.array([1.0, 2.0])}
    filtered_tgt_flat = {('decoder', 'weight'): jnp.zeros((2,), dtype=jnp.float32)}

    result = param_mapping.reshard_transfer(
        filtered_src_flat=filtered_src_flat,
        filtered_tgt_flat=filtered_tgt_flat,
        dst_state=None,
        reshard_fn=lambda source, target: source,
    )

    self.assertIn('decoder', result)
    np.testing.assert_array_equal(result['decoder']['weight'], jnp.array([1.0, 2.0]))

  def test_transfer_state_directly_public_api(self):
    src_state = nnx.Dict(base=nnx.Dict(weight=nnx.Param(jnp.array([1.0, 2.0]))))
    dst_state = nnx.Dict(model=nnx.Dict(weight=nnx.Param(jnp.zeros((2,)))))

    result = param_mapping.transfer_state_directly(
        src_state,
        dst_state,
        reshard_fn=lambda source, target: source,
        scan_axis=0,
    )

    self.assertIs(result, None)
    np.testing.assert_array_equal(
        dst_state['model']['weight'][...],
        jnp.array([1.0, 2.0]),
    )

  def test_transfer_state_with_mappings_public_api(self):
    src_state = MockState({
        'encoder.weight': MockParam(jnp.array([1.0, 2.0], dtype=jnp.float32)),
    })
    dst_state = MockState({
        'decoder.weight': MockParam(jnp.zeros(2, dtype=jnp.float32)),
        'decoder.bias': MockParam(jnp.array([-1.0, -1.0], dtype=jnp.float32)),
    })

    result = param_mapping.transfer_state_with_mappings(
        src_state,
        dst_state,
        {'encoder.weight': ('decoder.weight', None)},
    )

    self.assertIsInstance(result, MockState)
    np.testing.assert_array_equal(
        result.params['decoder.weight'],
        jnp.array([1.0, 2.0], dtype=jnp.float32),
    )
    np.testing.assert_array_equal(
        result.params['decoder.bias'],
        jnp.array([-1.0, -1.0], dtype=jnp.float32),
    )

  def test_fuse_moe_weights_skips_scanned_keys_without_error(self):
    src_flat = {
        ('decoder', 'layers', 'mlp', 'wi_0'): jnp.ones((2, 3, 2), dtype=jnp.float32),
        ('decoder', 'layers', 'mlp', 'wi_1'): jnp.full((2, 3, 2), 2.0, dtype=jnp.float32),
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'wi'): jnp.zeros((2, 4), dtype=jnp.float32),
    }

    fused = param_mapping.fuse_moe_weights(src_flat, tgt_flat)

    self.assertIn(('decoder', 'layers', 'mlp', 'wi_0'), fused)
    self.assertIn(('decoder', 'layers', 'mlp', 'wi_1'), fused)
    self.assertNotIn(('decoder', 'layers_1', 'mlp', 'wi'), fused)

  def test_transfer_planning_helpers_preserve_structural_rule_order(self):
    rules = param_mapping.default_structural_operation_rules()

    self.assertEqual(
        [rule.name for rule in rules],
        ['direct_match', 'scanned_layers', 'scanned_fused_moe'],
    )
    self.assertTrue(
        param_mapping.resolve_operation_target_key(
            ('decoder', 'layers_7', 'mlp', 'weight'),
            {('decoder', 'layers', 'mlp', 'weight'): jnp.array([1.0])},
            (rules[1],),
    )
        is not None
    )
    self.assertFalse(
        param_mapping.resolve_operation_target_key(
            ('decoder', 'layers_7', 'mlp'),
            {('decoder', 'layers', 'mlp', 'wi_0'): jnp.array([1.0])},
            (rules[2],),
    )
        is not None
    )

  def test_transfer_planning_helpers_extract_and_rewrite_layer_keys(self):
    target_key = ('decoder', 'layers_7', 'mlp', 'weight')

    self.assertEqual(param_mapping.extract_layer_index(target_key), 7)
    self.assertEqual(
        param_mapping.replace_layers_n_with_layers(target_key),
        ('decoder', 'layers', 'mlp', 'weight'),
    )
    self.assertEqual(
        param_mapping.remove_layers_n_component(target_key),
        ('decoder', 'mlp', 'weight'),
    )

  def test_transfer_planning_helpers_derive_fused_moe_source_keys(self):
    target_key = ('decoder', 'layers_3', 'mlp', 'wi')

    wi_0_key, wi_1_key = param_mapping.derive_moe_source_keys(target_key)

    self.assertEqual(wi_0_key, ('decoder', 'layers_3', 'mlp', 'wi_0'))
    self.assertEqual(wi_1_key, ('decoder', 'layers_3', 'mlp', 'wi_1'))

  def test_transfer_planning_helpers_reject_non_wi_moe_derivation(self):
    with self.assertRaises(param_mapping.MappingError):
      param_mapping.derive_moe_source_keys(('decoder', 'layers_3', 'mlp', 'wo'))

  def test_build_transfer_plan_direct_match(self):
    src_flat = {('decoder', 'layer0', 'weight'): jnp.array([1.0, 2.0])}
    tgt_flat = {
        ('decoder', 'layer0', 'weight'): jnp.zeros((2,), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'direct_match')
    self.assertEqual(plan[0].source_keys, (('decoder', 'layer0', 'weight'),))
    self.assertEqual(plan[0].target_key, ('decoder', 'layer0', 'weight'))

  def test_build_transfer_plan_scanned_layer_match(self):
    src_flat = {
        ('decoder', 'layers', 'mlp', 'weight'): jnp.array([[1.0], [2.0]])
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'weight'): jnp.zeros((1,), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'scanned_layers')
    self.assertEqual(
        plan[0].source_keys,
        (('decoder', 'layers', 'mlp', 'weight'),),
    )
    self.assertEqual(plan[0].target_key, ('decoder', 'layers_1', 'mlp', 'weight'))

  def test_build_transfer_plan_implicit_layers_match(self):
    src_flat = {('layers', 'mlp', 'weight'): jnp.array([1.0, 2.0])}
    tgt_flat = {
        ('layers', 'layers_1', 'mlp', 'weight'): jnp.zeros((), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'scanned_layers')
    self.assertEqual(plan[0].source_keys, (('layers', 'mlp', 'weight'),))
    self.assertEqual(
        plan[0].target_key,
        ('layers', 'layers_1', 'mlp', 'weight'),
    )

  def test_build_transfer_plan_scanned_fused_moe_match(self):
    src_flat = {
        ('decoder', 'layers', 'mlp', 'wi_0'): jnp.ones((2, 3, 2), dtype=jnp.float32),
        ('decoder', 'layers', 'mlp', 'wi_1'): jnp.ones((2, 3, 2), dtype=jnp.float32),
    }
    tgt_flat = {
        ('decoder', 'layers_2', 'mlp', 'wi'): jnp.zeros((2, 4), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'scanned_fused_moe')
    self.assertEqual(
        plan[0].source_keys,
        (
            ('decoder', 'layers', 'mlp', 'wi_0'),
            ('decoder', 'layers', 'mlp', 'wi_1'),
    ),
    )

  def test_build_transfer_plan_ordered_rule_wins_first(self):
    src_flat = {
        ('decoder', 'layers', 'mlp', 'weight'): jnp.array([[1.0], [2.0]]),
        ('decoder', 'custom_layer', 'weight'): jnp.array([9.0], dtype=jnp.float32),
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'weight'): jnp.zeros((1,), dtype=jnp.float32)
    }
    operation_rules = (
        param_mapping.OperationRule(
            name='explicit_override',
            source_patterns=(('decoder', 'custom_layer', 'weight'),),
            target_patterns=(('decoder', 'layers_1', 'mlp', 'weight'),),
            transforms=(
                param_mapping.Transform.align_shape(),
                param_mapping.Transform.cast_to_target(),
            ),
    ),
        *param_mapping.default_structural_operation_rules(),
    )

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=operation_rules,
    )

    self.assertLen(plan, 1)
    self.assertEqual(plan[0].rule_name, 'explicit_override')
    self.assertEqual(
        plan[0].source_keys,
        (('decoder', 'custom_layer', 'weight'),),
    )
    self.assertEqual(
        [transform.kind for transform in plan[0].transforms],
        ['align_shape', 'cast_to_target'],
    )

  def test_execute_transfer_plan_direct_match(self):
    src_flat = {('decoder', 'layer0', 'weight'): jnp.array([1.0, 2.0])}
    tgt_flat = {
        ('decoder', 'layer0', 'weight'): jnp.zeros((2,), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )
    filtered_src_flat, filtered_tgt_flat = param_mapping.execute_transfer_plan(
        plan, src_flat, tgt_flat, scan_axis=0
    )

    np.testing.assert_array_equal(
        filtered_src_flat[('decoder', 'layer0', 'weight')],
        jnp.array([1.0, 2.0]),
    )
    self.assertEqual(filtered_tgt_flat, tgt_flat)

  def test_execute_transfer_plan_scanned_layer_match(self):
    src_flat = {
        ('decoder', 'layers', 'mlp', 'weight'): jnp.array(
            [[10.0, 11.0], [20.0, 21.0]], dtype=jnp.float32
    )
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'weight'): jnp.zeros((2,), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )
    filtered_src_flat, filtered_tgt_flat = param_mapping.execute_transfer_plan(
        plan, src_flat, tgt_flat, scan_axis=0
    )

    np.testing.assert_array_equal(
        filtered_src_flat[('decoder', 'layers_1', 'mlp', 'weight')],
        jnp.array([20.0, 21.0], dtype=jnp.float32),
    )
    self.assertEqual(
        filtered_tgt_flat[('decoder', 'layers_1', 'mlp', 'weight')].shape,
        (2,),
    )
    np.testing.assert_array_equal(
        filtered_tgt_flat[('decoder', 'layers_1', 'mlp', 'weight')],
        jnp.zeros((2,), dtype=jnp.float32),
    )

  def test_execute_transfer_plan_scanned_fused_moe_match(self):
    wi_0 = jnp.array(
        [[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [30.0, 40.0]]],
        dtype=jnp.float32,
    )
    wi_1 = jnp.array(
        [[[100.0, 200.0], [1000.0, 2000.0]], [[300.0, 400.0], [3000.0, 4000.0]]],
        dtype=jnp.float32,
    )
    src_flat = {
        ('decoder', 'layers', 'mlp', 'wi_0'): wi_0,
        ('decoder', 'layers', 'mlp', 'wi_1'): wi_1,
    }
    tgt_flat = {
        ('decoder', 'layers_1', 'mlp', 'wi'): jnp.zeros((2, 4), dtype=jnp.float32)
    }

    plan = param_mapping.build_transfer_plan(
        src_flat,
        tgt_flat,
        operation_rules=param_mapping.default_structural_operation_rules(),
    )
    filtered_src_flat, filtered_tgt_flat = param_mapping.execute_transfer_plan(
        plan, src_flat, tgt_flat, scan_axis=1
    )

    np.testing.assert_array_equal(
        filtered_src_flat[('decoder', 'layers_1', 'mlp', 'wi')],
        jnp.concatenate([wi_0[:, 1, :], wi_1[:, 1, :]], axis=-1),
    )
    self.assertEqual(
        filtered_tgt_flat[('decoder', 'layers_1', 'mlp', 'wi')].shape,
        (2, 4),
    )

  def test_slice_scanned_param_with_repeatable_target(self):
    src = jnp.arange(4 * 3 * 2 * 8, dtype=jnp.float32).reshape(4, 3, 2, 8)
    tgt = jnp.zeros((4, 4, 8), dtype=jnp.float32)
    result = param_mapping_runtime._unstack_scanned_param(
        src, tgt, key_path='test'
    )[1]
    np.testing.assert_equal(result.shape, (4, 2, 8))
    np.testing.assert_array_equal(result, src[:, 1, :, :])

  def test_align_per_axis_attention_pure_repeat(self):
    src = jnp.array(
        [[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=jnp.float32
    )
    tgt_shape = (8, 4)
    result = param_mapping_runtime._align_per_axis(
        src, tgt_shape, tgt_sharding=None, key_path='layers.0.attn.q_proj'
    )
    self.assertEqual(result.shape, tgt_shape)
    expected = jnp.repeat(src, 4, axis=0)
    np.testing.assert_array_equal(np.asarray(result), expected)

  def test_align_per_axis_non_repeatable_non_moe_raises(self):
    src = jnp.zeros((2, 3), dtype=jnp.float32)
    with self.assertRaises(param_mapping.ShapeMismatchError):
      param_mapping_runtime._align_per_axis(
          src, (8, 4), tgt_sharding=None, key_path='layers.0.attn.q_proj'
      )

  def test_align_per_axis_moe_two_axes_zero_pad(self):
    src = jnp.array(
        [[[1., 2.], [3., 4.]]],
        dtype=jnp.float32,
    )
    result = param_mapping_runtime._align_per_axis(
        src, tgt_shape=(2, 2, 4), tgt_sharding=None, key_path='layers.0.wi'
    )
    self.assertEqual(result.shape, (2, 2, 4))
    expected = jnp.pad(src, ((0, 1), (0, 0), (0, 2)))
    np.testing.assert_array_equal(np.asarray(result), expected)


if __name__ == '__main__':
  absltest.main()
