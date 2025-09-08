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

"""Unit tests for sowed module wrapping."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.distillation.feature_extraction import sowed_module
from tunix.tests import test_common as tc


class SimpleLayer(nnx.Module):
  """A basic NNX layer for testing."""

  def __init__(self, tag: str):
    self.tag = tag  # Just to differentiate instances
    self.param = nnx.Param(jnp.array([1.0]))  # Dummy parameter

  def __call__(self, x: jax.Array) -> jax.Array:
    # Simple operation: add param and return
    return x + self.param.value


class Block(nnx.Module, pytree=False):
  """A block containing nested layers."""

  def __init__(self, tag: str):
    self.tag = tag
    self.layer1 = SimpleLayer(f'{tag}_layer1')
    self.layer2 = SimpleLayer(f'{tag}_layer2')  # Another instance
    self.other_layers = [
        SimpleLayer(f'{tag}_layer3'),
        SimpleLayer(f'{tag}_layer4'),
    ]

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.layer1(x)
    x = self.layer2(x)
    for layer in self.other_layers:
      x = layer(x)
    return x


class OuterModel(nnx.Module):
  """A top-level model with nested blocks."""

  def __init__(self):
    self.block1 = Block('block1')
    self.block2 = Block('block2')
    self.final_layer = SimpleLayer('final')

  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.block1(x)
    x = self.block2(x)
    x = self.final_layer(x)
    return x


class SowedModuleTest(absltest.TestCase):

  def test_wrap_model_with_sowed_modules_recursive(self):
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Target SimpleLayer for wrapping
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert all SimpleLayers are wrapped
    self.assertIsInstance(
        model.block1.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.final_layer,
        sowed_module.SowedModule,
    )
    # Check intermediate modules are untouched
    self.assertIsInstance(model.block1, Block)
    self.assertIsInstance(model.block2, Block)
    # Check wrapped models are the originals
    self.assertIs(model.block1.layer1.wrapped_model, original_b1_l1)
    self.assertIs(model.block1.layer2.wrapped_model, original_b1_l2)
    self.assertIs(
        model.block1.other_layers[0].wrapped_model, original_b1_other_layers_0
    )
    self.assertIs(
        model.block1.other_layers[1].wrapped_model, original_b1_other_layers_1
    )
    self.assertIs(model.block2.layer1.wrapped_model, original_b2_l1)
    self.assertIs(model.block2.layer2.wrapped_model, original_b2_l2)
    self.assertIs(
        model.block2.other_layers[0].wrapped_model, original_b2_other_layers_0
    )
    self.assertIs(
        model.block2.other_layers[1].wrapped_model, original_b2_other_layers_1
    )
    self.assertIs(model.final_layer.wrapped_model, original_final)

  def test_wrap_model_with_multiple_target_types(self):
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1 = model.block1
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2 = model.block2
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Target SimpleLayer for wrapping
    target_types = [SimpleLayer, Block]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert all SimpleLayers are wrapped
    self.assertIsInstance(
        model.block1.wrapped_model.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.final_layer,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(model.block1, sowed_module.SowedModule)
    self.assertIsInstance(model.block2, sowed_module.SowedModule)
    # Check wrapped models are the originals
    self.assertIs(
        model.block1.wrapped_model.layer1.wrapped_model, original_b1_l1
    )
    self.assertIs(
        model.block1.wrapped_model.layer2.wrapped_model, original_b1_l2
    )
    self.assertIs(
        model.block1.wrapped_model.other_layers[0].wrapped_model,
        original_b1_other_layers_0,
    )
    self.assertIs(
        model.block1.wrapped_model.other_layers[1].wrapped_model,
        original_b1_other_layers_1,
    )
    self.assertIs(
        model.block2.wrapped_model.layer1.wrapped_model, original_b2_l1
    )
    self.assertIs(
        model.block2.wrapped_model.layer2.wrapped_model, original_b2_l2
    )
    self.assertIs(
        model.block2.wrapped_model.other_layers[0].wrapped_model,
        original_b2_other_layers_0,
    )
    self.assertIs(
        model.block2.wrapped_model.other_layers[1].wrapped_model,
        original_b2_other_layers_1,
    )
    self.assertIs(model.final_layer.wrapped_model, original_final)
    self.assertIs(model.block1.wrapped_model, original_b1)
    self.assertIs(model.block2.wrapped_model, original_b2)

  def test_wrap_model_with_sowed_modules_forward_pass(self):
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer
    # Wrap the model (all SimpleLayers)
    sowed_module.wrap_model_with_sowed_modules(model, [SimpleLayer])
    # Assertions about captured state after forward pass
    output = model(dummy_input)  # Run forward pass to populate intermediates
    state = sowed_module.pop_sowed_intermediate_outputs(model).to_pure_dict()

    val_b1_l1 = original_b1_l1(dummy_input)  # Input: dummy_input
    val_b1_l2 = original_b1_l2(val_b1_l1)  # Input: output of b1_l1
    val_b1_other_layers_0 = original_b1_other_layers_0(
        val_b1_l2
    )  # Input: output of b1_l2
    val_b1_other_layers_1 = original_b1_other_layers_1(
        val_b1_other_layers_0
    )  # Input: output of b1_other_layers_0
    val_b2_l1 = original_b2_l1(
        val_b1_other_layers_1
    )  # Input: output of b1_other_layers_1
    val_b2_l2 = original_b2_l2(val_b2_l1)  # Input: output of b2_l1
    val_b2_other_layers_0 = original_b2_other_layers_0(
        val_b2_l2
    )  # Input: output of b2_l2
    val_b2_other_layers_1 = original_b2_other_layers_1(
        val_b2_other_layers_0
    )  # Input: output of b2_other_layers_0
    val_final = original_final(
        val_b2_other_layers_1
    )  # Input: output of b2_other_layers_1
    # Final output should match val_final
    tc.assert_close('final_output', output, val_final)

    # Check state structure and values for ALL wrapped layers
    tag = sowed_module.SowedModule._SOW_TAG
    expected_state = {
        'block1': {
            'layer1': {tag: (val_b1_l1,)},
            'layer2': {tag: (val_b1_l2,)},
            'other_layers': {
                0: {tag: (val_b1_other_layers_0,)},
                1: {tag: (val_b1_other_layers_1,)},
            },
        },
        'block2': {
            'layer1': {tag: (val_b2_l1,)},
            'layer2': {tag: (val_b2_l2,)},
            'other_layers': {
                0: {tag: (val_b2_other_layers_0,)},
                1: {tag: (val_b2_other_layers_1,)},
            },
        },
        'final_layer': {tag: (val_final,)},
    }
    jax.tree.map_with_path(
        tc.assert_close,
        state,
        expected_state,
    )

  def test_capture_avoids_double_wrapping(self):
    model = Block('block')
    original_l1 = model.layer1
    original_l2 = model.layer2
    # Manually wrap one layer first
    manual_wrapper = sowed_module.SowedModule(original_l1)
    model.layer1 = manual_wrapper

    # Run the utility function
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert manually wrapped layer wasn't re-wrapped
    self.assertIs(model.layer1, manual_wrapper)
    self.assertIs(model.layer1.wrapped_model, original_l1)
    # Assert the other target layers was wrapped
    self.assertIsInstance(
        model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIs(model.layer2.wrapped_model, original_l2)

  def test_unwrap_sowed_modules_recursive(self):
    model = OuterModel()
    # Store references to ALL original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Wrap the model (all SimpleLayers)
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Unwrap the sowed modules
    sowed_module.unwrap_sowed_modules(model)

    # Check if the instances are the originals
    self.assertIs(model.block1.layer1, original_b1_l1)
    self.assertIs(model.block1.layer2, original_b1_l2)
    self.assertIs(model.block1.other_layers[0], original_b1_other_layers_0)
    self.assertIs(model.block1.other_layers[1], original_b1_other_layers_1)
    self.assertIs(model.block2.layer1, original_b2_l1)
    self.assertIs(model.block2.layer2, original_b2_l2)
    self.assertIs(model.block2.other_layers[0], original_b2_other_layers_0)
    self.assertIs(model.block2.other_layers[1], original_b2_other_layers_1)
    self.assertIs(model.final_layer, original_final)


if __name__ == '__main__':
  absltest.main()
