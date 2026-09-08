"""Protect execution-bearing static state in P59 layer program cache keys."""

import jax
import jax.numpy as jnp
from flax import nnx

from tunix.rl import canonical_qwen3_adapter as adapter


class _Config:

  def __init__(self, scale):
    self.scale = scale


class _SlottedConfig:
  # vars() cannot see scale even though this object has a __dict__.
  __slots__ = ("scale", "__dict__")

  def __init__(self, scale):
    self.scale = scale


class _Layer(nnx.Module):

  def __init__(self, index, config):
    self.prefix = f"model.layers.{index}.projection"
    self.config = config
    self.weight = nnx.Param(jnp.asarray(1.0, jnp.float32))


def _key(layer, index):
  graphdef, _ = nnx.split(layer)
  return adapter._p59_layer_graph_program_key(graphdef, index)[0]


def test_equal_fields_do_not_erase_unknown_object_identity():
  left = _Layer(0, _Config(2))
  right = _Layer(1, _Config(2))
  assert _key(left, 0) != _key(right, 1)


def test_same_config_identity_can_share_after_prefix_normalization():
  config = _Config(2)
  left, right = _Layer(0, config), _Layer(1, config)
  assert _key(left, 0) == _key(right, 1)
  assert hash(_key(left, 0)) == hash(_key(right, 1))


def test_different_slot_state_never_collides():
  left, right = _Layer(0, _SlottedConfig(2)), _Layer(1, _SlottedConfig(3))
  assert vars(left.config) == vars(right.config) == {}
  assert left.config.scale != right.config.scale
  assert _key(left, 0) != _key(right, 1)


def test_key_construction_never_materializes_device_state():
  config = _Config(jnp.asarray([2.0], jnp.float32))
  left, right = _Layer(0, config), _Layer(1, config)
  with jax.transfer_guard("disallow"):
    assert _key(left, 0) == _key(right, 1)


def test_equal_type_names_do_not_erase_type_identity():
  first_type = type("SameName", (), {})
  second_type = type("SameName", (), {})
  assert first_type.__qualname__ == second_type.__qualname__
  left, right = _Layer(0, _Config(first_type)), _Layer(1, _Config(second_type))
  assert _key(left, 0) != _key(right, 1)
