"""Shared helpers and dataclasses for model weight mappings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Optional, Tuple


class BackendMappingMixin:
  """Provides helper methods to retrieve backend-specific weight mappings."""

  DEFAULT_BACKEND = 'vllm_jax'

  @classmethod
  def _backend_registry(cls) -> Dict[str, Any]:
    module = cls.__module__
    package_name = module.rsplit('.', 1)[0] if '.' in module else module
    package = importlib.import_module(package_name)
    return getattr(package, 'BACKEND_MAPPINGS', {})

  @classmethod
  def mapping_for(cls, backend: str | None = None) -> Dict[str, Any]:
    backend = backend or cls.DEFAULT_BACKEND
    registry = cls._backend_registry()
    if backend not in registry:
      raise RuntimeError(
          f'{backend} mappings not available for {cls.__name__}.'
      )
    return registry[backend]

  @classmethod
  def to_hf_mappings(cls, backend: str | None = None):
    mapping = cls.mapping_for(backend).get('to_hf_mappings')
    if mapping is None:
      raise RuntimeError(
          f'{backend} to_hf_mappings missing for {cls.__name__}.'
      )
    return mapping

  @classmethod
  def lora_to_hf_mappings(cls, backend: str | None = None):
    return cls.mapping_for(backend).get('lora_to_hf_mappings')

  @classmethod
  def to_hf_transpose_keys(cls, backend: str | None = None):
    result = cls.mapping_for(backend).get('to_hf_transpose_keys')
    return result or None

  @classmethod
  def to_hf_hook_fns(cls, backend: str | None = None):
    return cls.mapping_for(backend).get('to_hf_hook_fns')


@dataclass
class MappingConfig:
  """Describes how to translate trainer weights into backend weights.

  Prefer using `MappingConfig.build(...)`:
  * `MappingConfig.build(mapping_obj=..., model=..., backend=...)` accepts an
    existing config/dict/object and optionally falls back to extracting the
    data from `model`.
  """

  to_hf_mappings: Optional[Dict[str, Any]] = None
  lora_to_hf_mappings: Optional[Dict[str, Any]] = None
  to_hf_hook_fns: Optional[Dict[str, Any]] = None
  to_hf_transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None

  @classmethod
  def build(
      cls,
      mapping_obj: Any | None = None,
      model: Any | None = None,
      backend: str | None = None,
  ) -> 'MappingConfig':
    """Build the MappingConfg from existing MappingConfig, Dict or other qualified datastructure, or load from model."""
    assert (
        mapping_obj is not None or model is not None
    ), 'Either mapping_obj or model must be provided (both cannot be None)'

    if isinstance(mapping_obj, cls):
      return mapping_obj

    if mapping_obj is None:
      return cls.from_model(model, backend)

    keys = (
        'to_hf_mappings',
        'lora_to_hf_mappings',
        'to_hf_hook_fns',
        'to_hf_transpose_keys',
    )

    values: Dict[str, Any] = {}
    if isinstance(mapping_obj, dict):
      values.update(mapping_obj)
    else:
      for key in keys:
        if hasattr(mapping_obj, key):
          values[key] = getattr(mapping_obj, key)

    resolved: Dict[str, Any] = {}
    for key in keys:
      value = values.get(key)
      if callable(value):
        try:
          value = value()
        except TypeError:
          resolved[key] = value
          continue
      resolved[key] = value

    return cls(
        to_hf_mappings=resolved.get('to_hf_mappings'),
        lora_to_hf_mappings=resolved.get('lora_to_hf_mappings'),
        to_hf_hook_fns=resolved.get('to_hf_hook_fns'),
        to_hf_transpose_keys=resolved.get('to_hf_transpose_keys'),
    )

  @classmethod
  def from_model(
      cls,
      model: Any,
      backend: str = 'vllm_jax',
      **overrides: Any,
  ) -> MappingConfig:
    """Constructs a MappingConfig for the given model and backend."""

    def maybe_call(attr: str):
      value = getattr(model, attr, None)
      if value is None:
        return None
      if callable(value):
        try:
          return value(backend)
        except TypeError:
          return value()
      return value

    config = MappingConfig(
        to_hf_mappings=maybe_call('to_hf_mappings'),
        lora_to_hf_mappings=maybe_call('lora_to_hf_mappings'),
        to_hf_hook_fns=maybe_call('to_hf_hook_fns'),
        to_hf_transpose_keys=maybe_call('to_hf_transpose_keys'),
    )

    for key, value in overrides.items():
      if hasattr(config, key):
        setattr(config, key, value)

    return config
