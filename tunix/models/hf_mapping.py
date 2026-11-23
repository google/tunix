# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Declarative PrimOp Weight Conversion Protocol, manifest generator, and JAX engine."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
import dataclasses
import enum
import functools
import itertools
import json
import math
import string
from typing import Any, Final, TypeAlias, TypedDict, TypeVar

import jax
from jax import typing as jax_typing
import jax.numpy as jnp
import ml_dtypes
import numpy as np

SCHEMA_VERSION: Final[str] = "weight_mapping"

ShardingAxisSpec: TypeAlias = str | list[str] | tuple[str, ...] | None

_VariantT = TypeVar("_VariantT", bound=enum.Enum)

RULE_KEYS: Final[frozenset[str]] = frozenset(
    {"source_keys", "target_keys", "ops", "sharding"}
)

VALID_OP_SIGNATURES: Final[frozenset[tuple[str, frozenset[str]]]] = frozenset({
    ("stack", frozenset({"axis"})),
    ("unstack", frozenset({"axis"})),
    ("reshape", frozenset({"src_shape", "dst_shape"})),
    ("transpose", frozenset({"axes"})),
    ("scale", frozenset({"factor"})),
    ("concat", frozenset({"axis", "sizes"})),
    ("split", frozenset({"axis", "sizes"})),
    ("pad", frozenset({"axis", "sizes"})),
    ("unpad", frozenset({"axis", "sizes"})),
    ("cast", frozenset({"src_dtype", "dst_dtype"})),
    ("repeat", frozenset({"axis", "times"})),
    ("unrepeat", frozenset({"axis", "times"})),
})

SUPPORTED_DTYPES: Final[frozenset[str]] = frozenset({
    "float32",
    "bfloat16",
    "float16",
    "float8_e4m3fn",
    "float8_e5m2",
    "int32",
    "int8",
    "uint8",
})

_INV_OPS: Final[Mapping[str, str]] = {
    "stack": "unstack",
    "unstack": "stack",
    "concat": "split",
    "split": "concat",
    "pad": "unpad",
    "unpad": "pad",
    "repeat": "unrepeat",
    "unrepeat": "repeat",
}


class ConversionRuleDict(TypedDict, total=False):
  """JSON-serializable single weight-conversion rule with optional sharding."""

  source_keys: list[str]
  target_keys: list[str]
  ops: list[dict[str, dict[str, Any]]]
  sharding: list[ShardingAxisSpec] | None


class WeightMappingManifest(TypedDict, total=False):
  """Top-level JSON-serializable declarative weight-conversion manifest."""

  schema_version: str
  model_type: str
  rules: list[ConversionRuleDict]
  tied_keys: list[list[str]]
  dropped_keys: list[str]


class ModelTemplateRule(TypedDict, total=False):
  """Compact per-model-family template rule expanded at init time."""

  id: str
  axes: Mapping[str, int | Sequence[int]]
  source_pattern: list[str]
  target_pattern: list[str]
  ops: list[dict[str, dict[str, Any]]]


@dataclasses.dataclass(frozen=True)
class ParamRule:
  """Immutable representation of a single weight-conversion rule."""

  source_keys: tuple[str, ...]
  target_keys: tuple[str, ...]
  ops: tuple[dict[str, dict[str, Any]], ...] = ()
  sharding: tuple[ShardingAxisSpec, ...] | None = None

  def __post_init__(self) -> None:
    validate_rule(self.to_dict())

  def to_dict(self) -> dict[str, Any]:
    """Serializes the rule into a canonical JSON-compatible dictionary."""
    out: dict[str, Any] = {
        "source_keys": list(self.source_keys),
        "target_keys": list(self.target_keys),
        "ops": [{k: dict(v) for k, v in step.items()} for step in self.ops],
    }
    if self.sharding is not None:
      out["sharding"] = list(self.sharding)
    return out

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> ParamRule:
    """Constructs a validated `ParamRule` from a dictionary."""
    validate_rule(data)
    sharding = data["sharding"] if "sharding" in data else None
    return cls(
        source_keys=tuple(str(k) for k in data["source_keys"]),
        target_keys=tuple(str(k) for k in data["target_keys"]),
        ops=tuple(
            {str(k): dict(v) for k, v in step.items()} for step in data["ops"]
        ),
        sharding=tuple(sharding) if sharding is not None else None,
    )


@dataclasses.dataclass(frozen=True)
class WeightMappingSpec:
  """Immutable container for a declarative weight-conversion manifest."""

  rules: tuple[ParamRule, ...] = ()
  tied_keys: tuple[tuple[str, str], ...] = ()
  dropped_keys: tuple[str, ...] = ()
  schema_version: str = SCHEMA_VERSION
  model_type: str = ""

  def __post_init__(self) -> None:
    if self.schema_version != SCHEMA_VERSION:
      raise ValueError(f"Unsupported schema_version: {self.schema_version!r}")
    if not isinstance(self.model_type, str) or not _is_str_seq(
        self.dropped_keys, allow_empty=True
    ):
      raise ValueError("Invalid model_type or dropped_keys.")
    if not _is_seq(self.tied_keys) or not all(
        _is_str_seq(p) and len(p) == 2 for p in self.tied_keys
    ):
      raise ValueError(f"Invalid tied_keys: {self.tied_keys!r}")

  def to_dict(self) -> dict[str, Any]:
    """Serializes the manifest specification into a dictionary."""
    out: dict[str, Any] = {
        "schema_version": self.schema_version,
        "rules": [r.to_dict() for r in self.rules],
        "tied_keys": [[src, dst] for src, dst in self.tied_keys],
        "dropped_keys": list(self.dropped_keys),
    }
    if self.model_type:
      out["model_type"] = self.model_type
    return out

  def to_json(self) -> str:
    """Serializes the manifest specification into a canonical JSON string."""
    return json.dumps(self.to_dict(), sort_keys=True)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> WeightMappingSpec:
    """Constructs a validated `WeightMappingSpec` from a dictionary."""
    spec = parse_manifest(data)
    model_type = spec["model_type"] if "model_type" in spec else ""
    return cls(
        rules=tuple(ParamRule.from_dict(r) for r in spec["rules"]),
        tied_keys=tuple((str(a), str(b)) for a, b in spec["tied_keys"]),
        dropped_keys=tuple(str(k) for k in spec["dropped_keys"]),
        schema_version=str(spec["schema_version"]),
        model_type=str(model_type),
    )

  @classmethod
  def from_json(cls, payload: str) -> WeightMappingSpec:
    """Deserializes a `WeightMappingSpec` from a JSON string."""
    return cls.from_dict(parse_manifest(payload))


RuleSpec: TypeAlias = Mapping[str, Any] | str | WeightMappingSpec


def _is_int(val: Any) -> bool:
  """Returns True if `val` is an `int` and not a `bool`."""
  return isinstance(val, int) and not isinstance(val, bool)


def _is_seq(val: Any) -> bool:
  """Returns True if `val` is a non-string/bytes Sequence."""
  return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def _is_str_seq(val: Any, *, allow_empty: bool = False) -> bool:
  """Returns True if `val` is a sequence of non-empty strings (not a str)."""
  return (
      _is_seq(val)
      and (allow_empty or bool(val))
      and all(isinstance(s, str) and bool(s) for s in val)
  )


def _is_sharding_spec(val: Any) -> bool:
  """Returns True if `val` is None or a sequence of valid axis specs."""
  return val is None or (
      _is_seq(val)
      and all(
          a is None or _is_str_seq([a] if isinstance(a, str) else a)
          for a in val
      )
  )


_OP_CHECKS: Final[Mapping[str, Callable[[Mapping[str, Any]], bool]]] = {
    "stack": lambda kw: _is_int(kw["axis"]),
    "unstack": lambda kw: _is_int(kw["axis"]),
    "reshape": lambda kw: (
        min(kw["src_shape"], default=0) > 0
        and min(kw["dst_shape"], default=0) > 0
        and math.prod(kw["src_shape"]) == math.prod(kw["dst_shape"])
    ),
    "transpose": (
        lambda kw: bool(kw["axes"])
        and sorted(kw["axes"]) == list(range(len(kw["axes"])))
    ),
    "scale": (
        lambda kw: math.isfinite(float(kw["factor"]))
        and float(kw["factor"]) != 0.0
    ),
    "concat": (
        lambda kw: _is_int(kw["axis"]) and min(kw["sizes"], default=0) > 0
    ),
    "split": lambda kw: _is_int(kw["axis"]) and min(kw["sizes"], default=0) > 0,
    "pad": (
        lambda kw: _is_int(kw["axis"])
        and len(kw["sizes"]) == 2
        and min(kw["sizes"]) >= 0
    ),
    "unpad": (
        lambda kw: _is_int(kw["axis"])
        and len(kw["sizes"]) == 2
        and min(kw["sizes"]) >= 0
    ),
    "cast": lambda kw: not (
        {kw["src_dtype"], kw["dst_dtype"]} - SUPPORTED_DTYPES
    ),
    "repeat": (
        lambda kw: _is_int(kw["axis"])
        and _is_int(kw["times"])
        and kw["times"] >= 1
    ),
    "unrepeat": (
        lambda kw: _is_int(kw["axis"])
        and _is_int(kw["times"])
        and kw["times"] >= 1
    ),
}


def validate_op(op_name: str, kw: Mapping[str, Any]) -> None:
  """Validates a single primitive op name and its symmetric parameters."""
  if not isinstance(kw, Mapping) or (
      (op_name, frozenset(kw)) not in VALID_OP_SIGNATURES
  ):
    raise ValueError(f"Invalid primitive op {op_name!r} with kwargs {kw!r}")
  if not _OP_CHECKS[op_name](kw):
    raise ValueError(f"Invalid parameters for {op_name!r}: {dict(kw)}")


def invert_op(op_name: str, kw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
  """Returns the exact algebraic inverse of a single primitive operation."""
  validate_op(op_name, kw)
  if op_name in _INV_OPS:
    return {_INV_OPS[op_name]: dict(kw)}
  if op_name == "reshape":
    return {
        "reshape": {
            "src_shape": list(kw["dst_shape"]),
            "dst_shape": list(kw["src_shape"]),
        }
    }
  if op_name == "transpose":
    axes: Sequence[int] = kw["axes"]
    return {"transpose": {"axes": [axes.index(i) for i in range(len(axes))]}}
  if op_name == "scale":
    return {"scale": {"factor": 1.0 / float(kw["factor"])}}
  return {
      "cast": {
          "src_dtype": str(kw["dst_dtype"]),
          "dst_dtype": str(kw["src_dtype"]),
      }
  }


def validate_rule(rule: Mapping[str, Any]) -> None:
  """Validates a declarative conversion rule dictionary."""
  if not isinstance(rule, Mapping) or set(rule) | {"sharding"} != RULE_KEYS:
    raise ValueError(f"Invalid rule schema: {rule!r}")
  sharding = rule["sharding"] if "sharding" in rule else None
  if not (
      _is_str_seq(rule["source_keys"])
      and _is_str_seq(rule["target_keys"])
      and _is_seq(rule["ops"])
      and _is_sharding_spec(sharding)
  ):
    raise ValueError(f"Invalid rule keys, ops, or sharding: {rule!r}")
  for step in rule["ops"]:
    if not isinstance(step, Mapping) or len(step) != 1:
      raise ValueError(f"Each op step must be a 1-key mapping, got {step!r}")
    validate_op(*next(iter(step.items())))


def parse_manifest(manifest: RuleSpec) -> dict[str, Any]:
  """Normalizes and validates a JSON string, mapping, or `WeightMappingSpec`."""
  if isinstance(manifest, WeightMappingSpec):
    raw: Any = manifest.to_dict()
  elif isinstance(manifest, str):
    raw = json.loads(manifest)
  else:
    raw = manifest
  if not isinstance(raw, Mapping) or "rules" not in raw:
    raise ValueError(f"Manifest must be a mapping with 'rules', got {raw!r}")
  spec: dict[str, Any] = {
      "schema_version": SCHEMA_VERSION,
      "tied_keys": [],
      "dropped_keys": [],
      **raw,
  }
  WeightMappingSpec(
      tied_keys=spec["tied_keys"],
      dropped_keys=spec["dropped_keys"],
      schema_version=spec["schema_version"],
      model_type=spec["model_type"] if "model_type" in spec else "",
  )
  if not _is_seq(spec["rules"]):
    raise ValueError(
        f"Manifest rules must be a sequence, got {spec['rules']!r}"
    )
  for rule in spec["rules"]:
    validate_rule(rule)
  dropped = set(spec["dropped_keys"])
  targets = [
      *itertools.chain.from_iterable(r["target_keys"] for r in spec["rules"]),
      *(dst for _, dst in spec["tied_keys"] if dst not in dropped),
  ]
  if len(targets) != len(set(targets)):
    raise ValueError(f"Duplicate target_keys in manifest: {targets!r}")
  return spec


def invert_manifest(manifest: RuleSpec) -> dict[str, Any]:
  """Derives a single model's `Model -> HF` manifest from its `HF -> Model` spec.

  Note: In cross-repo distributed RL (`Trainer A -> HF -> Sampler B`), each
  engine authors its own directional manifest (`src_rule_A` vs. `dst_rule_B`).
  `invert_manifest` is a convenience for a single model implementation to
  mechanically derive its `Model -> HF` export manifest from its `HF -> Model`
  load manifest without maintaining two definitions.

  Args:
    manifest: Forward `HF -> Model` manifest specification.

  Returns:
    Inverted `Model -> HF` manifest dictionary.
  """
  spec = parse_manifest(manifest)
  inv_rules: list[dict[str, Any]] = []
  for r in spec["rules"]:
    inv_ops = [invert_op(*next(iter(s.items()))) for s in reversed(r["ops"])]
    inv_rules.append({
        "source_keys": list(r["target_keys"]),
        "target_keys": list(r["source_keys"]),
        "ops": inv_ops,
    })
  tied_dst = [dst for _, dst in spec["tied_keys"]]
  return {
      **spec,
      "rules": inv_rules,
      "tied_keys": [],
      "dropped_keys": [*spec["dropped_keys"], *tied_dst],
  }


def render_json_templates(
    *json_blocks: str, **template_vars: int | str
) -> list[ModelTemplateRule]:
  """Renders one or more JSON rule-list blocks via `string.Template`."""
  return list(
      itertools.chain.from_iterable(
          json.loads(string.Template(b).substitute(template_vars))
          for b in json_blocks
          if b
      )
  )


def resolve_hf_variant_from_keys(
    keys: Collection[str],
    affixes_by_variant: Mapping[_VariantT, Mapping[str, str]],
    *,
    anchor_suffix: str = "embed_tokens.weight",
) -> _VariantT:
  """Returns the unique variant whose `${lm}<anchor_suffix>` is in `keys`."""
  anchor_to_variant: dict[str, _VariantT] = {}
  for variant, affixes in affixes_by_variant.items():
    anchor = f"{affixes['lm']}{anchor_suffix}"
    if anchor in anchor_to_variant:
      raise ValueError(f"Duplicate HF variant anchor key: {anchor!r}")
    anchor_to_variant[anchor] = variant
  key_set = keys if isinstance(keys, (set, frozenset)) else set(keys)
  matched = [v for anchor, v in anchor_to_variant.items() if anchor in key_set]
  if len(matched) != 1:
    raise ValueError(
        "Checkpoint keys must match exactly one HF variant anchor in "
        f"{sorted(anchor_to_variant)}; matched {len(matched)}."
    )
  return matched[0]


def active_source_keys(manifest: RuleSpec) -> frozenset[str]:
  """Returns the active source keys from `manifest` (excluding `dropped_keys`)."""
  spec = parse_manifest(manifest)
  dropped = frozenset(spec["dropped_keys"])
  return frozenset(
      itertools.chain.from_iterable(
          r["source_keys"]
          for r in spec["rules"]
          if not dropped.intersection((*r["source_keys"], *r["target_keys"]))
      )
  )


def lora_mapping_from_manifest(
    to_hf_manifest: RuleSpec,
) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
  """Derives linear `(module_to_hf_key, transpose_rules)` from `Model -> HF`."""
  spec = parse_manifest(to_hf_manifest)
  key_map: dict[str, str] = {}
  transpose_rules: dict[str, tuple[int, ...]] = {}
  for r in spec["rules"]:
    src, targets, ops = r["source_keys"], r["target_keys"], r["ops"]
    if len(src) != 1 or not str(src[0]).endswith((".w", ".kernel")):
      continue
    if len(targets) != 1 and ops[:1] != [{"unstack": {"axis": 0}}]:
      continue
    mod = str(src[0]).rsplit(".", 1)[0]
    m_keys = (
        [mod]
        if len(targets) == 1
        else [f"{mod}.{i}" for i in range(len(targets))]
    )
    key_map.update(zip(m_keys, map(str, targets), strict=True))
    is_head_2d = (
        len(ops) >= 2
        and "transpose" in ops[-2]
        and "reshape" in ops[-1]
        and len(ops[-1]["reshape"]["dst_shape"]) == 2
    )
    if {"transpose": {"axes": [1, 0]}} in ops or is_head_2d:
      transpose_rules.update(dict.fromkeys(map(str, targets), (1, 0)))
  return key_map, transpose_rules


def generate_manifest(
    model_type: str,
    template_rules: Sequence[Mapping[str, Any]],
    *,
    axes: Mapping[str, int | Sequence[int | str]] | None = None,
    sharding_rules: Mapping[str, Sequence[ShardingAxisSpec]] | None = None,
    tied_keys: Sequence[tuple[str, str]] = (),
    dropped_keys: Sequence[str] = (),
) -> dict[str, Any]:
  """Expands model template rules into a declarative manifest."""
  base_axes = {} if axes is None else dict(axes)
  shard_map = {} if sharding_rules is None else dict(sharding_rules)
  fmt_parse = string.Formatter().parse
  rules: list[dict[str, Any]] = []
  for tmpl in template_rules:
    t_map: dict[str, Any] = {"id": "", "axes": {}, **tmpl}
    src_pats, tgt_pats = t_map["source_pattern"], t_map["target_pattern"]
    src_vars = set(
        filter(None, (t[1] for t in fmt_parse("".join(src_pats))))
    )
    tgt_vars = set(
        filter(None, (t[1] for t in fmt_parse("".join(tgt_pats))))
    )
    outer_vars = sorted(src_vars & tgt_vars)
    inner_src = sorted(src_vars - tgt_vars)
    inner_tgt = sorted(tgt_vars - src_vars)
    pools = {
        k: range(s) if isinstance(s, int) else s
        for k, s in {**base_axes, **t_map["axes"]}.items()
    }
    src_in = [
        dict(zip(inner_src, vs))
        for vs in itertools.product(*(pools[v] for v in inner_src))
    ]
    tgt_in = [
        dict(zip(inner_tgt, vs))
        for vs in itertools.product(*(pools[v] for v in inner_tgt))
    ]
    rule_id = str(t_map["id"])
    shard = (
        {"sharding": list(shard_map[rule_id])} if rule_id in shard_map else {}
    )
    for vs in itertools.product(*(pools[v] for v in outer_vars)):
      outer_env = dict(zip(outer_vars, vs))
      rules.append({
          "source_keys": [
              p.format_map(outer_env | ie)
              for ie, p in itertools.product(src_in, src_pats)
          ],
          "target_keys": [
              p.format_map(outer_env | ie)
              for ie, p in itertools.product(tgt_in, tgt_pats)
          ],
          "ops": [dict(step) for step in t_map["ops"]],
          **shard,
      })
  return parse_manifest({
      "schema_version": SCHEMA_VERSION,
      "model_type": model_type,
      "rules": rules,
      "tied_keys": [[s, d] for s, d in tied_keys],
      "dropped_keys": list(dropped_keys),
  })


_JAX_DTYPES: Final[Mapping[str, jax_typing.DTypeLike]] = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
    "int32": jnp.int32,
    "int8": jnp.int8,
    "uint8": jnp.uint8,
}

_NP_DTYPES: Final[Mapping[str, Any]] = {
    "float32": np.float32,
    "bfloat16": ml_dtypes.bfloat16,
    "float16": np.float16,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def _slice_axis(
    t: Any, axis: int, start: int, stop: int, step: int = 1
) -> Any:
  """Slices `t` along `axis` (`jax.Array` or `np.ndarray` view)."""
  idx: list[slice] = [slice(None)] * t.ndim
  idx[axis] = slice(start, stop, step)
  return t[tuple(idx)]


_UNARY_OPS: Final[Mapping[str, tuple[Callable[..., Any], str]]] = {
    "reshape": (lambda xp, t, s: xp.reshape(t, s), "dst_shape"),
    "transpose": (lambda xp, t, a: xp.transpose(t, a), "axes"),
    "scale": (
        lambda xp, t, f: (t.astype(xp.float32) * float(f)).astype(t.dtype),
        "factor",
    ),
}


def _unrepeat_axis(xp: Any, t: Any, kw: Mapping[str, Any]) -> Any:
  """Slices repeated groups along `axis`, verifying group equality in NumPy."""
  axis, times = int(kw["axis"]), int(kw["times"])
  first = _slice_axis(t, axis, 0, t.shape[axis], times)
  if (
      xp is np
      and times > 1
      and not np.array_equal(
          t, np.repeat(first, times, axis=axis), equal_nan=True
      )
  ):
    raise ValueError(
        f"Non-identical repeated slices along axis={axis} (times={times})"
    )
  return first


_AXIS_OPS: Final[Mapping[str, Callable[[Any, Any, Mapping[str, Any]], Any]]] = {
    "repeat": lambda xp, t, kw: xp.repeat(
        t, int(kw["times"]), axis=int(kw["axis"])
    ),
    "unrepeat": _unrepeat_axis,
    "unpad": lambda _xp, t, kw: _slice_axis(
        t,
        int(kw["axis"]),
        int(kw["sizes"][0]),
        t.shape[int(kw["axis"])] - int(kw["sizes"][1]),
    ),
}

_RUNTIME_SHAPE_CHECKS: Final[
    Mapping[str, Callable[[Sequence[Any], Mapping[str, Any]], bool]]
] = {
    "unstack": lambda ts, _kw: len(ts) == 1,
    "reshape": (
        lambda ts, kw: [t.shape for t in ts]
        == [tuple(kw["src_shape"])] * len(ts)
    ),
    "concat": (
        lambda ts, kw: bool(ts)
        and [t.shape[int(kw["axis"])] for t in ts]
        == list(kw["sizes"]) * (len(ts) // len(kw["sizes"]))
    ),
    "split": (
        lambda ts, kw: [t.shape[int(kw["axis"])] for t in ts]
        == [sum(kw["sizes"])] * len(ts)
    ),
    "unpad": lambda ts, kw: all(
        t.shape[int(kw["axis"])] > sum(kw["sizes"]) for t in ts
    ),
    "unrepeat": lambda ts, kw: not any(
        t.shape[int(kw["axis"])] % int(kw["times"]) for t in ts
    ),
    "cast": lambda ts, kw: all(
        np.dtype(t.dtype) == np.dtype(_NP_DTYPES[str(kw["src_dtype"])])
        for t in ts
    ),
}


def _apply_primitive_op(
    tensors: Sequence[Any],
    op_name: str,
    kw: Mapping[str, Any],
    *,
    use_numpy: bool = False,
) -> list[Any]:
  """Executes one of the self-invertible PrimOp weight-conversion ops."""
  validate_op(op_name, kw)
  if op_name in _RUNTIME_SHAPE_CHECKS and not _RUNTIME_SHAPE_CHECKS[op_name](
      tensors, kw
  ):
    raise ValueError(
        f"Runtime shape mismatch for {op_name!r} with {dict(kw)}: "
        f"{[t.shape for t in tensors]}"
    )
  xp: Any = np if use_numpy else jnp
  if op_name == "stack":
    return [xp.stack(tensors, axis=int(kw["axis"]))]
  if op_name == "concat":
    step = len(kw["sizes"])
    return [
        xp.concatenate(tensors[i : i + step], axis=int(kw["axis"]))
        for i in range(0, len(tensors), step)
    ]
  if op_name == "cast":
    dst = str(kw["dst_dtype"])
    if use_numpy:
      return [np.asarray(t, dtype=_NP_DTYPES[dst]) for t in tensors]
    return [jax.lax.convert_element_type(t, _JAX_DTYPES[dst]) for t in tensors]
  if op_name in _UNARY_OPS:
    fn, key = _UNARY_OPS[op_name]
    return [fn(xp, t, kw[key]) for t in tensors]
  if op_name in _AXIS_OPS:
    axis_fn = _AXIS_OPS[op_name]
    return [axis_fn(xp, t, kw) for t in tensors]
  if op_name == "unstack":
    t = tensors[0]
    axis = int(kw["axis"])
    return [
        xp.squeeze(s, axis=axis) for s in xp.split(t, t.shape[axis], axis=axis)
    ]
  if op_name == "split":
    cuts = tuple(itertools.accumulate(kw["sizes"][:-1]))
    axis = int(kw["axis"])
    return list(
        itertools.chain.from_iterable(
            xp.split(t, cuts, axis=axis) for t in tensors
        )
    )
  out: list[Any] = []
  for t in tensors:
    pads: list[tuple[int, int]] = [(0, 0)] * t.ndim
    pads[int(kw["axis"])] = (int(kw["sizes"][0]), int(kw["sizes"][1]))
    out.append(xp.pad(t, pads))
  return out


def apply_primitive_op(
    tensors: list[jax.Array], op_name: str, kw: Mapping[str, Any]
) -> list[jax.Array]:
  """Executes one of the self-invertible PrimOp weight-conversion ops."""
  return _apply_primitive_op(tensors, op_name, kw, use_numpy=False)


def _apply_rule(
    rule: Mapping[str, Any],
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
    *,
    use_numpy: bool = False,
) -> dict[str, Any]:
  """Executes a single weight-conversion rule on JAX or host NumPy."""
  validate_rule(rule)
  asarray = np.asarray if use_numpy else jnp.asarray
  tensors: list[Any] = [asarray(src_weights[k]) for k in rule["source_keys"]]
  for step in rule["ops"]:
    tensors = _apply_primitive_op(
        tensors, *next(iter(step.items())), use_numpy=use_numpy
    )
  sharding_spec = rule["sharding"] if "sharding" in rule else None
  if not use_numpy and mesh is not None and sharding_spec:
    pspec = jax.sharding.PartitionSpec(
        *(tuple(a) if isinstance(a, list) else a for a in sharding_spec)
    )
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    tensors = [jax.lax.with_sharding_constraint(t, sharding) for t in tensors]
  return dict(zip(rule["target_keys"], tensors, strict=True))


def apply_rule(
    rule: Mapping[str, Any],
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Executes a single declarative weight-conversion rule over `src_weights`."""
  return _apply_rule(rule, src_weights, mesh=mesh, use_numpy=False)


def _execute_manifest(
    manifest: RuleSpec,
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
    *,
    use_numpy: bool = False,
) -> dict[str, Any]:
  """Executes a declarative manifest on JAX or host NumPy."""
  spec = parse_manifest(manifest)
  dropped = set(spec["dropped_keys"])
  rules = spec["rules"]
  tied_dst = {dst for _, dst in spec["tied_keys"]}
  consumed = (
      set(itertools.chain.from_iterable(r["source_keys"] for r in rules))
      | dropped
      | tied_dst
  )
  asarray = np.asarray if use_numpy else jnp.asarray
  out: dict[str, Any] = {
      k: asarray(src_weights[k]) for k in src_weights.keys() - consumed
  }
  for rule in rules:
    if dropped.intersection((*rule["source_keys"], *rule["target_keys"])):
      continue
    out.update(_apply_rule(rule, src_weights, mesh=mesh, use_numpy=use_numpy))
  return out | {
      dst: out[src]
      for src, dst in spec["tied_keys"]
      if src in out and dst not in dropped
  }


def execute_manifest(
    manifest: RuleSpec,
    src_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Executes a declarative PrimOp weight-conversion manifest."""
  return _execute_manifest(manifest, src_weights, mesh=mesh, use_numpy=False)


def to_hf_impl(
    src_spec: RuleSpec,
    src_weights: Mapping[str, Any],
) -> dict[str, jax.Array]:
  """Pure JAX-jittable kernel converting source weights into HF layout.

  Architectural role: `to_hf_impl` is the inner JAX transformation kernel
  designed to be traced and fused directly inside `jax.jit` or `jax.eval_shape`
  (e.g., Scenario 1 colocated weight sync:
  `jax.jit(lambda w: from_hf_impl(dst_spec, to_hf_impl(src_spec, w), mesh=m))`).
  Unlike the cross-process boundary entrypoint `to_hf`, callers of `to_hf_impl`
  within a JIT-compiled function pass pre-parsed manifest specs or static rule
  mappings alongside traced `jax.Array` pytrees.

  Args:
    src_spec: Declarative `Trainer -> HF` weight-mapping specification authored
      for the source model's layout (independent of the target sampler's
      `dst_spec`; a single-model implementation may also derive this from its
      `HF -> Model` manifest via `invert_manifest`).
    src_weights: Mapping of source parameter names to `jax.Array`s.

  Returns:
    Mapping of canonical HuggingFace parameter names to transformed
    `jax.Array`s.
  """
  spec = parse_manifest(src_spec)
  tied_dst = [dst for _, dst in spec["tied_keys"]]
  src_manifest = {
      **spec,
      "tied_keys": [],
      "dropped_keys": [*spec["dropped_keys"], *tied_dst],
  }
  return execute_manifest(src_manifest, src_weights, mesh=None)


def from_hf_impl(
    dst_spec: RuleSpec,
    hf_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Pure JAX-jittable kernel converting HF weights into target layout.

  Architectural role: `from_hf_impl` is the inner JAX transformation and
  GSPMD sharding kernel designed to be traced and fused directly inside
  `jax.jit` or `jax.eval_shape` (e.g., Scenario 1 colocated weight sync:
  `jax.jit(lambda w: from_hf_impl(dst_spec, to_hf_impl(src_spec, w), mesh=m))`).
  Unlike the cross-process boundary entrypoint `from_hf`, `from_hf_impl` is the
  composable XLA kernel invoked inside JIT-compiled weight transfer pipelines.

  Args:
    dst_spec: Declarative `HF -> Target` weight-mapping specification for the
      target model layout.
    hf_weights: Mapping of canonical HuggingFace parameter names to
      `jax.Array`s.
    mesh: Optional target JAX device mesh for `with_sharding_constraint`.

  Returns:
    Mapping of target parameter names to transformed (and optionally sharded)
    `jax.Array`s.
  """
  return execute_manifest(dst_spec, hf_weights, mesh=mesh)


def to_hf(
    src_rule: str,
    src_weights: Mapping[str, Any],
) -> dict[str, jax.Array]:
  """Cross-process API boundary entrypoint converting weights into HF layout.

  Architectural role: `to_hf` is the public cross-process/RPC boundary
  entrypoint (Scenario 2 disaggregated weight sync and Scenario 3 checkpoint
  export). It accepts a raw serialized JSON manifest string (`src_rule: str`)
  across process boundaries, validates and deserializes the JSON payload via
  `WeightMappingSpec.from_json(src_rule)`, and delegates tensor execution to
  the pure JAX kernel `to_hf_impl`. Do not remove `to_hf_impl` or collapse the
  two layers.

  Args:
    src_rule: Raw serialized JSON string describing the source engine's `Trainer
      -> HF` weight-conversion rules (`src_rule_A`, independent of the target
      sampler's `dst_rule_B`).
    src_weights: Mapping of source parameter names to arrays.

  Returns:
    Mapping of canonical HuggingFace parameter names to `jax.Array`s.
  """
  parsed = WeightMappingSpec.from_json(src_rule)
  return to_hf_impl(parsed, src_weights)


def from_hf(
    dst_rule: str,
    hf_weights: Mapping[str, Any],
    mesh: jax.sharding.Mesh | None = None,
) -> dict[str, jax.Array]:
  """Cross-process API boundary entrypoint converting HF weights to target.

  Architectural role: `from_hf` is the public cross-process/RPC boundary
  entrypoint (Scenario 2 disaggregated weight sync and Scenario 3 checkpoint
  loading). It accepts a raw serialized JSON manifest string (`dst_rule: str`)
  across process boundaries, validates and deserializes the JSON payload via
  `WeightMappingSpec.from_json(dst_rule)`, and delegates tensor execution and
  sharding to the pure JAX kernel `from_hf_impl`. Do not remove `from_hf_impl`
  or collapse the two layers.

  Args:
    dst_rule: Raw serialized JSON string describing the target engine's `HF ->
      Sampler` weight-conversion rules (`dst_rule_B`, independent of the source
      trainer's `src_rule_A`).
    hf_weights: Mapping of canonical HuggingFace parameter names to arrays.
    mesh: Optional target JAX device mesh for sharding constraints.

  Returns:
    Mapping of target parameter names to `jax.Array`s.
  """
  parsed = WeightMappingSpec.from_json(dst_rule)
  return from_hf_impl(parsed, hf_weights, mesh=mesh)


def _step_streaming_rules(
    base_spec: Mapping[str, Any],
    remaining_rules: list[tuple[frozenset[str], Mapping[str, Any]]],
    rule_sources: AbstractSet[str],
    pending: dict[str, Any],
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
  """Buffers rule sources in `pending` and executes ready rules on host NumPy."""
  pending.update({k: tensors[k] for k in tensors.keys() & rule_sources})
  ready_rules = [
      rule for src_set, rule in remaining_rules if src_set <= pending.keys()
  ]
  remaining_rules[:] = [
      (src_set, rule)
      for src_set, rule in remaining_rules
      if not (src_set <= pending.keys())
  ]
  ready_sources = set(
      itertools.chain.from_iterable(r["source_keys"] for r in ready_rules)
  )
  still_needed = set(
      itertools.chain.from_iterable(src_set for src_set, _ in remaining_rules)
  )
  step_inputs = {
      **{k: tensors[k] for k in tensors.keys() - rule_sources},
      **{k: pending[k] for k in ready_sources},
  }
  for k in ready_sources - still_needed:
    pending.pop(k, None)
  return _execute_manifest(
      {**base_spec, "rules": ready_rules}, step_inputs, use_numpy=True
  )


def make_streaming_executor(
    manifest: RuleSpec,
    pending: dict[str, Any] | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, np.ndarray]]:
  """Returns a single-pass stateful host-NumPy executor per checkpoint load."""
  spec = parse_manifest(manifest)
  dropped = frozenset(spec["dropped_keys"])
  active_rules = [
      (frozenset(r["source_keys"]), r)
      for r in spec["rules"]
      if not dropped.intersection((*r["source_keys"], *r["target_keys"]))
  ]
  dropped_rule_sources = itertools.chain.from_iterable(
      r["source_keys"]
      for r in spec["rules"]
      if dropped.intersection((*r["source_keys"], *r["target_keys"]))
  )
  base_spec = {
      **spec,
      "dropped_keys": [*spec["dropped_keys"], *dropped_rule_sources],
  }
  rule_sources = frozenset(
      itertools.chain.from_iterable(src_set for src_set, _ in active_rules)
  )
  buf = {} if pending is None else pending
  return functools.partial(
      _step_streaming_rules, base_spec, active_rules, rule_sources, buf
  )
