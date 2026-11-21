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
"""Declarative 12-Op Weight Conversion Protocol and algebraic inversion.

This module is an explicit neutral boundary (`deps = []`, zero ML framework
imports) defining the self-invertible 12-Op weight-conversion schema and
algebraic manifest inversion (`invert_op`, `invert_manifest`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import math
from typing import Any, Final, TypeAlias, TypedDict

SCHEMA_VERSION: Final[str] = "weight_mapping"

ShardingAxisSpec: TypeAlias = str | list[str] | tuple[str, ...] | None

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
