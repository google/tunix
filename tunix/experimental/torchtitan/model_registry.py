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

"""Model registry and configuration resolution for TorchTitan training."""

from __future__ import annotations

import dataclasses
import json
from typing import Final

from etils import epath
from torchtitan.experiments.tpu import qwen3 as tpu_qwen3
from torchtitan.models import llama3 as llama3_models
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec

_QWEN3_FLAVOR_MAP: Final[dict[str, str]] = {
    **{k.lower().replace("_", "-"): k for k in tpu_qwen3.qwen3_configs},
    **{k.lower(): k for k in tpu_qwen3.qwen3_configs},
    "30ba3b": "30B-A3B",
}
_LLAMA3_FLAVOR_MAP: Final[dict[str, str]] = {
    **{k.lower().replace("_", "-"): k for k in llama3_models.llama3_configs},
    **{k.lower(): k for k in llama3_models.llama3_configs},
}
_HF_VARIANT_SUFFIXES: Final[tuple[str, ...]] = (
    "-instruct",
    "-base",
    "-bf16",
    "-fp8",
)


def _strip_hf_suffix(text: str) -> str:
  for suffix in _HF_VARIANT_SUFFIXES:
    if text.endswith(suffix):
      return text[: -len(suffix)]
  return text


def fit_rope_to_context(
    model_config: BaseModel.Config | None, max_context_length: int
) -> None:
  """Traverses model_config and clamps any rope.max_context_length down to max_context_length."""
  if model_config is None or max_context_length <= 0:
    return
  for _, rope, parent, attr in model_config.traverse(RoPE.Config):
    if not isinstance(rope, RoPE.Config):
      continue
    if parent is None or rope.max_context_length <= max_context_length:
      continue
    new_rope = dataclasses.replace(rope, max_context_length=max_context_length)
    if isinstance(parent, list):
      assert isinstance(attr, int)
      parent[attr] = new_rope
    else:
      assert isinstance(attr, str)
      setattr(parent, attr, new_rope)


def resolve_model_name_and_flavor(
    model_name_or_path: str, explicit_flavor: str = ""
) -> tuple[str, str]:
  """Normalizes model_name_or_path to canonical (model_name, flavor)."""
  norm = model_name_or_path.strip()
  if not norm and not explicit_flavor:
    raise ValueError("model_name_or_path cannot be empty.")

  if explicit_flavor:
    flavor_key = explicit_flavor.strip().lower().replace("_", "-")
    name_lower = norm.lower().replace("_", "-")
    if "qwen3" in name_lower or (
        not name_lower and flavor_key in _QWEN3_FLAVOR_MAP
    ):
      if flavor_key not in _QWEN3_FLAVOR_MAP:
        raise ValueError(
            f"Unsupported Qwen3 flavor {explicit_flavor!r}; expected one of"
            f" {sorted(tpu_qwen3.qwen3_configs)}."
        )
      return "qwen3", _QWEN3_FLAVOR_MAP[flavor_key]
    if "llama" in name_lower:
      if flavor_key not in _LLAMA3_FLAVOR_MAP:
        raise ValueError(
            f"Unsupported Llama3 flavor {explicit_flavor!r}; expected one of"
            f" {sorted(llama3_models.llama3_configs)}."
        )
      return "llama3", _LLAMA3_FLAVOR_MAP[flavor_key]
    raise ValueError(
        f"Unsupported model_name_or_path={model_name_or_path!r} with"
        f" explicit_flavor={explicit_flavor!r}."
    )

  base_name = norm.rstrip("/").split("/")[-1]
  base_lower = _strip_hf_suffix(base_name.lower().replace("_", "-"))

  if "qwen3" in base_lower or "qwen-3" in base_lower:
    normalized_qwen = base_lower.replace("qwen-3", "qwen3")
    idx = normalized_qwen.find("qwen3")
    raw_flavor = normalized_qwen[idx + len("qwen3") :].lstrip("-")
    if not raw_flavor:
      raw_flavor = "debugmodel"
    if raw_flavor not in _QWEN3_FLAVOR_MAP:
      raise ValueError(
          f"Unsupported Qwen3 flavor in {model_name_or_path!r}; expected one of"
          f" {sorted(tpu_qwen3.qwen3_configs)}."
      )
    return "qwen3", _QWEN3_FLAVOR_MAP[raw_flavor]

  if "llama" in base_lower:
    for token, canonical in (
        ("debugmodel", "debugmodel"),
        ("405b", "405B"),
        ("70b", "70B"),
        ("8b", "8B"),
        ("3b", "3B"),
        ("1b", "1B"),
    ):
      if token in base_lower:
        return "llama3", canonical
    raise ValueError(
        f"Unsupported Llama3 flavor in {model_name_or_path!r}; expected one of"
        f" {sorted(llama3_models.llama3_configs)}."
    )

  if base_lower in _QWEN3_FLAVOR_MAP:
    return "qwen3", _QWEN3_FLAVOR_MAP[base_lower]

  raise ValueError(
      f"Unknown or unsupported model_name_or_path: {model_name_or_path!r}"
  )


def resolve_model_spec(
    model_name: str,
    flavor: str,
    *,
    attn_backend: str = "sdpa",
    max_context_length: int = 0,
    hf_assets_path: str = "",
) -> ModelSpec:
  """Resolves and configures a ModelSpec for the given model_name and flavor."""
  canonical_name, canonical_flavor = resolve_model_name_and_flavor(
      model_name, explicit_flavor=flavor
  )
  if canonical_name == "qwen3":
    model_spec = tpu_qwen3.model_registry(
        canonical_flavor, attn_backend=attn_backend
    )
  elif canonical_name == "llama3":
    model_spec = llama3_models.model_registry(
        canonical_flavor, attn_backend=attn_backend
    )
  else:
    raise ValueError(f"Unsupported model_name: {model_name!r}")

  if hf_assets_path:
    cfg_file = epath.Path(hf_assets_path) / "config.json"
    if cfg_file.exists():
      hf_cfg = json.loads(cfg_file.read_text())
      if "tie_word_embeddings" in hf_cfg and isinstance(
          model_spec.model,
          (tpu_qwen3.Qwen3Model.Config, llama3_models.Llama3Model.Config),
      ):
        model_spec.model.enable_weight_tying = bool(
            hf_cfg["tie_word_embeddings"]
        )

  fit_rope_to_context(model_spec.model, max_context_length)
  return model_spec
