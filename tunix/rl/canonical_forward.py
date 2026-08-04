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

"""Fail-closed registry for the differentiable canonical Qwen3 forward."""

from __future__ import annotations

import os
from typing import Any, Protocol


ENV = "CANON_ENGINE_MODULE_C"


class CanonicalForwardError(RuntimeError):
  """Raised when canonical C is requested without a valid implementation."""


class CanonicalForwardAdapter(Protocol):
  implementation_id: str
  is_engine_module: bool
  supports_value_and_grad: bool

  def compute_per_token_logps(self, **kwargs: Any) -> Any:
    ...


_adapter: CanonicalForwardAdapter | None = None


def enabled() -> bool:
  return os.environ.get(ENV, "") == "1"


def register(adapter: CanonicalForwardAdapter) -> None:
  """Register the one canonical adapter before tracing any train function."""
  global _adapter
  if not enabled():
    raise CanonicalForwardError(
        f"refusing to register canonical C while {ENV} is not 1"
    )
  implementation_id = getattr(adapter, "implementation_id", "")
  if not implementation_id:
    raise CanonicalForwardError("canonical C adapter has no implementation_id")
  if not getattr(adapter, "is_engine_module", False):
    raise CanonicalForwardError(
        "canonical C adapter does not attest is_engine_module=True"
    )
  if not getattr(adapter, "supports_value_and_grad", False):
    raise CanonicalForwardError(
        "canonical C adapter does not attest supports_value_and_grad=True"
    )
  if _adapter is not None and _adapter is not adapter:
    raise CanonicalForwardError(
        "a different canonical C adapter is already registered in this process"
    )
  _adapter = adapter


def require_registered() -> CanonicalForwardAdapter:
  if not enabled():
    raise CanonicalForwardError(f"canonical C requested but {ENV} is not 1")
  if _adapter is None:
    raise CanonicalForwardError(
        "canonical C requested but no tpu_inference engine-module adapter was "
        "registered; refusing native NNX fallback"
    )
  return _adapter


def compute_per_token_logps(**kwargs: Any) -> Any:
  return require_registered().compute_per_token_logps(**kwargs)


def attestation() -> dict[str, Any]:
  adapter = require_registered()
  return {
      "implementation_id": adapter.implementation_id,
      "is_engine_module": bool(adapter.is_engine_module),
      "supports_value_and_grad": bool(adapter.supports_value_and_grad),
  }


def _clear_for_test() -> None:
  global _adapter
  _adapter = None
