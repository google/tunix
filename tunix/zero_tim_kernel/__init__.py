# Copyright 2026 Google LLC
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
"""Zero-TIM canonical kernels with bitwise reproducibility."""

from __future__ import annotations

import os

# Zero-TIM canonical kernels require XLA with excess precision disabled
# (root cause R3): without it XLA may keep bf16 intermediates in f32
# across fused elementwise chains, breaking bitwise determinism and parity.
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_allow_excess_precision=" not in _flags:
  os.environ["XLA_FLAGS"] = f"{_flags} --xla_allow_excess_precision=false".strip()
