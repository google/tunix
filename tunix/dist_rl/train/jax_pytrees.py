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

"""Registers the framework-neutral trainer payloads as JAX pytrees.

`tunix.dist_rl.common.datatypes` keeps `TrainerPayload` and its subclasses as
plain dataclasses so the orchestrator (numpy-only control plane) never imports
JAX. JAX trainers, however, trace, shard and `jax.tree.map` over payloads, so
they need the classes registered as pytrees. Importing this module performs
that registration exactly once per process; field roles come from the
dataclass metadata (`datatypes.STATIC_FIELD_METADATA` marks `meta_fields`,
everything else is a data leaf), so there is no second field list to keep in
sync.

Import it from every JAX entry point that handles payloads before the first
tree operation. `peft_trainer_v2` does so directly; `tunix.sft.utils` also
imports it so that engines which only take `LossOutput` / `WeightedMetric` from
Tunix (MaxText's `training_engine`) get the registration without a code change.
"""

import jax
from tunix.dist_rl.common import datatypes

# `register_dataclass` without explicit field lists reads
# `dataclasses.field(metadata={"static": True})`, which is exactly what
# `datatypes.STATIC_FIELD_METADATA` sets, and uses the C++ dataclass
# flatten/unflatten fast path.
TRAINER_PAYLOAD_TYPES: tuple[type[datatypes.TrainerPayload], ...] = (
    datatypes.TrainerPayload,
    datatypes.SFTTrainerPayload,
    datatypes.RLTrainerPayload,
    datatypes.LogprobsScoringPayload,
)

for _payload_type in TRAINER_PAYLOAD_TYPES:
  jax.tree_util.register_dataclass(_payload_type)
