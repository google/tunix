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

"""Constants for performance metrics."""

# Tags for Span / Event based data model.

STEP = "step"
MINI_BATCH = "mini_batch_step"
MICRO_BATCH = "micro_batch_step"
ROLE = "role"
GROUP_ID = "group_id"
PAIR_INDEX = "pair_index"
NAME = "NAME"
# Common Span / Event names.

DATA_LOADING = "data_loading"
# TODO(noghabi): Consider renaming to train, inference, etc.
ROLLOUT = "rollout"
WEIGHT_SYNC = "weight_sync"
REFERENCE_INFERENCE = "reference_inference"
OLD_ACTOR_INFERENCE = "old_actor_inference"
ADVANTAGE_COMPUTATION = "advantage_computation"
PEFT_TRAIN = "peft_train"
ENVIRONMENT = "environment"
QUEUE = "queue"
