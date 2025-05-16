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
"""Module containing the distillation strategies."""

# pylint: disable=g-importing-member

from tunix.distillation.strategies.attention import AttentionProjectionStrategy
from tunix.distillation.strategies.attention import AttentionTransferStrategy
from tunix.distillation.strategies.base_strategy import BaseStrategy
from tunix.distillation.strategies.base_strategy import ModelForwardCallable
from tunix.distillation.strategies.feature_pooling import FeaturePoolingStrategy
from tunix.distillation.strategies.feature_projection import FeatureProjectionStrategy
from tunix.distillation.strategies.logit import LogitStrategy
