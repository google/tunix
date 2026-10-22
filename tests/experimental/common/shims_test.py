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

"""Tests for temporary cross-repo compatibility shims in tunix.experimental."""

import importlib
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import datatypes as dist_datatypes
from tunix.dist_rl.train import abstract_trainer as dist_abstract_trainer
from tunix.dist_rl.weight_sync import raiden_synchronizer as dist_raiden_synchronizer
from tunix.experimental.common import datatypes as exp_datatypes
from tunix.experimental.train import abstract_trainer as exp_abstract_trainer
from tunix.experimental.weight_sync import raiden_synchronizer as exp_raiden_synchronizer


class ExperimentalShimsTest(absltest.TestCase):

  def test_module_alias_identity(self):
    self.assertIs(exp_datatypes, dist_datatypes)
    self.assertIs(exp_abstract_trainer, dist_abstract_trainer)
    self.assertIs(exp_raiden_synchronizer, dist_raiden_synchronizer)
    self.assertIs(
        importlib.import_module("tunix.experimental.common.datatypes"),
        dist_datatypes,
    )
    self.assertIs(
        importlib.import_module("tunix.experimental.train.abstract_trainer"),
        dist_abstract_trainer,
    )
    self.assertIs(
        importlib.import_module(
            "tunix.experimental.weight_sync.raiden_synchronizer"
        ),
        dist_raiden_synchronizer,
    )

  def test_mock_patch_via_legacy_path_mutates_dist_rl_module(self):
    sentinel_cls = object()
    with mock.patch(
        "tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer",
        sentinel_cls,
    ):
      self.assertIs(dist_raiden_synchronizer.RaidenSynchronizer, sentinel_cls)

    with mock.patch(
        "tunix.experimental.common.datatypes.TrainerPayload",
        sentinel_cls,
    ):
      self.assertIs(dist_datatypes.TrainerPayload, sentinel_cls)

    with mock.patch(
        "tunix.experimental.train.abstract_trainer.AbstractTrainer",
        sentinel_cls,
    ):
      self.assertIs(dist_abstract_trainer.AbstractTrainer, sentinel_cls)


if __name__ == "__main__":
  absltest.main()
