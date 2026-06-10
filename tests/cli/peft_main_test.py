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

"""Tests for multi-controller JAX initialization in tunix.cli.peft_main."""

from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from tunix.cli import peft_main


class InitMultiControllerJaxTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # absl flags must be parsed before _PATHWAYS_BNS.value can be read.
    if not flags.FLAGS.is_parsed():
      flags.FLAGS(['peft_main'])

  def test_single_host_init_is_a_noop_on_value_error(self):
    # On a single host with no coordinator, jax.distributed.initialize() raises
    # ValueError ("coordinator_address should be defined."). That must be
    # swallowed so the run proceeds single-controller.
    with mock.patch.object(
        peft_main.jax.distributed,
        'initialize',
        side_effect=ValueError('coordinator_address should be defined.'),
    ) as mock_init:
      # Should not raise.
      peft_main._init_multi_controller_jax()
    mock_init.assert_called_once_with()

  def test_runtime_error_is_swallowed(self):
    with mock.patch.object(
        peft_main.jax.distributed,
        'initialize',
        side_effect=RuntimeError('no cluster'),
    ):
      peft_main._init_multi_controller_jax()  # must not raise

  def test_unexpected_error_propagates(self):
    # Only RuntimeError/ValueError are treated as "no cluster"; anything else is
    # a real failure and must not be hidden.
    with mock.patch.object(
        peft_main.jax.distributed,
        'initialize',
        side_effect=KeyError('boom'),
    ):
      with self.assertRaises(KeyError):
        peft_main._init_multi_controller_jax()

  def test_successful_initialize_is_called(self):
    with mock.patch.object(
        peft_main.jax.distributed, 'initialize'
    ) as mock_init:
      peft_main._init_multi_controller_jax()
    mock_init.assert_called_once_with()

  def test_pathways_path_skips_multi_controller_init(self):
    # When pathways_bns is set, Pathways (single-controller) is used and
    # jax.distributed.initialize() must NOT be invoked.
    with mock.patch.object(
        peft_main, '_setup_jax_pathways'
    ) as mock_pathways, mock.patch.object(
        peft_main, '_init_multi_controller_jax'
    ) as mock_mc_init, mock.patch.object(
        peft_main, 'PeftPipeline'
    ) as mock_pipeline, flagsaver.flagsaver(
        pathways_bns='some/bns/address'
    ):
      peft_main.main(['peft_main'])

    mock_pathways.assert_called_once_with('some/bns/address')
    mock_mc_init.assert_not_called()
    mock_pipeline.assert_called_once()

  def test_non_pathways_path_runs_multi_controller_init(self):
    with mock.patch.object(
        peft_main, '_init_multi_controller_jax'
    ) as mock_mc_init, mock.patch.object(
        peft_main, 'PeftPipeline'
    ) as mock_pipeline, flagsaver.flagsaver(
        pathways_bns=None
    ):
      peft_main.main(['peft_main'])

    mock_mc_init.assert_called_once_with()
    mock_pipeline.assert_called_once()


if __name__ == '__main__':
  absltest.main()
