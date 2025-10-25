# Copyright 2025 Google LLC
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

"""Checkpoint manager for PEFT."""

import os
import time
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp

_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    max_to_keep=3,
)


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      # When using Pathways, the checkpoint manager only supports persistence
      # APIs now.
      if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
        }
        logging.info('Using persistence APIs for checkpointing with Pathways.')
      else:
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(),
        }
      item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_handlers=item_handlers,
          options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
      )

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if self._checkpoint_manager is None:
      return None
    return self._checkpoint_manager.latest_step()

  def save(
      self,
      step: int,
      model: nnx.Module,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint was saved.
    """
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)
    checkpoint_args = ocp.args.PyTreeSave(
        item=params, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params)
    )
    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            model_params=checkpoint_args,
        ),
        custom_metadata=custom_metadata or {},
        force=force,
    )

  def maybe_restore(
      self,
      model: nnx.Module,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> Tuple[int, dict[str, Any]]:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if self._checkpoint_manager is None:
      return 0, {}
    if step is None:
      step = self._checkpoint_manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0, {}
    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)

    def map_to_pspec(data):
      return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)

    restore_args_dict = jax.tree_util.tree_map(map_to_pspec, abstract_params)
    checkpoint_args = ocp.args.PyTreeRestore(
        item=abstract_params, restore_args=restore_args_dict
    )

    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            model_params=checkpoint_args,
        ),
    )
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, ckpt.model_params)
    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    metadata = self._checkpoint_manager.metadata(step)
    custom_metadata = metadata.custom_metadata if metadata else {}
    return step, custom_metadata

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()
