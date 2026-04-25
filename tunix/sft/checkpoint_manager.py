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

import functools
import os
import time
from typing import Any

from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_options


class CheckpointManager:
  """Checkpoint manager for PEFT."""

  def __init__(
      self,
      root_directory: str | None = None,
      options: checkpoint_options.CheckpointingOptions | None = None,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
    """
    self._checkpointer: ocp.v1.training.Checkpointer | None = None
    self._options = checkpoint_options.resolve_checkpointing_defaults(
        options
    )
    if root_directory is not None:
      with self.context:
        self._checkpointer = ocp.v1.training.Checkpointer(
            root_directory,
            save_decision_policy=self._options.save_decision_policy,
            preservation_policy=self._options.preservation_policy,
            step_name_format=self._options.step_name_format,
        )

  @functools.cached_property
  def context(self) -> ocp.v1.Context:
    """Returns the orbax context."""
    # When using Pathways, the checkpoint manager only supports persistence
    # APIs now.
    if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
      return ocp.v1.Context(
          array_options=ocp.v1.options.ArrayOptions(
              saving=ocp.v1.options.ArrayOptions.Saving(
                  use_ocdbt=False,
                  use_zarr3=False,
              )
          ),
          async_options=self._options.async_options,
      )
    else:
      return ocp.v1.Context(async_options=self._options.async_options)

  def latest_step(self) -> int | None:
    """Returns the latest step."""
    if self._checkpointer is None or self._checkpointer.latest is None:
      return None
    return self._checkpointer.latest.step

  def save(
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      optimizer: The optimizer to save the params for. If None, the optimizer
        will not be saved.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.
      custom_metadata: Custom metadata to save with the checkpoint.

    Returns:
      Whether the checkpoint save operation was successful if syncronous,
      otherwise whether the save operation was initiated.
    """
    if self._checkpointer is None:
      return False
    if save_only_lora_params:
      params = nnx.state(model, nnx.LoRAParam)
    else:
      params = nnx.state(model)

    if optimizer is not None:
      checkpointables = {
          'model_params': params,
          'optimizer_state': nnx.state(optimizer, nnx.optimizer.OptState),
      }
    else:
      checkpointables = {
          'model_params': params,
      }
    with self.context:
      if self._options.enable_async_checkpointing:
        self._checkpointer.save_checkpointables_async(
            step,
            checkpointables,
            force=force,
            custom_metadata=custom_metadata,
        )
        return True
      return self._checkpointer.save_checkpointables(
          step,
          checkpointables,
          force=force,
          custom_metadata=custom_metadata,
      )

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> tuple[int, Any]:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      optimizer: The optimizer to restore the params for. If None or if
        optimizer state is not found in the checkpoint, the optimizer will not
        be restored.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      A tuple of the step of the restored checkpoint or 0 if no checkpoint is
      available, and the custom metadata.

    Raises:
      RuntimeError: If the checkpoint cannot be restored.
    """
    restore_start = time.time()
    if self._checkpointer is None:
      return 0, {}
    if step is None:
      step = self.latest_step()
      if step is None:
        return 0, {}

    with self.context:
      metadata = self._checkpointer.checkpointables_metadata(step)

    if restore_only_lora_params:
      model_params_state = nnx.state(model, nnx.LoRAParam)
    else:
      model_params_state = nnx.state(model)
    abstract_checkpointables = {'model_params': model_params_state}

    def fix_sharding(state):
      # Scalar values in optimizer states like step and count is initialized as
      # SingleDeviceSharding, which will fail if optimizer is sharded. To fix
      # it, we will replicate the scalar values.
      mesh = next(
          (
              x.sharding.mesh
              for x in jax.tree_util.tree_leaves(state)
              if getattr(x, 'sharding', None)
              and isinstance(x.sharding, jax.sharding.NamedSharding)
          ),
          None,
      )

      if mesh is None:
        return state

      target_shardings = nnx.get_named_sharding(state, mesh)
      return jax.tree_util.tree_map(
          lambda x, shd: jax.ShapeDtypeStruct(
              getattr(x, 'shape', ()),
              getattr(x, 'dtype', jax.numpy.dtype(type(x))),
              sharding=shd,
          ),
          state,
          target_shardings,
      )

    if (
        optimizer is not None
        and metadata is not None
        and 'optimizer_state' in metadata.metadata
    ):
      optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
      abstract_checkpointables['optimizer_state'] = fix_sharding(
          optimizer_state
      )
      with self.context:
        restored_checkpointables = self._checkpointer.load_checkpointables(
            step,
            abstract_checkpointables,
        )
      nnx.update(optimizer, restored_checkpointables['optimizer_state'])
    else:
      with self.context:
        restored_checkpointables = self._checkpointer.load_checkpointables(
            step,
            abstract_checkpointables,
        )
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, restored_checkpointables['model_params'])
    logging.info(
        'Restored params from step: %d in %.3f seconds',
        step,
        time.time() - restore_start,
    )
    custom_metadata = metadata.custom_metadata if metadata else {}
    return step, custom_metadata

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpointer is None:
      return
    self._checkpointer.close()
