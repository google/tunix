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

import time
from absl import logging
from flax import nnx
import jax
import orbax.checkpoint as ocp
# from pathwaysutils.persistence import helper
from concurrent import futures
import datetime

# original_write_one_array = helper.write_one_array
# def my_write_one_array(
#     location: str,
#     name: str,
#     value: jax.Array,
#     timeout: datetime.timedelta,
# ):
#   try:
#     # Call the original write_one_array function. This will have the same
#     # behavior as the original write_one_array function if there is no error.
#     write_future = original_write_one_array(location, name, value, timeout)
#   except TypeError as e:
#     # If there is an error, print the error, some information, and continue.
#     print("Found an array that failed to write")
#     print(f"TypeError: {e}")

#     print(f"{value=}")
#     print(f"{type(value)=}")
#     print(f"{location=}")
#     print(f"{name=}")
#     print(f"{timeout=}")
#     print("Skipping this failed array and continuing")

#     # Return a dummy future that is already done with no result
#     write_future = futures.Future()
#     write_future.set_result(None)

#   return write_future

# # Override the write_one_array with the my_write_one_array function.
# # This will be used by Orbax to write the arrays.
# helper.write_one_array = my_write_one_array


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
      checkpoint_storage_use_ocdbt: bool = False,
      checkpoint_storage_use_zarr3: bool = False,
  ):
    """Initializes the checkpoint manager.

    Args:
      root_directory: The root directory for the checkpoint manager. If None,
        the checkpoint manager will be disabled.
      options: The options for the checkpoint manager.
      checkpoint_storage_use_ocdbt: Whether to use OCDBT format.
      checkpoint_storage_use_zarr3: Whether to use Zarr3 format.
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      item_handlers = {
          "model_params": ocp.PyTreeCheckpointHandler(
              use_ocdbt=checkpoint_storage_use_ocdbt,
              use_zarr3=checkpoint_storage_use_zarr3,
          )
      }
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
  ) -> bool:
    """Saves the params for the given step.

    Args:
      step: The step to save the params for.
      model: The model to save the params for.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save the checkpoint regardless of the save decision
        policy.

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
    params_dict_with_jax_array = {
        k: v.value for k, v in nnx.to_flat_state(params)
    }
    logging.info(
        "Checkpointing params: %s", list(params_dict_with_jax_array.keys())
    )
    checkpoint_args = ocp.args.PyTreeSave(
        item=params_dict_with_jax_array,
        save_args=jax.tree.map(
            lambda _: ocp.SaveArgs(), params_dict_with_jax_array
        ),
    )
    return self._checkpoint_manager.save(
        step,
        args=ocp.args.Composite(model_params=checkpoint_args),
        force=force,
    )

  def maybe_restore(
      self,
      model: nnx.Module,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> int:
    """Restores the params from the latest checkpoint if available and updates the model provided.

    Args:
      model: The model to restore the params for.
      step: The step to restore the params from. If None, the latest step will
        be used.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      The step of the restored checkpoint or 0 if no checkpoint is available.
    """
    restore_start = time.time()
    if self._checkpoint_manager is None:
      return 0
    if step is None:
      step = self._checkpoint_manager.latest_step()
      # If no checkpoint is available, return 0.
      if step is None:
        return 0
    # Load the params from the checkpoint.
    if restore_only_lora_params:
      abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
      abstract_params = nnx.state(model)

    abstract_params_dict = dict(nnx.to_flat_state(abstract_params))
    abstract_params_dict_with_jax_array = {
        k: v.value for k, v in nnx.to_flat_state(abstract_params)
    }
    def map_to_pspec(data):
      return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)

    restore_args_dict = jax.tree_util.tree_map(
        map_to_pspec, abstract_params_dict_with_jax_array
    )
    checkpoint_args = ocp.args.PyTreeRestore(
        item=abstract_params_dict_with_jax_array, restore_args=restore_args_dict
    )
    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(model_params=checkpoint_args),
    )
    for k, v in ckpt.model_params.items():
      abstract_params_dict[k].value = v
    # Update the model state with params from the restored checkpoint.
    nnx.update(model, nnx.from_flat_state(abstract_params_dict))
    logging.info(
        "Restored params from step: %d in %.3f seconds",
        step,
        time.time() - restore_start,
    )
    return step

  def close(self):
    """Closes the checkpoint manager."""
    if self._checkpoint_manager is None:
      return
    self._checkpoint_manager.close()