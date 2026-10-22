"""RPC utilities for the Tunix Orchestrator and Workers."""

import dataclasses
import numpy as np


def validate_wire_safe(obj: object) -> None:
  """Raise TypeError if a wire payload holds a device array.

  Wire payloads cross process boundaries via cloudpickle and must carry numpy
  arrays, not device arrays (e.g. ``jax.Array``): a sharded device array cannot
  be reconstructed in a process that does not share its source mesh. This walks
  nested dataclasses, lists/tuples, and dict values, raising on the first
  array-like value that is not a ``numpy.ndarray``.

  Args:
    obj: The object to validate.

  Raises:
    TypeError: If the object contains a device array (e.g. jax.Array) that can't
    be transported across process boundaries.
  """
  stack = [obj]
  visited = set()

  while stack:
    value = stack.pop()

    val_id = id(value)
    if val_id in visited:
      continue
    visited.add(val_id)

    if isinstance(value, np.ndarray):
      continue

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      stack.extend(getattr(value, f.name) for f in dataclasses.fields(value))
    elif isinstance(value, dict):
      stack.extend(value.keys())
      stack.extend(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
      stack.extend(value)
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
      raise TypeError(
          "wire payload contains a non-numpy array of type "
          f"{type(value).__module__}.{type(value).__qualname__}; convert device"
          " arrays (e.g. jax.Array) to numpy before serialization"
      )
