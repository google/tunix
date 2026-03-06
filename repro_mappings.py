import os
import sys

# Add the path to python path potentially if needed, but in google3 it should be handled by blaze/environment.
# However, for this script we might need to rely on the environment being set up correctly.

try:
  from tunix.models.qwen3 import model as qwen3_model
  from tunix.models.qwen3 import mapping_sglang_jax
  import tunix.models.qwen3 as qwen3_pkg

  print(f"Qwen3 module: {qwen3_model.Qwen3.__module__}")
  print(f"Qwen3 package: {qwen3_pkg}")
  if hasattr(qwen3_pkg, "BACKEND_MAPPINGS"):
    print(f"BACKEND_MAPPINGS keys: {list(qwen3_pkg.BACKEND_MAPPINGS.keys())}")
  else:
    print("BACKEND_MAPPINGS not found in qwen3 package")

  try:
    mapping = qwen3_model.Qwen3.mapping_for("sglang_jax")
    print("Mapping found for sglang_jax")
  except RuntimeError as e:
    print(f"Error: {e}")

except ImportError as e:
  print(f"ImportError: {e}")
except Exception as e:
  print(f"An error occurred: {e}")