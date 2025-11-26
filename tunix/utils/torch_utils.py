"""Common utilities for torch."""
import re


def torch_key_to_jax_key(mapping, source_key):
  """Convert torch key to jax key using the provided mapping."""
  subs = [
      (re.sub(pat, repl, source_key), reshape)
      for pat, (repl, reshape) in mapping.items()
      if re.match(pat, source_key)
  ]
  if len(subs) != 1:
    raise ValueError(f"Only one key should be found: {subs} for {source_key}")
  else:
    return subs[0]

