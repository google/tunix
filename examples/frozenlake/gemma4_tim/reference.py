"""Ordinary real-Tunix E2B oracle. Never registered as canonical trainer C."""

import jax
import jax.numpy as jnp


def kv_producers(layer_types, shared_layers):
  if not layer_types or not 0 <= shared_layers < len(layer_types):
    raise ValueError("invalid shared-KV layer count")
  if set(layer_types) - {"local", "global"}:
    raise ValueError("unknown attention type")
  boundary = len(layer_types) - shared_layers
  last, result = {}, []
  for index, kind in enumerate(layer_types):
    if index < boundary:
      last[kind] = index
      result.append(index)
    else:
      if kind not in last:
        raise ValueError("shared attention has no producer of its kind")
      result.append(last[kind])
  return tuple(result)


E2B_LAYER_TYPES = tuple("global" if (i + 1) % 5 == 0 else "local" for i in range(35))
E2B_KV_PRODUCERS = kv_producers(E2B_LAYER_TYPES, 20)


def action_logps(model, tokens, valid_mask, action_mask, *, temperature=.7,
                 cache=None, chunk_offset=0):
  """Full-recompute t-1 -> t oracle, with explicit validity/EOS semantics.

  This helper's processed logsoftmax is mathematical reference only. Engine
  B must use the actual engine's sampling transforms, not this host replica.
  """
  if cache is not None or chunk_offset != 0:
    raise ValueError("ordinary Gemma oracle requires full recompute")
  if (tokens.ndim != 2 or tokens.shape[1] < 2 or valid_mask.shape != tokens.shape
      or action_mask.shape != tokens.shape or valid_mask.dtype != jnp.bool_
      or action_mask.dtype != jnp.bool_ or not jnp.issubdtype(tokens.dtype, jnp.integer)
      or temperature <= 0):
    raise ValueError("invalid reference tokens, masks or temperature")
  positions = jnp.maximum(jnp.cumsum(valid_mask, axis=1) - 1, 0)
  causal = jnp.tril(jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))
  attention = causal[None] & valid_mask[:, None, :]
  logits, _ = model(tokens, positions=positions, attention_mask=attention)
  logps = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32) / temperature, axis=-1)
  selected = jnp.take_along_axis(logps, tokens[:, 1:, None], axis=-1)[..., 0]
  scored = action_mask[:, 1:] & valid_mask[:, 1:] & valid_mask[:, :-1]
  return jnp.where(scored, selected, 0)
