import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
from flax import nnx
from jax.typing import ArrayLike

def linear_cross_entropy(
    embeddings: jax.Array,
    classifier: jax.Array,
    targets: jax.Array,
    shift: bool = False,
    chunk_size: int = 4096,
    return_lse: bool = False,
    ignore_index: int = -100,
    reduction: str = 'mean',
    softcap: Optional[float] = None,
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    if shift:
        if embeddings.ndim != 3:
            raise ValueError("Input must be [Batch, Seq, Hidden] when shift=True")
        x = embeddings[:, :-1, :] 
        y = targets[:, 1:]        
    else:
        x = embeddings
        y = targets

    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if y.ndim > 1:
        y = y.reshape(-1)

    hidden_dim = x.shape[-1]
    vocab_size = classifier.shape[0]
    
    chunk_size = min(chunk_size, vocab_size)
    
    valid_mask = (y != ignore_index)
    safe_y = jnp.where(valid_mask, y, 0)
    target_w = classifier[safe_y]
    
    logits_correct = jnp.sum(x.astype(jnp.float32) * target_w.astype(jnp.float32), axis=-1)

    if softcap is not None:
        logits_correct = jnp.tanh(logits_correct / softcap) * softcap

    num_full_chunks = vocab_size // chunk_size
    
    init_val = (
        jnp.full((x.shape[0],), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((x.shape[0],), dtype=jnp.float32)
    )

    def scan_body(carry, chunk_idx):
        running_max, running_sum = carry
        start = chunk_idx * chunk_size
        
        w_chunk = jax.lax.dynamic_slice(
            classifier, (start, 0), (chunk_size, hidden_dim)
        )
        
        chunk_logits = jnp.dot(x, w_chunk.T)

        if softcap is not None:
            chunk_logits = jnp.tanh(chunk_logits / softcap) * softcap

        chunk_max = jnp.max(chunk_logits, axis=1).astype(jnp.float32)
        new_max = jnp.maximum(running_max, chunk_max)
        
        exp_scale = jnp.exp(running_max - new_max)
        exp_scale = jnp.where(running_max == -jnp.inf, 0.0, exp_scale)
        
        chunk_exp = jnp.sum(
            jnp.exp(chunk_logits.astype(jnp.float32) - new_max[:, None]), 
            axis=1
        )
        
        new_sum = running_sum * exp_scale + chunk_exp
        return (new_max, new_sum), None

    # Apply gradient checkpointing (remat) to scan_body.
    # This ensures that intermediate logits are not stored for the backward pass,
    # preventing OOMs by re-computing them on demand.
    scan_body = jax.checkpoint(scan_body)

    (final_max, final_sum), _ = jax.lax.scan(
        scan_body, init_val, jnp.arange(num_full_chunks)
    )

    remainder = vocab_size % chunk_size
    if remainder > 0:
        start = num_full_chunks * chunk_size
        w_rem = jax.lax.dynamic_slice(
            classifier, (start, 0), (remainder, hidden_dim)
        )
        
        logits_rem = jnp.dot(x, w_rem.T)

        if softcap is not None:
            logits_rem = jnp.tanh(logits_rem / softcap) * softcap

        rem_max = jnp.max(logits_rem, axis=1).astype(jnp.float32)
        new_global_max = jnp.maximum(final_max, rem_max)
        
        exp_scale = jnp.exp(final_max - new_global_max)
        exp_scale = jnp.where(final_max == -jnp.inf, 0.0, exp_scale)
        
        rem_exp = jnp.sum(
            jnp.exp(logits_rem.astype(jnp.float32) - new_global_max[:, None]), 
            axis=1
        )
        
        final_sum = final_sum * exp_scale + rem_exp
        final_max = new_global_max

    lse = final_max + jnp.log(final_sum)

    loss = lse - logits_correct
    loss = jnp.where(valid_mask, loss, 0.0)
    
    if reduction == 'mean':
        total_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
        final_loss = jnp.sum(loss) / total_valid
    elif reduction == 'sum':
        final_loss = jnp.sum(loss)
    elif reduction == 'none':
        final_loss = loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    if return_lse:
        return final_loss, lse
    return final_loss

def cce_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
  """CCE loss function for PEFT training on TPU."""
  
  # Get embeddings
  embeddings, _ = model(input_tokens, positions, None, attention_mask, return_embeddings=True)
  
  # Manual Shifting (Correct)
  embeddings = embeddings[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Access shared embedding weights
  classifier = model.embedder.input_embedding.value
  
  # Explicitly stop gradients on the classifier weights. 
  # This prevents JAX from allocating a massive buffer to accumulate 
  # weight gradients inside the scan loop.
  #classifier = jax.lax.stop_gradient(classifier)
  
  # CCE targets
  cce_targets = jnp.where(target_mask > 0, target_tokens, -100)

  # Check for softcap config (Gemma 2 support)
  # Standard Gemma 2 uses 30.0 for final logits, but check your specific config
  softcap = getattr(model.config, 'final_logit_softcap', None)

  # Compute loss
  loss_sum = linear_cross_entropy(
      embeddings, 
      classifier, 
      cce_targets, 
      reduction='sum',
      chunk_size=8192,
      softcap=softcap
  )
  
  token_count = jnp.sum(target_mask)
  # Return mean loss to be compatible with standard trainer expectations (perplexity, gradient scale)
  return loss_sum / jnp.maximum(token_count, 1.0), token_count