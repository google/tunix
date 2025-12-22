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
    ignore_index: int = -100,
    reduction: str = 'mean',
    softcap: Optional[float] = None,
) -> jax.Array:
    # --- 1. Preparation ---
    if shift:
        x = embeddings[:, :-1, :]
        y = targets[:, 1:]
    else:
        x = embeddings
        y = targets

    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if y.ndim > 1:
        y = y.reshape(-1)

    # --- 2. Compute "Correct" Logits (Numerator) ---
    # We do this outside the loop to avoid gathering inside the scan
    valid_mask = (y != ignore_index)
    safe_y = jnp.where(valid_mask, y, 0)
    target_w = classifier[safe_y]
    
    # Use einsum for clarity: batch dot product
    logits_correct = jnp.einsum('bd,bd->b', x.astype(jnp.float32), target_w.astype(jnp.float32))

    if softcap is not None:
        logits_correct = jnp.tanh(logits_correct / softcap) * softcap

    # --- 3. Padding & Reshaping (The Speed Fix) ---
    vocab_size, hidden_dim = classifier.shape
    remainder = vocab_size % chunk_size
    
    if remainder > 0:
        pad_amt = chunk_size - remainder
        # Pad weights with 0.0 (value doesn't matter much as we mask later)
        classifier_padded = jnp.pad(classifier, ((0, pad_amt), (0, 0)), constant_values=0.0)
        
        # Create Mask: 1 for Real, 0 for Pad
        # We use int8 or bool to save memory bandwidth
        row_mask = jnp.concatenate(
            [jnp.ones((vocab_size,), dtype=jnp.bool_),
             jnp.zeros((pad_amt,), dtype=jnp.bool_)],
            axis=0
        )
    else:
        classifier_padded = classifier
        row_mask = jnp.ones((vocab_size,), dtype=jnp.bool_)

    # Reshape into chunks for efficient scanning
    # Weights: [Num_Chunks, Chunk_Size, Hidden]
    # Mask:    [Num_Chunks, Chunk_Size]
    classifier_chunks = classifier_padded.reshape(-1, chunk_size, hidden_dim)
    row_mask_chunks = row_mask.reshape(-1, chunk_size)

    # --- 4. Scan Body (The Memory Fix) ---
    init_val = (
        jnp.full((x.shape[0],), -jnp.inf, dtype=jnp.float32), # running_max
        jnp.zeros((x.shape[0],), dtype=jnp.float32)           # running_sum
    )

    def scan_body(carry, inputs):
        running_max, running_sum = carry
        w_chunk, mask_chunk = inputs 

        # 1. Matmul
        chunk_logits = jnp.dot(x, w_chunk.T)

        # 2. Softcap (Gemma 2)
        if softcap is not None:
            chunk_logits = jnp.tanh(chunk_logits / softcap) * softcap

        # 3. Apply Mask (Correctness Fix)
        # Broadcast mask_chunk (C,) to (B, C) automatically
        # Set padded positions to -inf so they contribute 0 to exp() sum
        chunk_logits = jnp.where(mask_chunk, chunk_logits, -jnp.inf)

        # 4. Online LogSumExp
        chunk_max = jnp.max(chunk_logits, axis=1).astype(jnp.float32)
        new_max = jnp.maximum(running_max, chunk_max)
        
        # Scale previous sum to new max
        exp_scale = jnp.exp(running_max - new_max)
        exp_scale = jnp.where(running_max == -jnp.inf, 0.0, exp_scale)
        
        # Add current chunk
        # Note: if chunk_logits is -inf, exp is 0.
        chunk_exp = jnp.sum(
            jnp.exp(chunk_logits.astype(jnp.float32) - new_max[:, None]), 
            axis=1
        )
        
        new_sum = running_sum * exp_scale + chunk_exp
        return (new_max, new_sum), None

    scan_body = jax.checkpoint(scan_body)

    # Scan over both weights and masks
    (final_max, final_sum), _ = jax.lax.scan(
        scan_body, init_val, (classifier_chunks, row_mask_chunks)
    )

    # --- 5. Final Loss ---
    lse = final_max + jnp.log(final_sum)
    loss = lse - logits_correct
    loss = jnp.where(valid_mask, loss, 0.0)
    
    if reduction == 'mean':
        total_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
        return jnp.sum(loss) / total_valid
    elif reduction == 'sum':
        return jnp.sum(loss)
    
    return loss
