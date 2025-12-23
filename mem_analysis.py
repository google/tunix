# Memory analysis for Gemma2-2B training
# TPUv5e-8 has 8 chips with 16GB HBM each = 16GB per chip

batch = 64
seq = 2048
hidden = 2304
heads = 8
kv_heads = 4
head_dim = 256
layers = 26
ffw_dim = 9216
vocab = 256128
bf16_bytes = 2

print('=== Gemma2-2B Memory Analysis ===')
print(f'Batch: {batch}, Seq: {seq}, Hidden: {hidden}')
print(f'Heads: {heads}, KV Heads: {kv_heads}, Head Dim: {head_dim}')
print(f'Layers: {layers}, FFW Dim: {ffw_dim}')
print()

# Model parameters (static, sharded across devices)
embedding_params = vocab * hidden * bf16_bytes
layer_params = (
    # attention projections
    hidden * heads * head_dim + # Q
    hidden * kv_heads * head_dim * 2 + # K, V
    heads * head_dim * hidden + # O
    # FFW 
    hidden * ffw_dim * 2 + # gate & up
    ffw_dim * hidden # down
) * bf16_bytes
total_params = (embedding_params + layer_params * layers) / 1e9
print(f'Model params: ~{total_params:.2f} GB')
print()

# Activation memory per sample (the problem area)
# For GQA with 4 kv_heads and 8 heads (2 groups per kv head):
# logits shape: [B, T, K, G, S] where K=kv_heads, G=heads/kv_heads, S=seq
groups_per_kv = heads // kv_heads
attn_scores_shape = (batch, seq, kv_heads, groups_per_kv, seq)
attn_scores_mem = batch * seq * kv_heads * groups_per_kv * seq * bf16_bytes
print(f'Attention scores per layer: {attn_scores_mem / 1e9:.3f} GB')
print(f'  Shape: {attn_scores_shape}')

# FFW activation: [B, seq, ffw_dim]
ffw_act = batch * seq * ffw_dim * bf16_bytes
print(f'FFW activation per layer: {ffw_act / 1e9:.3f} GB')

# Hidden states per layer
hidden_act = batch * seq * hidden * bf16_bytes
print(f'Hidden states per layer: {hidden_act / 1e9:.3f} GB')
print()

# Per-layer peak during attention backward (the issue!)
q_proj = batch * seq * heads * head_dim * bf16_bytes
kv_proj = batch * seq * kv_heads * head_dim * bf16_bytes * 2
attn_out = batch * seq * heads * head_dim * bf16_bytes
print(f'Q proj: {q_proj / 1e9:.3f} GB')
print(f'K+V proj: {kv_proj / 1e9:.3f} GB')
print(f'Attn output: {attn_out / 1e9:.3f} GB')
print(f'Total attn activations: {(q_proj + kv_proj + attn_out + attn_scores_mem) / 1e9:.3f} GB')
print()

# The 512MB comes from: bf16[8,4,2048,2,2048]
attention_intermediate = 8 * 4 * 2048 * 2 * 2048 * bf16_bytes
print(f'Large attention intermediate (error shows): {attention_intermediate / 1e9:.3f} GB')
print()

# Gradient accumulation state
grad_accum = total_params  # Same size as params
print(f'Gradient accumulation buffer: {grad_accum:.2f} GB')

print()
print('=== Summary ===')
print(f'The error shows 30GB vs 16GB available.')
print(f'Key issue: attention intermediates during backward pass.')
print(f'Even with remat on attention, the backward pass still creates large intermediates.')
print()
print('=== The real issue ===')
print('The REAL batch size is 64, but data is sharded across 8 TPU chips.')
print(f'Per-chip batch size: {batch // 8} samples')
per_chip_batch = batch // 8
per_chip_attn_scores = per_chip_batch * seq * kv_heads * groups_per_kv * seq * bf16_bytes
print(f'Per-chip attention scores: {per_chip_attn_scores / 1e9:.3f} GB')
print()
print('But during backward pass with JVP, multiple intermediate tensors are needed.')
print('The error shows allocation from jvp(attention) which means the backward pass.')
