# Detailed memory analysis for TPUv5e-8 OOM
# TPUv5e-8: 8 chips, 16GB HBM per chip

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
num_chips = 8
grad_accum_steps = 12

print('=== Per-Chip Memory Analysis ===')
per_chip_batch = batch // num_chips
print(f'Per-chip batch: {per_chip_batch}')
print(f'HBM per chip: 16 GB')
print()

# Model parameters (sharded with FSDP)
embedding_params = vocab * hidden * bf16_bytes
layer_attn_params = (hidden * heads * head_dim + 
                     hidden * kv_heads * head_dim * 2 + 
                     heads * head_dim * hidden) * bf16_bytes
layer_ffn_params = (hidden * ffw_dim * 3) * bf16_bytes  # gate, up, down
total_params_bytes = embedding_params + (layer_attn_params + layer_ffn_params) * layers
# With FSDP, params are sharded
sharded_params = total_params_bytes / num_chips / 1e9
print(f'Sharded model params: {sharded_params:.3f} GB')

# Optimizer state (AdamW: 2x params for m and v)
optimizer_state = sharded_params * 2
print(f'Optimizer state (AdamW m+v): {optimizer_state:.3f} GB')

# Gradient accumulation buffer (same size as params)
grad_accum_buffer = sharded_params
print(f'Gradient accumulation buffer: {grad_accum_buffer:.3f} GB')
print()

# ===== ACTIVATIONS (the real issue) =====
print('=== Per-Chip Activations During Training ===')

# With FSDP on batch dimension, each chip processes per_chip_batch samples
# Hidden states stored at each layer output
hidden_per_layer = per_chip_batch * seq * hidden * bf16_bytes / 1e9
print(f'Hidden states per layer: {hidden_per_layer:.4f} GB')

# With remat on ATTENTION ONLY:
# - Attention activations are recomputed during backward (good!)
# - FFW activations are STORED for all layers (bad!)

ffn_gate_act = per_chip_batch * seq * ffw_dim * bf16_bytes / 1e9
ffn_up_act = per_chip_batch * seq * ffw_dim * bf16_bytes / 1e9
print(f'FFW gate activation per layer: {ffn_gate_act:.4f} GB')
print(f'FFW up activation per layer: {ffn_up_act:.4f} GB')

# During backward pass, we need to store FFW activations for ALL layers
# Because only attention is rematerialized, not FFW!
total_ffn_activations = (ffn_gate_act + ffn_up_act) * layers
print(f'Total FFW activations (ALL {layers} layers): {total_ffn_activations:.3f} GB')
print()

# Attention score memory (per layer, during recomputation)
# This is recomputed thanks to remat, so only need 1-2 layers worth at a time
attn_scores_per_layer = per_chip_batch * seq * heads * seq * bf16_bytes / 1e9
print(f'Attention scores per layer (during remat): {attn_scores_per_layer:.4f} GB')

# Q, K, V projections per layer
qkv_per_layer = per_chip_batch * seq * (heads + 2*kv_heads) * head_dim * bf16_bytes / 1e9
print(f'QKV projections per layer: {qkv_per_layer:.4f} GB')
print()

print('=== Total Estimated Per-Chip Memory ===')
static_mem = sharded_params + optimizer_state + grad_accum_buffer
print(f'Static (params + opt + grad_accum): {static_mem:.3f} GB')

# Peak activation memory during backward
# The backward pass through attention with remat stores:
# - Current layer attention intermediates
# - ALL FFW activations (not rematerialized!)
# - Gradients for current operations
peak_activations = total_ffn_activations + attn_scores_per_layer * 3 + qkv_per_layer * 3
print(f'Peak activations (FFW + attn remat): {peak_activations:.3f} GB')

total_estimated = static_mem + peak_activations
print(f'Total estimated: {total_estimated:.3f} GB')
print()

print('=== THE PROBLEM ===')
print('Current remat config (RematConfig.BLOCK) only rematerializes ATTENTION.')
print('FFW activations for ALL 26 layers are stored in memory!')
print(f'FFW alone consumes: {total_ffn_activations:.3f} GB per chip')
print()
print('Plus, during JVP (backward), attention still creates large intermediates.')
print('The error shows 30GB which suggests multiple layers of both are being retained.')
print()

print('=== SOLUTION OPTIONS ===')
print('1. Reduce batch size (currently 64 total, 8 per chip)')
print('2. Reduce sequence length (currently 2048)')
print('3. Add remat to FFW/MLP (not just attention)')
print('4. Use Flash Attention (reduces attention memory footprint)')
print()

# What batch size would fit?
# Available HBM: 16GB
# Static memory: ~2GB
# Headroom needed: ~2GB for XLA overhead
available_for_activations = 16 - static_mem - 2
print(f'Available for activations: {available_for_activations:.2f} GB')

# With current setup, FFW alone takes too much
# If we had remat on FFW too, we'd only need 1 layer at a time
ffn_one_layer = ffn_gate_act + ffn_up_act
attn_one_layer = attn_scores_per_layer + qkv_per_layer
print(f'If full remat: only ~{ffn_one_layer + attn_one_layer:.3f} GB for activations')
