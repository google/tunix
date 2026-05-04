import functools
import gc
import logging
import os
import sys
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from flax import nnx
from jax import config as jax_config
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
import torch
from tunix.models.qwen3 import model as qwen3_lib
from vllm import LLM, SamplingParams

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.configs import pyconfig
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# JAX compilation cache settings - adjust as needed for your environment
_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

# Flags
FLAGS = flags.FLAGS
_XPROF = flags.DEFINE_bool('xprof', False, 'xprof')
_RAND_INIT = flags.DEFINE_bool('rand_init', False, 'Whether to use random initialization instead of loading from checkpoint, for faster testing.')  

def _setup_jax_compilation_cache():
  jax_config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
  jax_config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax_config.update("jax_enable_compilation_cache", True)


def _setup_vllm():
  # for vLLM we can skip JAX precompilation with this flag, it makes startup faster
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["JAX_RANDOM_WEIGHTS"] = "False"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def _clean_device_memory():
  """Forces Python garbage collection and waits for JAX devices to idle."""
  logging.info("Cleaning JAX device memory...")
  # Run Python's garbage collector to free Python-level references
  gc.collect()
  # Wait for all devices to finish pending operations.
  # This allows JAX to reclaim memory associated with arrays
  # that are no longer referenced.
  for x in jax.live_arrays():
      x.delete()      
  logging.info("Device memory cleanup complete.")

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.4f} seconds")  

def _get_maxtext_model(config):
  """Creates and returns a Tunix-adapted MaxText model and mesh."""
  logging.info(f'Creating model with config: {config}')
  model, mesh = model_creation_utils.create_nnx_model(
    # config, model_mode=MODEL_MODE_AUTOREGRESSIVE)
    config, model_mode=MODEL_MODE_AUTOREGRESSIVE, use_rand_init=_RAND_INIT.value)    
  with mesh:
    tunix_model = TunixMaxTextAdapter(base_model=model)
    # Use the appropriate model config based on the model name
    if config.model_name == "qwen3-30b-a3b":
      model_config = qwen3_lib.ModelConfig.qwen3_30b_a3b()
    elif config.model_name == "qwen3-0.6b":
      model_config = qwen3_lib.ModelConfig.qwen3_0_6b()
    elif config.model_name == "qwen3-235b-a22b":
      model_config = qwen3_lib.ModelConfig(
        num_layers=94,
        vocab_size=151936,
        embed_dim=4096,
        hidden_dim=1536,
        num_heads=64,
        head_dim=128,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=5_000_000,
        num_experts=128,
        num_experts_per_tok=8,
    )
    else:
      raise ValueError(f"Unsupported model: {config.model_name}")
    tunix_model.config = model_config
  return tunix_model, mesh

def _load_maxtext_model(base_yaml_path):
  # Initialize MaxText config
  # config_ref = pyconfig.initialize(
  #     [ "", BASE_YAML_PATH, ],
  #     base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
  #     run_name="test-tunix-maxtext-qwen3-8b",
  #     tokenizer_type="huggingface",
  #     tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
  #     # model_name="qwen3-0.6b",
  #     # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-0.6b/scanned/2026-01-21-11-35/0/items",
  #     model_name="qwen3-30b-a3b",
  #     load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items",
  #     # load_parameters_path="/dev/shm/hengtaoguo/0/items/0/items",
  #     # model_name="qwen3-235b-a22b",
  #     # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-235b-a22b/scanned/001/0/items",
  #     # scan_layers="true",
  #     per_device_batch_size=1,
  #     max_prefill_predict_length=8,
  #     max_target_length=16,
  #     steps=100,
  #     async_checkpointing="false",
  #     checkpoint_period=5,
  #     skip_jax_distributed_system="true",
  #     weight_dtype="bfloat16",
  #     attention="dot_product",
  #     remat_policy="custom",
  #     decoder_layer_input="offload",
  #     query_proj="offload",
  #     key_proj="offload",
  #     value_proj="offload",
  #     ici_fsdp_parallelism=2,
  #     ici_tensor_parallelism=4,
  #     override_model_config="true",
  #     # base_num_decoder_layers=2,
  # )  
  config_ref = pyconfig.initialize(
      [ "", base_yaml_path, ],
      base_output_directory="gs://wyzhang-dev/tmp",  # Not used in Tunix.
      run_name="test-tunix-maxtext-qwen3",
      tokenizer_type="huggingface",
      tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "qwen3-tokenizer"),
      # model_name="qwen3-235b-a22b",
      # load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-235b-a22b/scanned/Qwen3-235B-A22B/0/items",
      model_name="qwen3-30b-a3b",
      load_parameters_path="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3-30b-a3b/scanned/2026-01-23-14-00/0/items/0/items",
      scan_layers="true",
      per_device_batch_size=1,
      max_prefill_predict_length=8,
      max_target_length=16,
      steps=100,
      async_checkpointing="false",
      checkpoint_period=5,
      skip_jax_distributed_system="true",
      weight_dtype="bfloat16",
      attention="dot_product",
      remat_policy="custom",
      decoder_layer_input="offload",
      query_proj="offload",
      key_proj="offload",
      value_proj="offload",
      ici_expert_parallelism=4,
      # ici_fsdp_parallelism=2,
      # ici_tensor_parallelism=2,
      rollout_tensor_parallelism=4,
      override_model_config="true",
      # debug_sharding="true",
      checkpoint_storage_concurrent_gb=80,
      async_scheduling="false",
  )
  model, mesh = _get_maxtext_model(config_ref)
  return config_ref, model, mesh

class MaxTextToVLLMConverter:
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.num_layers = config.base_num_decoder_layers
        self.vllm_state = {}
        self.vllm_tp = self.config.rollout_tensor_parallelism

    # --- 1. Top-Level Entry Point ---
    def convert(self, model_state: dict):
        """Main entry point to convert all weights."""
        logging.info(f"\n{GREEN}Starting Conversion...{RESET}")
        start_time = time.time()

        with timer("Convert Global Weights"):
          self._convert_global(model_state)
        with timer("Convert Attention Weights"):
          self._convert_attn(model_state)
        with timer("Convert MoE Weights"):
          self._convert_moe(model_state)
        
        return self.vllm_state

    def _convert_global(self, params):
        logging.info("_convert_global: embed_tokens...")
        self._to_embed_tokens(params)
        logging.info("_convert_global: final_norm...")
        self._to_final_norm(params)
        logging.info("_convert_global: lm_head...")
        self._to_lm_head(params)
        logging.info("_convert_global: done")
                
    def _convert_attn(self, params):
      @jax.jit
      def _transpose_unstack(x):
        return jnp.unstack(jnp.transpose(x, (1, 0)))

      logging.info("_convert_attn: pre_self_attention_layer_norm...")
      pre_ln = params['base']['decoder']['layers']['pre_self_attention_layer_norm']['scale']
      convert_pre_ln = _transpose_unstack(pre_ln)
      assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
      for i, layer in enumerate(convert_pre_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.input_layernorm.weight': layer})
      del convert_pre_ln

      logging.info("_convert_attn: post_self_attention_layer_norm...")
      post_ln = params['base']['decoder']['layers']['post_self_attention_layer_norm']['scale']
      converted_post_ln = _transpose_unstack(post_ln)
      assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
      for i, layer in enumerate(converted_post_ln):
        self.vllm_state.update({f'vllm_model.model.layers.{i}.post_attention_layernorm.weight': layer})      
      del post_ln

      logging.info("_convert_attn: self_attention (qkv/o/norms)...")
      attn = params['base']['decoder']['layers']['self_attention']
      self_attn = self._to_attn(attn)
      for key, layers in self_attn.items():
        self.vllm_state.update({f'vllm_model.model.layers.{i}.{key}': layer for i, layer in enumerate(layers)})
      del attn
      logging.info("_convert_attn: done")
      gc.collect()

    def _convert_moe(self, params):
      logging.info("_convert_moe: extracting moe_block...")
      moe = params['base']['decoder']['layers']['moe_block'].to_pure_dict()
      prefix = "vllm_model.model.layers"

      logging.info("_convert_moe: gate weights...")
      self.vllm_state.update({
          f"{prefix}.{i}.mlp.gate.weight": w 
          for i, w in enumerate(self._to_mlp_gate(moe['gate']['kernel']))
      })
      del moe['gate']['kernel']
      gc.collect()

      logging.info("_convert_moe: expert down (w2) weights...")
      self.vllm_state.update({
          f"{prefix}.{i}.mlp.experts.w2_weight": w 
          for i, w in enumerate(self._to_mlp_expert_down(moe['wo']))
      })
      del moe['wo']
      gc.collect()

      logging.info("_convert_moe: expert gate+up (w13) weights (fuse_all jit+vmap)...")
      self._to_mlp_expert_gate_up(
          moe['wi_0'], moe['wi_1'], 
          self.num_layers, prefix, 'mlp.experts.w13_weight'
      )
      del moe['wi_0'], moe['wi_1']
      logging.info("_convert_moe: done")
      gc.collect()
      
    def _to_final_norm(self, params):
      self.vllm_state["vllm_model.model.norm.weight"] = params['base']['decoder']['decoder_norm']['scale']

    def _to_embed_tokens(self, params):
      self.vllm_state["vllm_model.model.embed_tokens.weight"] = params['base']['token_embedder']['embedding']

    def _to_lm_head(self, params):
      @jax.jit
      def _transpose(x):
        return jnp.transpose(x, (1, 0))
      self.vllm_state["vllm_model.lm_head.weight"] = _transpose(
          params['base']['decoder']['logits_dense']['kernel']
      )
      
    def _to_attn(self, attn: PyTree) -> dict[str, jax.Array]:
      tp = min(self.vllm_tp, self.config.base_num_kv_heads)  # Don't TP-shard more heads than exist in the model.

      @jax.jit
      def _compute(attn):
        # (d_model, l, h, d) -> (l, d_model, h, d)
        q = jnp.transpose(attn['query']['kernel'], (1, 0, 2, 3))
        k = jnp.transpose(attn['key']['kernel'], (1, 0, 2, 3))
        v = jnp.transpose(attn['value']['kernel'], (1, 0, 2, 3))

        num_q_heads = q.shape[2]
        num_kv_heads = k.shape[2]
        head_dim = q.shape[3]
        l, d_model = q.shape[0], q.shape[1]

        kv_per_tp = num_kv_heads // tp
        q_per_tp = num_q_heads // tp

        q_by_tp = q.reshape(l, d_model, tp, q_per_tp, head_dim)
        k_by_tp = k.reshape(l, d_model, tp, kv_per_tp, head_dim)
        v_by_tp = v.reshape(l, d_model, tp, kv_per_tp, head_dim)

        qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
        qkv_flat = qkv_by_tp.reshape(l, d_model, -1)
        qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))

        o = jnp.transpose(attn['out']['kernel'], (1, 3, 0, 2))
        o_proj = o.reshape(o.shape[0], o.shape[1], -1)

        q_norm = jnp.transpose(attn['query_norm']['scale'], (1, 0))
        k_norm = jnp.transpose(attn['key_norm']['scale'], (1, 0))

        return {
            "self_attn.qkv_proj.weight": jnp.unstack(qkv_proj),
            "self_attn.o_proj.weight": jnp.unstack(o_proj),
            "self_attn.q_norm.weight": jnp.unstack(q_norm),
            "self_attn.k_norm.weight": jnp.unstack(k_norm)
        }

      return _compute(attn)
    
    def _to_mlp_gate(self, param):
      # param: [d_model, l, total_e] -> [l, total_e, d_model]
      # shard_map removed: plain transpose lets GSPMD propagate sharding
      # without requiring param and mesh to be on the same device set.
      @jax.jit
      def _transpose(param):
        return jnp.transpose(param, (1, 2, 0))
      param = _transpose(param)
      return self._unstack_layer(param)

    def _to_mlp_expert_down(self, param):
      # param: [E, L, Hidden, Inter] -> [L, E, Inter, Hidden]
      @jax.jit
      def _transpose(param):
        return jnp.transpose(param, (1, 0, 3, 2))
      param = _transpose(param)
      # vLLM 0.17+ expects (E, Hidden, Inter) -> (E, Inter, Hidden)
      # So for each layer, do param[i]: (E, Inter, Hidden) -> (E, Hidden, Inter)
      param = jnp.transpose(param, (0, 1, 3, 2))
      return self._unstack_layer(param)

    def _to_mlp_expert_gate_up(self, wi_0, wi_1, num_layers, layer_key_prefix, layer_key_suffix):
      # Process all layers in one JIT call using vmap to avoid per-layer dispatch
      # overhead (which was ~50 separate device syncs on multi-host v5p-64).
      tp = self.vllm_tp
      
      @jax.jit
      def _fuse_all(wi_0, wi_1):
        # wi_0, wi_1: (e, l, d_model, d_inner) -> (l, e, d_model, d_inner)
        wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))  # -> (l, e, d_model, d_inner)
        wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

        def _fuse_single(w0, w1):
          # [e, d_model, d_inner] -> [e, 2*padded_chunk_size*tp, d_model]
          w0 = jnp.transpose(w0, (0, 2, 1))  # gate: [e, d_inner, d_model]
          w1 = jnp.transpose(w1, (0, 2, 1))  # up:   [e, d_inner, d_model]
          e, d_inner, d_model = w0.shape
          # Chunk-level interleave to match vLLM TP sharding:
          # layout: [gate_chunk0, up_chunk0, gate_chunk1, up_chunk1, ...]
          chunk_size = d_inner // tp
          # Pad each TP chunk up to the next multiple of 128 for TPU GMM kernel
          # alignment (same logic as process_w13_for_gmm in tpu_inference).
          # e.g. d_inner=768, tp=4 → chunk=192 → padded=256; tp=8 → 96 → 128.
          padded_chunk_size = ((chunk_size + 127) // 128) * 128
          pad_amount = padded_chunk_size - chunk_size
          gate_chunks = w0.reshape(e, tp, chunk_size, d_model)
          up_chunks = w1.reshape(e, tp, chunk_size, d_model)
          if pad_amount > 0:
            gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
            up_chunks   = jnp.pad(up_chunks,   ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          combined = jnp.stack([gate_chunks, up_chunks], axis=2)
          return combined.reshape(e, 2 * padded_chunk_size * tp, d_model)

        return jax.vmap(_fuse_single)(wi_0, wi_1)  # -> (l, e, 2*d_inner, d_model)

      logging.info("_to_mlp_expert_gate_up: dispatching _fuse_all (single JIT+vmap)...")
      fused = _fuse_all(wi_0, wi_1)
      logging.info("_to_mlp_expert_gate_up: _fuse_all complete, shape=%s, unstacking layers...", fused.shape)
      del wi_0, wi_1
      gc.collect()

      # vLLM 0.17+ expects (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
      for i, layer_i in enumerate(jnp.unstack(fused, axis=0)):
        layer_i = jnp.transpose(layer_i, (0, 2, 1))  # (e, 2*d_inner, d_model) -> (e, d_model, 2*d_inner)
        self.vllm_state[f"{layer_key_prefix}.{i}.{layer_key_suffix}"] = layer_i
        if i % 8 == 7:
          gc.collect()
      del fused
      gc.collect()
      
    @staticmethod
    @jax.jit
    def _unstack_layer(param):
        return jnp.unstack(param, axis=0)


def save_dict_to_file(dict, filename):
    with open(filename, 'w') as f:
        for key in sorted(dict.keys()):
            f.write(f"{key}: {dict[key].shape}\n")


def main():
  print(f"JAX devices: {jax.devices()}")  
  _setup_jax_compilation_cache()
  _setup_vllm()
  _clean_device_memory()

  FLAGS(sys.argv)

  # Load maxtext model
  print("="*80)
  print("Loading MaxText model...")
  print("="*80)
  # path1 = "/home/wyzhang_google_com/mnt/rl/maxtext/src/maxtext/configs/base.yml"
  path1 = "/home/hengtaoguo_google_com/projects/maxtext/src/maxtext/configs/base.yml"
  path2 = os.path.join(os.path.expanduser("~"), "mnt/rl/maxtext/src/maxtext/configs/base.yml")
  if os.path.exists(path1):
    base_yaml_path = path1
  elif os.path.exists(path2):
    base_yaml_path = path2
  else:
    raise FileNotFoundError(
        f"Could not find base.yml in the expected locations: {path1} or {path2}"
    )
  config, model, mesh = _load_maxtext_model(base_yaml_path)
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {config.model_name}")
  print(f"Mesh: {mesh}")

  # Convert weights to VLLM format
  print("="*80)
  print("Converting weights to VLLM format")
  print("="*80)
  model_state = nnx.state(model)
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, 'shape') and hasattr(leaf, 'sharding'):
      path_str = jax.tree_util.keystr(path)
      logging.info(f"Name: {path_str}, shape: {leaf.shape}")
      logging.info(f"\tSharding: {leaf.sharding}")
  
  converter = MaxTextToVLLMConverter(config, mesh)
  with timer("Overall Conversion "):
    vllm_state = converter.convert(model_state)
  del model_state
  gc.collect()
  # save_dict_to_file(vllm_state, "vllm_state_shapes.txt")

  # Load vLLM model and run generation test
  print("="*80)
  print("Loading vLLM model for generation test...")
  print("="*80)
  llm = LLM(
    "Qwen/Qwen3-30B-A3B",
    max_model_len=16,
    tensor_parallel_size=4,
    # tensor_parallel_size=4,
    # tensor_parallel_size=2,
    # data_parallel_size=2,
    # enable_expert_parallel=True,
    gpu_memory_utilization=0.65,
    load_format="dummy",
    async_scheduling=False,
  )
  print("\n" + "="*80)
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
  # save_dict_to_file(llm_state, "llm_state_shapes.txt")

  # Detect once whether MaxText and vLLM meshes share the same physical
  # devices (single-VM) or are disjoint (multi-host cluster partition).
  # Same devices → jax.jit with out_shardings lets XLA emit on-device
  # collectives (fast, ~0.16 s for 30 B).  Disjoint devices → jax.jit
  # raises "incompatible devices"; fall back to jax.device_put which uses
  # ICI/DCN for the cross-host transfer without a CPU roundtrip.
  _any_src = next(iter(vllm_state.values()))
  _any_src_arr = _any_src.value if hasattr(_any_src, 'value') else _any_src
  _any_dst = next(iter(llm_state.values()))
  _same_devices = (
      frozenset(d.id for d in _any_src_arr.sharding.mesh.devices.flat) ==
      frozenset(d.id for d in _any_dst.sharding.mesh.devices.flat)
  )
  logging.info("Weight sync: same_devices=%s (jit=%s, device_put=%s)",
               _same_devices, _same_devices, not _same_devices)

  @functools.lru_cache(maxsize=None)
  def _get_reshard_fn(dst_sharding):
    if _same_devices:
      return jax.jit(lambda x: x, out_shardings=dst_sharding)
    else:
      return functools.partial(jax.device_put, device=dst_sharding)

  with timer(f"Assigning {len(vllm_state)} weights to vLLM model"):
    for key, weight in vllm_state.items():
      # Unwrap NNX Param → plain jax.Array; plain arrays pass through unchanged.
      weight_array = weight.value if hasattr(weight, 'value') else weight
      dst_sharding = llm_state[key].sharding
      print(f"{key}: mt {weight_array.shape}, vllm {llm_state[key].shape}")
      llm_state[key] = _get_reshard_fn(dst_sharding)(weight_array)
      # array_allclose = np.allclose(
      #     np.asarray(llm_state[key]), np.asarray(weight_array), rtol=1e-2, atol=1e-2
      # )
      # print(f"{key}: {array_allclose}")
      # llm_state[key] = jax.device_put(np.asarray(weight_array), dst_sharding)
    jax.effects_barrier()  # wait for all on-device reshards to finish


  sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
  print("\n" + "="*80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate("Paris is", sampling_params=sampling_params))


if __name__ == "__main__":
  main()