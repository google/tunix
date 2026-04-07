from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from gemma.gm.nn.gemma4 import _config as up_config
from gemma.gm.nn.gemma4 import _layers
from gemma.gm.nn.gemma4 import _modules
from gemma.gm.nn.gemma4 import _moe
from gemma.gm.nn.gemma4 import _transformer as up_transformer
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import model as tunix_model
from tunix.models.gemma4 import moe as tunix_moe_new
from tunix.models.gemma4.params import map_from_upstream_checkpoint


class ComponentEquivalenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(42)

  def test_rmsnorm_match(self):
    tunix_norm = tunix_model.RMSNorm(64, rngs=self.rngs)
    up_norm = _layers.RMSNorm()

    x = jax.random.normal(self.rngs.params(), (4, 128, 64))
    variables = up_norm.init(self.rngs.params(), x)

    tunix_norm.scale.value = variables['params']['scale']

    tunix_out = tunix_norm(x)
    up_out = up_norm.apply(variables, x)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)

  def test_embedder_match(self):
    class DummyConfig:
      num_embed = 100
      embed_dim = 64
      num_layers = 2
      per_layer_input_dim = 0
      param_dtype = jnp.float32
      attn_type = 'standard'

      class ShdConfig:
        per_layer_model_projection = (None, None, None)
        per_layer_projection_norm = None
        per_layer_input_embedding = (None, None, None)

      shd_config = ShdConfig()

    config = DummyConfig()
    tunix_emb = tunix_model.Embedder(config=config, rngs=self.rngs)

    up_emb = _modules.Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        per_layer_input_dim=config.per_layer_input_dim,
    )

    x = jax.random.randint(self.rngs.params(), (4, 128), 0, config.num_embed)
    variables = up_emb.init(self.rngs.params(), x, method=up_emb.encode)

    tunix_emb.input_embedding.value = variables['params']['input_embedding']

    tunix_out = tunix_emb.encode(x)
    up_out = up_emb.apply(variables, x, method=up_emb.encode)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)

    # ------------------------------------------------------------------
    # Verify encode_per_layer_input as well
    # ------------------------------------------------------------------
    class DummyConfigPL:
      num_embed = 100
      embed_dim = 64
      num_layers = 2
      per_layer_input_dim = 16
      param_dtype = jnp.float32
      attn_type = 'standard'

      class ShdConfig:
        per_layer_model_projection = (None, None, None)
        per_layer_projection_norm = None
        per_layer_input_embedding = (None, None, None)

      shd_config = ShdConfig()

    config_pl = DummyConfigPL()
    tunix_emb_pl = tunix_model.Embedder(config=config_pl, rngs=self.rngs)

    up_emb_pl = _modules.Embedder(
        vocab_size=config_pl.num_embed,
        embed_dim=config_pl.embed_dim,
        per_layer_input_dim=config_pl.per_layer_input_dim,
        num_layers=config_pl.num_layers,
    )

    x_pl = jax.random.normal(self.rngs.params(), (4, 128, config_pl.embed_dim))
    t_pl = jax.random.randint(
        self.rngs.params(), (4, 128), 0, config_pl.num_embed
    )

    variables_pl = up_emb_pl.init(
        self.rngs.params(), x_pl, t_pl, method=up_emb_pl.encode_per_layer_input
    )

    tunix_emb_pl.per_layer_input_embedding.value = variables_pl['params'][
        'per_layer_embeddings'
    ]

    tunix_emb_pl.per_layer_model_projection.w.value = variables_pl['params'][
        'per_layer_model_projection'
    ]['w']
    tunix_emb_pl.per_layer_projection_norm.scale.value = variables_pl['params'][
        'per_layer_projection_norm'
    ]['scale']

    tunix_out_pl = tunix_emb_pl.encode_per_layer_input(x_pl, t_pl)
    up_out_pl = up_emb_pl.apply(
        variables_pl, x_pl, t_pl, method=up_emb_pl.encode_per_layer_input
    )

    np.testing.assert_allclose(tunix_out_pl, up_out_pl, rtol=1e-5, atol=1e-5)

  @parameterized.parameters([
      tunix_model.AttentionType.GLOBAL,
      tunix_model.AttentionType.LOCAL_SLIDING,
  ])
  def test_attention_match(self, attention_type: tunix_model.AttentionType):
    class DummyConfig:
      num_heads = 4
      num_kv_heads = 1
      embed_dim = 64
      head_dim = 16
      param_dtype = jnp.float32
      sliding_window_size = 32
      local_base_frequency = 10000
      global_base_frequency = 10000
      local_scale_factor = 1.0
      global_scale_factor = 1.0
      local_rope_proportion = 1.0
      global_rope_proportion = 1.0
      remat_config = tunix_model.RematConfig.NONE
      shd_config = tunix_model.ShardingConfig.get_default_sharding()
      k_eq_v_global = False
      num_global_kv_heads = None
      global_key_size = None
      dtype = jnp.float32


    config = DummyConfig()
    rngs = nnx.Rngs(42)

    tunix_attn = tunix_model.Attention(
        config=config,
        attn_type=attention_type,
        rngs=rngs,
    )

    up_attention_type = (
        _modules.AttentionType.LOCAL_SLIDING
        if attention_type == tunix_model.AttentionType.LOCAL_SLIDING
        else _modules.AttentionType.GLOBAL
    )

    up_attn = _modules.Attention(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        features=config.embed_dim,
        key_size=config.head_dim,
        attn_type=up_attention_type,
        rope_proportion=1.0,
        sliding_window_size=config.sliding_window_size,
    )

    x = jax.random.normal(rngs.params(), (4, 128, config.embed_dim))
    segment_pos = jnp.tile(jnp.arange(128, dtype=jnp.int32), (4, 1))
    attn_mask = jnp.tril(jnp.ones((4, 128, 128)), k=0).astype(jnp.bool_)

    variables = up_attn.init(rngs.params(), x, segment_pos, None, attn_mask)

    tunix_attn.q_einsum.w.value = variables['params']['q_einsum']['w']
    tunix_attn.kv_einsum.w.value = variables['params']['kv_einsum']['w']
    tunix_attn._query_norm.scale.value = variables['params']['query_norm'][
        'scale'
    ]
    tunix_attn._key_norm.scale.value = variables['params']['key_norm']['scale']
    tunix_attn.attn_vec_einsum.w.value = variables['params']['attn_vec_einsum'][
        'w'
    ]

    _, tunix_out = tunix_attn.block(x, segment_pos, None, attn_mask)
    _, up_out = up_attn.apply(variables, x, segment_pos, None, attn_mask)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)

  def test_feedforward_match(self):
    class DummyConfig:
      embed_dim = 64
      hidden_dim = 128
      param_dtype = jnp.float32

    config = DummyConfig()

    tunix_ff = tunix_model.FeedForward(config=config, rngs=self.rngs)

    up_ff = _modules.FeedForward(
        features=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )

    x = jax.random.normal(self.rngs.params(), (4, 128, config.embed_dim))

    variables = up_ff.init(self.rngs.params(), x)

    up_gate_w = variables['params']['gating_einsum']
    tunix_ff.gate_proj.kernel.value = up_gate_w[0].T
    tunix_ff.up_proj.kernel.value = up_gate_w[1].T
    tunix_ff.down_proj.kernel.value = variables['params']['linear']

    tunix_out = tunix_ff(x)
    up_out = up_ff.apply(variables, x)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)

  def test_decoder_layer_match(self):
    class DummyConfig:
      num_heads = 4
      num_kv_heads = 1
      embed_dim = 64
      head_dim = 16
      hidden_dim = 128
      dtype = jnp.float32
      param_dtype = jnp.float32
      sliding_window_size = 16
      local_base_frequency = 10_000
      global_base_frequency = 1_000_000
      global_rope_proportion = 0.25
      local_rope_proportion = 1.0
      local_scale_factor = 1.0
      global_scale_factor = 1.0
      remat_config = tunix_model.RematConfig.NONE
      enable_moe = False
      per_layer_input_dim = 0
      num_global_kv_heads = None
      global_key_size = None
      k_eq_v_global = False
      global_rope_proportion = 0.25
      local_rope_proportion = 1.0

      shd_config = tunix_model.ShardingConfig.get_default_sharding()

    config = DummyConfig()
    rngs = nnx.Rngs(42)
    attn_type = tunix_model.AttentionType.GLOBAL
    up_attn_type = (
        _modules.AttentionType.GLOBAL
        if attn_type == tunix_model.AttentionType.GLOBAL
        else _modules.AttentionType.LOCAL_SLIDING
    )

    tunix_layer = tunix_model.DecoderLayer(
        config=config,
        attn_type=attn_type,
        rngs=rngs,
    )

    up_layer = _modules.Block(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        embed_dim=config.embed_dim,
        head_dim=config.head_dim,
        hidden_dim=config.hidden_dim,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attn_type=up_attn_type,
        rope_base_frequency=config.global_base_frequency
        if up_attn_type == _modules.AttentionType.GLOBAL
        else config.local_base_frequency,
        sliding_window_size=config.sliding_window_size,
        global_rope_proportion=config.global_rope_proportion,
        local_rope_proportion=config.local_rope_proportion,
    )

    x = jax.random.normal(rngs.params(), (4, 128, config.embed_dim))
    segment_pos = jnp.tile(jnp.arange(128, dtype=jnp.int32), (4, 1))
    attn_mask = jnp.tril(jnp.ones((4, 128, 128)), k=0).astype(jnp.bool_)

    variables = up_layer.init(rngs.params(), x, segment_pos, None, attn_mask)

    # Map weights
    tunix_layer.pre_attention_norm.scale.value = variables['params'][
        'pre_attention_norm'
    ]['scale']

    up_attn = variables['params']['attn']
    tunix_layer.attn.q_einsum.w.value = up_attn['q_einsum']['w']
    tunix_layer.attn.kv_einsum.w.value = up_attn['kv_einsum']['w']
    tunix_layer.attn._query_norm.scale.value = up_attn['query_norm']['scale']
    tunix_layer.attn._key_norm.scale.value = up_attn['key_norm']['scale']
    tunix_layer.attn.attn_vec_einsum.w.value = up_attn['attn_vec_einsum']['w']

    tunix_layer.post_attention_norm.scale.value = variables['params'][
        'post_attention_norm'
    ]['scale']

    tunix_layer.pre_ffw_norm.scale.value = variables['params']['pre_ffw_norm'][
        'scale'
    ]

    up_gate_w = variables['params']['mlp']['gating_einsum']
    tunix_layer.mlp.gate_proj.kernel.value = up_gate_w[0].T
    tunix_layer.mlp.up_proj.kernel.value = up_gate_w[1].T
    tunix_layer.mlp.down_proj.kernel.value = variables['params']['mlp'][
        'linear'
    ]

    tunix_layer.post_ffw_norm.scale.value = variables['params'][
        'post_ffw_norm'
    ]['scale']

    tunix_layer.skip_scale.value = variables['params']['skip_scale']

    _, tunix_out = tunix_layer(x, segment_pos, None, attn_mask)
    _, up_out = up_layer.apply(variables, x, segment_pos, None, attn_mask)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)


  def test_moe_new_file_match(self):
    rngs = nnx.Rngs(42)
    features = 64
    hidden_dim = 128
    num_experts = 4
    num_experts_per_datapoint = 2

    tunix_moe_inst = tunix_moe_new.MoE(
        features=features,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_datapoint=num_experts_per_datapoint,
        rngs=rngs,
    )

    up_moe_inst = _moe.MoE(
        features=features,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_datapoint=num_experts_per_datapoint,
    )

    x = jax.random.normal(rngs.params(), (4, 128, features))

    variables = up_moe_inst.init(rngs.params(), x)

    # Map weights
    tunix_moe_inst.router_logits.value = variables['params']['router_logits']['w']
    tunix_moe_inst.linear.value = variables['params']['linear']['w']
    tunix_moe_inst.gating_einsum.value = variables['params']['gating_einsum']['w']
    tunix_moe_inst.per_expert_scale.value = variables['params']['per_expert_scale']
    tunix_moe_inst.router_scale.value = variables['params']['router_scale']

    tunix_out = tunix_moe_inst(x)
    up_out = up_moe_inst.apply(variables, x)

    np.testing.assert_allclose(tunix_out, up_out, rtol=1e-5, atol=1e-5)

  def test_full_model_match(self):
    # Gemma 4 E2B model config.
    cfg = up_config.TransformerConfig(
        attention_types=(
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.GLOBAL,
        ),
        embed_dim=1536,
        hidden_dim=6144,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        num_embed=1024,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        sliding_window_size=512,
        per_layer_input_dim=256,
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        qk_norm_with_scale=True,
        global_key_size=512,
        k_eq_v_global=False,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
    )

    config = tunix_model.ModelConfig(
        num_layers=4,
        num_embed=1024,
        embed_dim=1536,
        hidden_dim=6144,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        per_layer_input_dim=256,
        sliding_window_size=512,
        param_dtype=jnp.float32,
        attention_pattern=(
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.GLOBAL,
        ),
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_key_size=512,
        k_eq_v_global=False,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
    )

    rng = jax.random.PRNGKey(0)
    upstream = up_transformer.Transformer(config=cfg, dtype=jnp.float32)
    init_vars = upstream.init(rng, jnp.ones((1, 1), dtype=jnp.int32))

    mapped_params = map_from_upstream_checkpoint(init_vars['params'])
    mapped_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and x.dtype != jnp.int32
        else x,
        mapped_params,
    )
    model = tunix_model.Gemma4(config, rngs=nnx.Rngs(0))
    nnx.update(model, mapped_params)

    T = 128
    tokens = jax.random.randint(rng, (1, T), 0, config.num_embed)
    positions = jnp.arange(T)[None, :]
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, ...]

    tunix_logits, _ = model(
        tokens, positions=positions, attention_mask=causal_mask
    )
    up_out = upstream.apply(
        init_vars, tokens, positions=positions, attention_mask=causal_mask
    )
    up_logits = up_out.logits

    np.testing.assert_allclose(tunix_logits, up_logits, rtol=1e-4, atol=1e-4)


  def test_full_model_moe_26b_match(self):
    # Gemma 4 26B MoE model config (reduced to 6 layers).
    cfg = up_config.TransformerConfig(
        num_embed=1000, # Reduced
        embed_dim=512, # Reduced
        hidden_dim=1024, # Reduced
        num_heads=4, # Reduced
        head_dim=128, # Reduced
        num_kv_heads=2, # Reduced
        final_logit_softcap=30.0,
        num_global_kv_heads=2,
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        qk_norm_with_scale=True,
        attention_types=(
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.LOCAL_SLIDING,
            _modules.AttentionType.GLOBAL,
        ),
        global_key_size=512,
        k_eq_v_global=True,
        global_rope_proportion=0.25,
        local_rope_proportion=1.0,
        sliding_window_size=1024,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        per_layer_input_dim=0,
        enable_moe=True,
        num_experts=12,  # Reduced experts to save memory/time
        expert_dim=704,
        top_k_experts=8,
        moe_dense_hidden_dim=2112,
    )

    config = tunix_model.ModelConfig(
        num_layers=6,
        num_embed=1000, # Reduced
        embed_dim=512, # Reduced
        hidden_dim=1024, # Reduced
        num_heads=4, # Reduced
        head_dim=128, # Reduced
        num_kv_heads=2, # Reduced
        per_layer_input_dim=0,
        sliding_window_size=1024,
        param_dtype=jnp.float32,
        attention_pattern=(
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.LOCAL_SLIDING,
            tunix_model.AttentionType.GLOBAL,
        ),
        final_logit_softcap=30.0,
        local_rope_proportion=1.0,
        global_rope_proportion=0.25,
        global_key_size=512,
        k_eq_v_global=True,
        local_base_frequency=10000,
        global_base_frequency=1000000,
        local_scale_factor=1.0,
        global_scale_factor=1.0,
        enable_moe=True,
        num_experts=12,  # Reduced experts
        expert_dim=704,
        num_experts_per_tok=8,
    )

    rng = jax.random.PRNGKey(0)
    upstream = up_transformer.Transformer(config=cfg, dtype=jnp.float32)
    init_vars = upstream.init(rng, jnp.ones((1, 1), dtype=jnp.int32))

    mapped_params = map_from_upstream_checkpoint(init_vars['params'])
    mapped_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, 'dtype') and x.dtype != jnp.int32
        else x,
        mapped_params,
    )
    model = tunix_model.Gemma4(config, rngs=nnx.Rngs(0))
    nnx.update(model, mapped_params)

    T = 32 # Reduced seq len to save time
    tokens = jax.random.randint(rng, (1, T), 0, config.num_embed)
    positions = jnp.arange(T)[None, :]
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, ...]

    tunix_logits, _ = model(
        tokens, positions=positions, attention_mask=causal_mask
    )
    up_out = upstream.apply(
        init_vars, tokens, positions=positions, attention_mask=causal_mask
    )
    up_logits = up_out.logits

    np.testing.assert_allclose(tunix_logits, up_logits, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  absltest.main()
