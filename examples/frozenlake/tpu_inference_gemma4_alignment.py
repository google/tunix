"""Check the model correctness for Tunix nnx implemented Gemma 4 models.

The test will compare the first N decoder layer output between Tunix model and
HF PyTorch model, typically we will expect the logits difference to be within
1e-3 in fp32.
"""

import inspect
import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.gemma4 import params_safetensors as gemma4_params
from tunix.sft import utils
from tunix.tests import test_common as tc

K_MASK = -2.3819763e38


def create_pytorch_causal_mask(seq_len):
  """Creates a causal attention mask for a sequence of a given length."""
  mask = torch.ones(seq_len, seq_len, dtype=torch.float).tril(diagonal=0)
  mask = mask.masked_fill(mask == 0, K_MASK)
  mask = mask.masked_fill(mask == 1, 0)
  return mask


def get_hf_output(model, seq_len: int):
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
  return model(x, attn_mask, position_ids).logits.detach().numpy()


def get_jax_output(model, seq_len: int):
  x = (jnp.arange(seq_len) + 1).reshape(1, -1)
  positions = jnp.arange(seq_len).reshape(1, -1)
  attn_mask = utils.make_causal_attn_mask(jnp.ones((1, seq_len)))
  output, _ = model(x, positions, None, attn_mask)
  return output


def get_per_layer_hf_output(model, seq_len: int, num_layer_to_run: int = 1):
  """Get the first decoder layer output from the HF model."""
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)

  m = model.get_decoder()
  emb = m.embed_tokens(x)

  try:
    position_embeddings = m.rotary_emb(emb, position_ids)
  except Exception:
    position_embeddings = None

  logits = emb
  for i in range(num_layer_to_run):
    layer = m.layers[i]
    sig = inspect.signature(layer.forward)
    kwargs = {}
    if position_embeddings is not None:
      if "position_embeddings" in sig.parameters:
        kwargs["position_embeddings"] = position_embeddings
      elif "rotary_pos_emb" in sig.parameters:
        kwargs["rotary_pos_emb"] = position_embeddings
    logits = layer(logits, attn_mask, position_ids, **kwargs)

  return logits[0].detach().numpy()


def get_per_layer_jax_output(model, seq_len: int, num_layer_to_run: int = 1):
  """Get the first Tunix decoder layer output."""
  x = (jnp.arange(seq_len) + 1).reshape(1, -1)
  positions = jnp.arange(seq_len).reshape(1, -1)
  attn_mask = utils.make_causal_attn_mask(jnp.ones((1, seq_len)))

  logits = model.embedder.encode(x)
  for i in range(num_layer_to_run):
    _, logits, _ = model.layers[i](logits, positions, None, attn_mask)

  return logits


class Gemma4AlignTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma4_e2b",
          model_name="google/gemma-4-E2B-it",
          model_config=gemma4_model.ModelConfig.gemma4_e2b,
          model_params=gemma4_params,
          tolerance=2e-3,
      ),
  )
  def test_gemma4_model_alignment(
      self, model_name, model_config, model_params, tolerance
  ):
    local_cache_path = "/home/linchai_google_com/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/905e84b50c4d2a365ebde34e685027578e6728db"
    if os.path.exists(local_cache_path):
      model_path = local_cache_path
      print(f"Using local cached model from {model_path}")
    else:
      model_path = os.path.join(tempfile.gettempdir(), "models", model_name)
      tc.download_from_huggingface(repo_id=model_name, model_path=model_path)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    )
    print("HF model loaded.")

    jax_model = model_params.create_model_from_safe_tensors(
        model_path,
        model_config(),
        mesh=jax.make_mesh(
            (1, 1),
            ("fsdp", "tp"),
            axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
        ),
        dtype=jnp.float32,
    )
    print("JAX model loaded.")

    # Make sure model weights are the same (check q_einsum weight)
    hf_query_weight = (
        hf_model.get_decoder()
        .layers[0]
        .self_attn.q_proj.weight.detach()
        .numpy()
    )
    jax_query_weight = jax_model.layers[0].attn.q_einsum.w
    n, d, h = jax_query_weight.shape
    jax_query_weight = jax_query_weight.transpose(0, 2, 1).reshape(-1, d)
    np.testing.assert_equal(
        hf_query_weight,
        jax_query_weight,
        err_msg=(
            "Query weights are not equal, are you sure the loaded model weight"
            " between HF and JAX is identical?"
        ),
    )

    seq_len = 128

    layer_to_run = model_config().num_layers
    hf_logits = get_per_layer_hf_output(hf_model, seq_len, layer_to_run)
    jax_logits = get_per_layer_jax_output(jax_model, seq_len, layer_to_run)
    np.testing.assert_allclose(
        hf_logits.squeeze(),
        jax_logits.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )

    # Do a check on entire model output
    hf_output = get_hf_output(hf_model, seq_len)
    jax_output = get_jax_output(jax_model, seq_len)
    np.testing.assert_allclose(
        hf_output.squeeze(),
        jax_output.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )

    print("Logits are close! Model alignment check passed :)")

    if model_path != local_cache_path:
      tc.delete_directory(model_path)


if __name__ == "__main__":
  absltest.main()
