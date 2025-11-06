"""Check the model correctness for Tunix nnx implemented models.

The test will compare the first N decoder layer output between Tunix model and
HF PyTorch model, typically we will expect the logits differnece to be within
1e-3 in fp32.
"""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen2 import params as qwen2_params
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.sft import utils
from tunix.tests import test_common as tc

K_MASK = -2.3819763e38


def create_pytorch_causal_mask(seq_len):
  """Creates a causal attention mask for a sequence of a given length.

  Args:
    seq_len: The length of the sequence.

  Returns:
    A boolean tensor of shape (seq_len, seq_len) where:
    - mask[i, j] is True if token i can attend to token j (j <= i).
    - mask[i, j] is False if token i cannot attend to token j (j > i).
  """
  # Create a lower triangular matrix of ones
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
  position_embeddings = m.rotary_emb(emb, position_ids)

  logits = emb
  for i in range(num_layer_to_run):
    logits = m.layers[i](
        logits, attn_mask, position_ids, position_embeddings=position_embeddings
    )

  return logits[0].detach().numpy()


def get_per_layer_jax_output(
    model_name: str, model, seq_len: int, num_layer_to_run: int = 1
):
  """Get the first decoder layer output from the Tunix model."""
  x = (jnp.arange(seq_len) + 1).reshape(1, -1)
  positions = jnp.arange(seq_len).reshape(1, -1)
  attn_mask = utils.make_causal_attn_mask(jnp.ones((1, seq_len)))

  if model_name in [
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
      "Qwen/Qwen2.5-1.5B-Instruct",
  ]:
    sin, cos = qwen2_model._generate_pos_embeddings(  # pylint: disable=protected-access
        positions, model.config.head_dim, model.config.rope_theta
    )

    logits = model.embedder.encode(x)
    for i in range(num_layer_to_run):
      _, logits = model.layers[i](logits, None, attn_mask, sin, cos)

    return logits
  elif model_name == "Qwen/Qwen3-0.6B":
    logits = model.embedder.encode(x)
    for i in range(num_layer_to_run):
      _, logits = model.layers[i](logits, positions, None, attn_mask)

    return logits
  else:
    raise ValueError(f"Unsupported model: {model_name}")


class QwenAlignTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="deepseek_r1_distill_qwen_1_5b",
          model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
          model_config=qwen2_model.ModelConfig.deepseek_r1_distill_qwen_1_5b,
          model_params=qwen2_params,
          tolerance=2e-3,
      ),
      dict(
          testcase_name="qwen2_5_1_5b_instruct",
          model_name="Qwen/Qwen2.5-1.5B-Instruct",
          model_config=qwen2_model.ModelConfig.qwen2_5_1_5b,
          model_params=qwen2_params,
          tolerance=1e-3,
      ),
      dict(
          testcase_name="qwen3_0_6b",
          model_name="Qwen/Qwen3-0.6B",
          model_config=qwen3_model.ModelConfig.qwen3_0_6b,
          model_params=qwen3_params,
          tolerance=1e-3,
      ),
  )
  def test_qwen_model_alignment(
      self, model_name, model_config, model_params, tolerance
  ):
    model_path = os.path.join(tempfile.gettempdir(), "models", model_name)

    tc.download_from_huggingface(repo_id=model_name, model_path=model_path)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    )
    print("HF model loaded.")

    jax_model = model_params.create_model_from_safe_tensors(
        model_path,
        model_config(),
        mesh=jax.make_mesh((1, 1), ("fsdp", "tp")),
        dtype=jnp.float32,
    )
    print("JAX model loaded.")

    # Make sure model weights are the same (only check the first query weight)
    hf_query_weight = (
        hf_model.get_decoder()
        .layers[0]
        .self_attn.q_proj.weight.detach()
        .numpy()
    )
    jax_query_weight = jax_model.layers[0].attn.q_proj.w
    d, _, _ = jax_query_weight.shape
    jax_query_weight = jax_query_weight.reshape(d, -1).transpose()
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
    jax_logits = get_per_layer_jax_output(
        model_name, jax_model, seq_len, layer_to_run
    )
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

    # clean up
    tc.delete_directory(model_path)


if __name__ == "__main__":
  absltest.main()
