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
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params_safetensors as gemma3_params
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


class GemmaAlignTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma3_270m_it",
          model_name="google/gemma-3-270m-it",
          model_config=gemma3_model.ModelConfig.gemma3_270m,
          tolerance=1e-3,
      ),
  )
  def test_gemma_model_alignment(self, model_name, model_config, tolerance):
    model_path = os.path.join(tempfile.gettempdir(), "models", model_name)

    tc.download_from_huggingface(repo_id=model_name, model_path=model_path)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    )
    print("HF model loaded.")

    jax_model = gemma3_params.create_model_from_safe_tensors(
        model_path,
        model_config(),
        mesh=jax.make_mesh((1, 1), ("fsdp", "tp")),
        dtype=jnp.float32,
    )
    print("JAX model loaded.")

    # Make sure model weights are the same (only check the first query weight)
    hf_emb_weight = hf_model.get_decoder().embed_tokens.weight.detach().numpy()
    jax_emb_weight = jax_model.embedder.input_embedding.value
    np.testing.assert_equal(
        hf_emb_weight,
        jax_emb_weight,
        err_msg=(
            "Embedding weights are not equal, are you sure the loaded model"
            " weight between HF and JAX is identical?"
        ),
    )
    hf_query_weight = (
        hf_model.get_decoder()
        .layers[0]
        .self_attn.q_proj.weight.detach()
        .numpy()
    )
    jax_query_weight = jax_model.layers[0].attn.q_einsum.w
    _, d, _ = jax_query_weight.shape
    jax_query_weight = jax_query_weight.transpose(0, 2, 1).reshape(-1, d)
    np.testing.assert_equal(
        hf_query_weight,
        jax_query_weight,
        err_msg=(
            "Query weights are not equal, are you sure the loaded model weight"
            " between HF and JAX is identical?"
        ),
    )
    print("Model weights check passed :)")

    seq_len = 128

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
