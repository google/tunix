from flax import nnx
from huggingface_hub import snapshot_download
import jax
import qwix
from transformers import AutoTokenizer
from tunix.models.llama3 import model as llama3_lib
from tunix.models.llama3 import params as llama3_params


def get_model(model_config):
  local_dir = snapshot_download(repo_id=model_config["name"])
  mesh_shape = model_config["mesh"]
  mesh = jax.make_mesh(
      (mesh_shape.get("fsdp"), mesh_shape.get("tp")),
      ("fsdp", "tp"),
      devices=jax.devices(),
  )
  # TODO: support other models
  match model_config["name"]:
    case "meta-llama/Llama-3.2-1B-Instruct":
      config = llama3_lib.ModelConfig.llama3_2_1b()
      params = llama3_params
    case _:
      raise ValueError(f"Model {model_config['name']} not supported.")
  model = params.create_model_from_safe_tensors(local_dir, config, mesh)
  tokenizer = AutoTokenizer.from_pretrained(local_dir)
  return model, tokenizer, mesh


def get_lora_model(base_model, rank, alpha, model_mesh=None):
  """Creates a LoRA model from a base model.

  Args:
    base_model: The base model to apply LoRA to.
    rank: The rank of the LoRA layers.
    alpha: The alpha of the LoRA layers.
    model_mesh: The mesh to use for sharding the model.

  Returns:
    A LoRA model.
  """
  if isinstance(base_model, llama3_lib.Llama3):
    module_path = (
        ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
    )
  else:
    module_path = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"

  lora_provider = qwix.LoraProvider(
      module_path=(module_path),
      rank=rank,
      alpha=alpha,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with model_mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model
