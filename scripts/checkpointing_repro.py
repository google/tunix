import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../maxtext")))

import functools


import jax
from flax import nnx
import flax.linen as nn

import MaxText as mt
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adaptor import TunixMaxTextLlama

from tunix.models.llama3 import model as llama3_lib
import orbax.checkpoint as ocp

def get_ref_maxtext_model(config):

  # python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_DIRECTORY} dataset_path=${DATASET_PATH} tokenizer_path=assets/tokenizer.gemma load_parameters_path=${CONVERTED_CHECKPOINT} per_device_batch_size=1 run_name=${FINETUNE_RUN_NAME} max_target_length=8192 steps=10 async_checkpointing=false model_name=gemma-2b checkpoint_period=5

  def create_model(config):
    return mt.from_pretrained(config, rngs=nnx.Rngs(params=0, dropout=1))

  abstract_model = nnx.eval_shape(create_model, config=config)
  graphdef, abstract_state = nnx.split(abstract_model)
  print("The abstract NNX state (all leaves are abstract arrays):")
  nnx.display(abstract_state)
  specs = nnx.get_partition_spec(abstract_state)
  mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  with nn.logical_axis_rules(config.logical_axis_rules):
    out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

  @functools.partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    # This will be JIT-compiled. JAX knows the output sharding and can
    # initialize the parameters directly on the target devices in a sharded way.
    model = create_model(config)
    return nnx.state(model)

  with mesh:

    # Create the model with sharded parameters.
    sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    tunix_model = TunixMaxTextLlama(
        base_model=model,
        use_attention_mask=False,  # trust Tunix loss masking
    )

    model_config = llama3_lib.ModelConfig.llama3_1_8b()
    tunix_model.config = model_config

  return tunix_model, mesh, model_config


config_ref = pyconfig.initialize(
    [
        "",
        "../maxtext/MaxText/configs/base.yml",
    ],  # TODO: @mazumdera: why decode.py?
    base_output_directory="gs://dummy_output_dir",  # This is not used in Tunix.
    run_name="test-tunix-maxtext-llama3.1-8b",
    # run_name="test-tunix-maxtext-llama3.1-8b",
    # dataset_path=we use Tunix's dataset
    # TODO: @mazumdera: change this to use checkpoint
    tokenizer_type="tiktoken",
    tokenizer_path="assets/tokenizer_llama3.tiktoken",
    load_parameters_path="",
    # tokenizer_path="assets/tokenizer.gemma",
    per_device_batch_size=1,
    max_prefill_predict_length=4,
    max_target_length=16,
    steps=10,
    async_checkpointing="false",
    model_name="llama3.1-8b",
    # model_name="gemma-2b",
    checkpoint_period=5,
    skip_jax_distributed_system="true",
    weight_dtype="bfloat16",
    attention="dot_product",
    remat_policy="custom",
    decoder_layer_input="offload",
    query_proj="offload",
    key_proj="offload",
    value_proj="offload",
    opt_type="sgd",
)

# MaxText model
maxtext_model, mesh, model_config = get_ref_maxtext_model(config_ref)
nnx.display(maxtext_model)

checkpoint_path = "gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items"
target_for_restore = jax.tree.map(
        lambda v: v.value,
        nnx.state(maxtext_model),
        is_leaf=lambda n: isinstance(n, nnx.Variable),
    )

try:
    checkpointer = ocp.PyTreeCheckpointer()
    loaded_a = checkpointer.restore(checkpoint_path, item=target_for_restore)
except Exception as e:
    print(f"Failed: {e}")

############################################## ERROR #########################################################
 
# User-provided restore item and on-disk value metadata tree structures do not match: {'base': Diff(lhs={'decoder': {'decoder_norm': {'scale': Array([1, 1, 1, ..., 1, 1, 1], dtype=bfloat16)}, 'layers': {'mlp': {'wi_0': {'kernel': Array([[[-0.0246582, 0.0134888, 0.0126343, ..., 0.00717163, -0.00267029,
#          0.0105591],
#         [-0.0266113, 0.0202637, -0.00334167, ..., 0.0090332,
#          -0.00436401, -0.0109863],
#         [0.00233459, 0.00787354, 0.00540161, ..., 0.0332031, 0.0223389,
#          0.0202637],
#         ...,
#         [-0.00976562, -0.000331879, -0.0354004, ..., 0.013916,
#          0.0130615, -0.0275879],
#         [-0.013916, -0.0314941, 0.00939941, ..., -0.0158691, 0.00787354,
# ...
#        [0.365234, -0.796875, -1.65625, ..., 0.808594, -1.95312,
#         0.0441895]], dtype=bfloat16)}}, rhs=None), 'step': Diff(lhs=None, rhs=ValueMetadataEntry(value_type='scalar', skip_deserialize=False, write_shape=None)), 'params': Diff(lhs=None, rhs={'params': {'decoder': {'decoder_norm': {'scale': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'layers': {'mlp': {'wi_0': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'wi_1': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'wo': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}}, 'post_self_attention_layer_norm': {'scale': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'pre_self_attention_layer_norm': {'scale': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'self_attention': {'key': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'out': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'query': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}, 'value': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}}}, 'logits_dense': {'kernel': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}}, 'token_embedder': {'embedding': ValueMetadataEntry(value_type='jax.Array', skip_deserialize=False, write_shape=None)}}}), 'opt_state': Diff(lhs=None, rhs=ValueMetadataEntry(value_type='Dict', skip_deserialize=True, write_shape=None))}

# Approach B: Load without target structure