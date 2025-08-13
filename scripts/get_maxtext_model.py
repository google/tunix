from tunix.generate.vllm_sampler import VllmSampler, MappingConfig
import jax
import transformers


model_tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

TOTAL_TPU_TO_USE = 8
MESH = [(1, TOTAL_TPU_TO_USE), ("fsdp", "tp")]  # YY

mesh =   model_mesh = jax.make_mesh(*MESH, devices=jax.devices()[:TOTAL_TPU_TO_USE])


sampler = VllmSampler(
    mesh=mesh,
    tokenizer=model_tokenizer,
    max_model_len=1024,
    model_version="meta-llama/Llama-3.1-8b",
    mapping_config=MappingConfig({}, {}, {}, None),
    hbm_utilization=0.3,  # Adjust based on your TPU memory
)

print(sampler([ "Hello, my dog is cute", "Hello, my cat is cute"  ], 10))