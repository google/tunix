"""Training script for Gemma 3 VLM."""

import dataclasses
from absl import app
from absl import flags
import datasets
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as model_lib
from tunix.models.gemma3 import params as params_lib
from tunix.processors import image_processor as image_processor_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer


INPUT_TEMPLATE = (
    "<start_of_turn>user\n<start_of_image>Write the LaTeX representation"
    " for this image.<end_of_turn>\n<start_of_turn>model\n"
)
MODEL_CKPT_PATH = "gs://gemma-data/checkpoints/gemma3-4b-pt"
TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

_SEED = flags.DEFINE_integer("seed", 42, "Random seed.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8, "Batch size per device.")
_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
_MAX_SEQ_LEN = flags.DEFINE_integer(
    "max_seq_len", 768, "Maximum sequence length."
)
_EVAL_EVERY_N_STEPS = flags.DEFINE_integer(
    "eval_every_n_steps", 20, "Evaluate every n steps."
)
_MAX_STEPS = flags.DEFINE_integer("max_steps", 100, "Maximum training steps.")
_FULL_CKPT_DIR = flags.DEFINE_string(
    "full_ckpt_dir", "full_ckpt_dir", "Directory to save full checkpoints."
)
_LOGGING_DIR = flags.DEFINE_string(
    "logging_dir", "logging_dir", "Directory for logs."
)

# LoRA flags
_LORA_RANK = flags.DEFINE_integer("lora_rank", 8, "LoRA rank.")
_LORA_ALPHA = flags.DEFINE_integer("lora_alpha", 8, "LoRA alpha.")
_LORA_TARGET_MODULES = flags.DEFINE_string(
    "lora_target_modules",
    r".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj|.*query_proj|.*key_proj|.*value_proj|.*out_proj|.*fc1|.*fc2",
    "LoRA target modules regex.",
)


@dataclasses.dataclass
class LoraConfig:
  """Configuration for LoRA (Low-Rank Adaptation).

  Attributes:
    rank: LoRA rank.
    alpha: LoRA alpha.
    target_modules: Regex for target modules.
  """

  rank: int = 8
  alpha: int = 8
  target_modules: str = (
      r".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj|.*query_proj|.*key_proj|.*value_proj|.*out_proj|.*fc1|.*fc2"
  )


class _Preprocess(grain.MapTransform):
  """Preprocess the image and tokenize the input."""

  def __init__(self, image_processor, tokenizer):
    self.image_processor = image_processor
    self.tokenizer = tokenizer

  def map(self, element):
    """Preprocess the image and tokenize the input."""
    image = np.array(self.image_processor(element["image"])[0])

    src = INPUT_TEMPLATE.replace(
        "<start_of_image>",
        "\n\n<start_of_image>" + "<img>" * 256 + "<end_of_image>\n\n",
    )
    src_tokens = self.tokenizer.tokenize(src, add_eos=False)
    dst_tokens = self.tokenizer.tokenize(element["text"], add_eos=True)
    return image, src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self.max_seq_len = max_seq_len
    self.pad_value = pad_value

  def map(self, element):
    images, src_tokens, dst_tokens = element
    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = np.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self.pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return peft_trainer.TrainingInput(
        input_tokens=tokens, input_mask=mask, images=images
    )

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self.max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  """Filter out overlength examples."""

  def __init__(self, max_seq_len):
    self.max_seq_len = max_seq_len

  def filter(self, element) -> bool:
    return element.input_tokens.shape[0] <= self.max_seq_len


def load_data_processors(model_config):
  """Loads data processors.

  Args:
    model_config: Model configuration.

  Returns:
    A tuple of (image_processor, tokenizer).
  """
  image_processor = image_processor_lib.ImageProcessor(
      config=model_config.vision_config
  )
  tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=TOKENIZER_PATH)
  return image_processor, tokenizer


def load_dataset(
    image_processor,
    tokenizer,
    batch_size: int,
    num_epochs: int,
    max_seq_len: int,
    split="train",
):
  """Loads the dataset.

  Args:
    image_processor: Image processor.
    tokenizer: Tokenizer.
    batch_size: Batch size.
    num_epochs: Number of epochs.
    max_seq_len: Maximum sequence length.
    split: Dataset split to load.

  Returns:
    A grain DataLoader.
  """
  data_source = datasets.load_dataset("unsloth/LaTeX_OCR", split=split)

  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Preprocess(image_processor, tokenizer),
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )


def load_model(
    model_config,
    mesh,
    lora_config: LoraConfig | None = None,
):
  """Loads the model and optionally applies LoRA.

  Args:
    model_config: The model configuration.
    mesh: The mesh for sharding.
    lora_config: Optional LoRA configuration.

  Returns:
    The loaded model, potentially with LoRA applied.
  """
  model = params_lib.create_model_from_checkpoint(
      checkpoint_path=MODEL_CKPT_PATH,
      model_config=model_config,
      mesh=mesh,
      dtype=jnp.bfloat16,
  )

  if lora_config is not None:
    lora_provider = qwix.LoraProvider(
        module_path=lora_config.target_modules,
        rank=lora_config.rank,
        alpha=lora_config.alpha,
    )
    # We need dummy input to trace the graph for LoRA application.
    model = qwix.apply_lora_to_model(
        model, lora_provider, **model.get_model_input()
    )

  return model


def train(
    train_ds,
    val_ds,
    tokenizer,
    model,
    eval_every_n_steps,
    max_steps,
    full_ckpt_dir,
    logging_dir,
    mesh,
):
  """Training loop.

  Args:
    train_ds: Training dataset.
    val_ds: Validation dataset.
    tokenizer: Tokenizer.
    model: Model to train.
    eval_every_n_steps: Evaluation frequency.
    max_steps: Maximum training steps.
    full_ckpt_dir: Checkpoint directory.
    logging_dir: Log directory.
    mesh: Device mesh.
  """

  def gen_model_input_fn(x):
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions, attention_mask = model.get_positions_and_attention_mask(
        x.input_tokens,
        inputs_mask=pad_mask,
    )
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
        "images": x.images,
    }

  full_logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir=logging_dir, flush_every_n_steps=20
  )

  training_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=eval_every_n_steps,
      max_steps=max_steps,
      metrics_logging_options=full_logging_options,
      checkpoint_root_directory=full_ckpt_dir,
  )

  trainer = peft_trainer.PeftTrainer(
      model, optax.adamw(1e-5), training_config
  ).with_gen_model_input_fn(gen_model_input_fn)

  with mesh:
    trainer.train(train_ds, val_ds)


def main(argv):
  del argv  # Unused.

  num_tpus = len(jax.devices())
  # Define mesh.
  if num_tpus == 8:
    mesh_counts = (1, 4)
  elif num_tpus == 1:
    mesh_counts = (1, 1)
  else:
    raise ValueError(f"Unsupported number of TPUs: {num_tpus}")

  mesh = [mesh_counts, ("fsdp", "tp")]
  mesh = jax.make_mesh(
      *mesh, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh[0])
  )

  # Model config.
  model_config = model_lib.ModelConfig.gemma3_4b_pt(text_only=False)

  # Load preprocessors.
  image_processor, tokenizer = load_data_processors(model_config)

  # Load the dataset.
  train_dataset = load_dataset(
      image_processor=image_processor,
      tokenizer=tokenizer,
      batch_size=_BATCH_SIZE.value,
      num_epochs=_NUM_EPOCHS.value,
      max_seq_len=_MAX_SEQ_LEN.value,
      split="train",
  )
  val_dataset = load_dataset(
      image_processor=image_processor,
      tokenizer=tokenizer,
      batch_size=_BATCH_SIZE.value,
      num_epochs=1,
      max_seq_len=_MAX_SEQ_LEN.value,
      split="test",
  )

  # Lora Config
  lora_config = LoraConfig(
      rank=_LORA_RANK.value,
      alpha=_LORA_ALPHA.value,
      target_modules=_LORA_TARGET_MODULES.value,
  )

  # Load the model.
  model = load_model(
      model_config=model_config,
      mesh=mesh,
      lora_config=lora_config,
  )

  # Train.
  train(
      train_dataset,
      val_dataset,
      tokenizer,
      model,
      eval_every_n_steps=_EVAL_EVERY_N_STEPS.value,
      max_steps=_MAX_STEPS.value,
      full_ckpt_dir=_FULL_CKPT_DIR.value,
      logging_dir=_LOGGING_DIR.value,
      mesh=mesh,
  )


if __name__ == "__main__":
  app.run(main)
