# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT script for Gemma 3 vision-language model."""

import csv
import dataclasses
import os
from absl import app
from absl import flags
import datasets
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
import qwix
from tunix.generate import sampler as sampler_lib
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
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 4, "Batch size per device.")
_VAL_BATCH_SIZE = flags.DEFINE_integer(
    "val_batch_size", 64, "Eval batch size per device."
)
_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
_MAX_SEQ_LEN = flags.DEFINE_integer(
    "max_seq_len", 768, "Maximum sequence length."
)
_EVAL_EVERY_N_STEPS = flags.DEFINE_integer(
    "eval_every_n_steps", 20, "Evaluate every n steps. Set to -1 to skip eval."
)
_VAL_SPLIT = flags.DEFINE_float("val_split", 0.1, "Validation split.")
_TEST_DATASET_FRAC = flags.DEFINE_float(
    "test_dataset_frac", 0.1, "Test dataset percentage."
)
_QUAL_EVAL_NUM_SAMPLES = flags.DEFINE_integer(
    "qual_eval_num_samples", 5, "Number of samples for qualitative eval."
)
_MAX_STEPS = flags.DEFINE_integer("max_steps", 100, "Maximum training steps.")
_FULL_CKPT_DIR = flags.DEFINE_string(
    "full_ckpt_dir", "full_ckpt_dir", "Directory to save full checkpoints."
)
_LOGGING_DIR = flags.DEFINE_string(
    "logging_dir", "logging_dir", "Directory for logs."
)
_RESULTS_DIR = flags.DEFINE_string(
    "results_dir", "results", "Directory for results."
)
_SAVE_INTERVAL_STEPS = flags.DEFINE_integer(
    "save_interval_steps", 100, "Save interval steps."
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
    data_source,
    image_processor,
    tokenizer,
    batch_size: int,
    num_epochs: int,
    max_seq_len: int,
):
  """Loads the dataset.

  Args:
    data_source: The loaded HF dataset.
    image_processor: Image processor.
    tokenizer: Tokenizer.
    batch_size: Batch size.
    num_epochs: Number of epochs.
    max_seq_len: Maximum sequence length.
    split: Dataset split to load.

  Returns:
    A grain DataLoader.
  """
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
      worker_count=8,
      worker_buffer_size=4,
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
  checkpointing_options = None
  if full_ckpt_dir is not  None:
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=_SAVE_INTERVAL_STEPS.value, max_to_keep=1
    )

  training_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=eval_every_n_steps,
      max_steps=max_steps,
      metrics_logging_options=full_logging_options,
      checkpointing_options=checkpointing_options,
      checkpoint_root_directory=full_ckpt_dir,
  )

  trainer = peft_trainer.PeftTrainer(
      model, optax.adamw(1e-5), training_config
  ).with_gen_model_input_fn(gen_model_input_fn)

  with mesh:
    trainer.train(train_ds, val_ds)


def qual_eval_sample(
    element,
    sampler,
    temperature=None,
    top_k=1,
    top_p=1.0,
    max_generation_steps=768,
):
  """Qualitative evaluation."""

  image = [[element["image"]]]
  text = INPUT_TEMPLATE.replace(
      "<start_of_image>",
      "\n\n<start_of_image>" + "<img>" * 256 + "<end_of_image>\n\n",
  )

  out = sampler(
      input_strings=text,
      images=image,
      max_generation_steps=max_generation_steps,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=True,
      seed=_SEED.value,
      eos_tokens=[1, 106],
  )
  return out.text[0]


def run_qual_eval(
    model, tokenizer, image_processor, model_config, test_data_source, prefix
):
  """Runs qualitative evaluation."""
  print(f"Running qualitative evaluation {prefix} training...")
  sampler = sampler_lib.Sampler(
      transformer=model,
      tokenizer=tokenizer,
      cache_config=sampler_lib.CacheConfig(
          cache_size=1300,
          num_layers=model_config.num_layers,
          num_kv_heads=model_config.num_kv_heads,
          head_dim=model_config.head_dim,
      ),
      image_processor=image_processor,
  )

  eval_dir = os.path.join(_RESULTS_DIR.value, f"{prefix}_training")
  os.makedirs(eval_dir, exist_ok=True)
  csv_path = os.path.join(eval_dir, "results.csv")

  with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "generated_text"])
    for i in range(_QUAL_EVAL_NUM_SAMPLES.value):
      element = test_data_source[i]
      out_text = qual_eval_sample(element, sampler)

      img = element["image"]
      img_path = os.path.join(eval_dir, f"eval_image_{i}.png")
      img.save(img_path)

      writer.writerow([img_path, out_text])


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
  do_eval = _EVAL_EVERY_N_STEPS.value > 0
  val_dataset = None
  train_data_source = datasets.load_dataset("unsloth/LaTeX_OCR", split="train")
  if do_eval:
    split_dataset = train_data_source.train_test_split(
        test_size=0.1, seed=_SEED.value
    )
    train_data_source = split_dataset["train"]
    val_data_source = split_dataset["test"]

    val_dataset = load_dataset(
        data_source=val_data_source,
        image_processor=image_processor,
        tokenizer=tokenizer,
        batch_size=_VAL_BATCH_SIZE.value,
        num_epochs=1,
        max_seq_len=_MAX_SEQ_LEN.value,
    )

  train_dataset = load_dataset(
      data_source=train_data_source,
      image_processor=image_processor,
      tokenizer=tokenizer,
      batch_size=_BATCH_SIZE.value,
      num_epochs=_NUM_EPOCHS.value,
      max_seq_len=_MAX_SEQ_LEN.value,
  )

  test_data_source = datasets.load_dataset(
      "unsloth/LaTeX_OCR", split=f"test[:{int(_TEST_DATASET_FRAC.value*100)}%]"
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

  run_qual_eval(
      model,
      tokenizer,
      image_processor,
      model_config,
      test_data_source,
      prefix="before",
  )

  # Train.
  full_ckpt_dir = _FULL_CKPT_DIR.value
  if full_ckpt_dir.lower() == "none":
    full_ckpt_dir = None
  train(
      train_dataset,
      val_dataset,
      tokenizer,
      model,
      eval_every_n_steps=_EVAL_EVERY_N_STEPS.value,
      max_steps=_MAX_STEPS.value,
      full_ckpt_dir=full_ckpt_dir,
      logging_dir=_LOGGING_DIR.value,
      mesh=mesh,
  )

  run_qual_eval(
      model,
      tokenizer,
      image_processor,
      model_config,
      test_data_source,
      prefix="after",
  )


if __name__ == "__main__":
  app.run(main)
