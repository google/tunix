# Copyright 2025 Google LLC
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

"""Data loading and preprocessing."""

from collections.abc import Iterable
from typing import Any, Dict, List 

import datasets
from etils import epath
from grain import python as grain
import numpy as np
import tensorflow_datasets as tfds
import transformers
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member

import sentencepiece as spm

INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}

INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}


class GemmaTokenizer(spm.SentencePieceProcessor):
  """Tokenizing and encoding/decoding text using the Sentencepiece tokenizer."""

  _GEMMA2_TOKENIZER_PATH: epath.PathLike = (
      'gs://gemma-data/tokenizers/tokenizer_gemma2.model'
  )

  def __init__(self, model_path: str = _GEMMA2_TOKENIZER_PATH):
    model_proto = epath.Path(model_path).read_bytes()
    super().__init__()
    self.LoadFromSerializedProto(model_proto)

  def tokenize(
      self,
      example: str,
      prefix: str = "",
      suffix: str = "",
      add_eos: bool = True,
  ) -> np.ndarray:
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self.bos_id()]
    int_list.extend(self.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())

    return np.array(int_list, dtype=np.int32)


def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: Any,
    instruct_tuned: bool = False,
    input_template: dict[str, str] | None = None,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  """Creates train and eval data iterator.

  Args:
    dataset_name: The name of the dataset to use.
    global_batch_size: The global batch size to use for both train and eval.
    max_target_length: The maximum length of the target sequence.
    num_train_epochs: The number of epochs to use for training. If None, the
      dataset will be repeated indefinitely.
    tokenizer: The tokenizer to use for tokenizing the dataset.
    instruct_tuned: Whether the dataset should be instruct tuned.
    input_template: The input template to use for the dataset.

  Returns:
    A tuple of train and eval data iterators.
  """
  train_ds = None
  eval_ds = None
  if dataset_name == "mtnt/en-fr":
    train_ds, eval_ds = tfds.data_source(dataset_name, split=("train", "valid"))
  elif dataset_name == "Helsinki-NLP/opus-100":  # Hugging Face dataloader
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, data_dir="en-fr", split=("train", "validation")
    )
  elif dataset_name == "HuggingFaceH4/ultrachat_200k":
    train_ds = datasets.load_dataset(dataset_name, data_dir="", split=("train_sft"))
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  input_template = INPUT_TEMPLATE_IT if instruct_tuned else INPUT_TEMPLATE

  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  if not eval_ds:
    return train_loader
  else:
    eval_loader = _build_data_loader(
        data_source=eval_ds,
        batch_size=global_batch_size,
        num_epochs=1,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        input_template=input_template,
    )
  return train_loader, eval_loader

def _get_pad_id(tokenizer: GemmaTokenizer | transformers.PreTrainedTokenizerFast):
  """
  Returns the padding token ID from the tokenizer.

  Args:
      tokenizer: The tokenizer instance.

  Returns:
      The integer ID of the padding token.

  Raises:
      TypeError: If the tokenizer type is not supported.
      AttributeError: If the expected pad ID attribute/method is missing.
  """
  if GemmaTokenizer and isinstance(tokenizer, GemmaTokenizer):

      if callable(getattr(tokenizer,'pad_id')):
          return tokenizer.pad_id()
      else:
            # Default for SentencePiece base is often 0, but can vary.
            print("Warning: .pad_id attribute not found on Keras GemmaTokenizer.")
            return 0 # Fallback, adjust as needed

  elif transformers.PreTrainedTokenizerFast and isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
      print("Tokenizer is PreTrainedTokenizerFast")
      # Hugging Face tokenizers typically use .pad_token_id attribute
      if hasattr(tokenizer, 'pad_token_id'):
        if tokenizer.pad_token_id is not None:
          return tokenizer.pad_token_id
        else:
          return 0
        
      else:
          raise AttributeError("PreTrainedTokenizerFast instance is missing 'pad_token_id' attribute")

  else:
      tokenizer_type = type(tokenizer).__name__
      # Fallback for other types or if imports failed
      print(f"Warning: Unknown or unimported tokenizer type: {tokenizer_type}. Trying common attributes.")
      if hasattr(tokenizer, 'pad_id'):
          prop = getattr(tokenizer, 'pad_id')
          return prop() if callable(prop) else prop
      elif hasattr(tokenizer, 'pad_token_id'):
          return tokenizer.pad_token_id
      else:
          raise TypeError(f"Unsupported tokenizer type: {tokenizer_type}, and could not find pad ID attribute.")

def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: Any,
    input_template: dict[str, str],
) -> grain.DataLoader:
  """Builds a data loader for the given data source."""
  
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Tokenize(tokenizer, input_template),
          _BuildTrainInput(max_seq_len, _get_pad_id(tokenizer)),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )

def combine_columns(example, columns, data_column):
  """Combine columns such as 'prompt' and 'completion' for sft training"""
  assert len(columns) > 1
  combined = []
  for i in range(len(example[columns[0]])):
    for c in columns:
      combined.append(example[c][i])
  example[data_column] = combined
  return example


def tokenize_conversational_sft(
    example: Dict[str, Any],
    tokenizer: Any,  # Type hint for a Hugging Face tokenizer
    data_column_name: str,
    max_length: int = 2048
) -> Dict[str, List[List[int]]]:
    """Formats and tokenizes conversational data into src/dst pairs for SFT.

    Each assistant message in the conversation becomes a target (dst_tokens),
    with the full preceding conversation history forming the source (src_tokens).

    Args:
        example: A dictionary containing conversational data. It is expected to
            have a key specified by `data_column_name` that holds a list of
            messages. Each message is a dict with "role" and "content".
        tokenizer: The tokenizer instance (e.g., from Hugging Face Transformers)
            which has the `apply_chat_template` and `encode` methods, and
            an `eos_token_id`.
        data_column_name: The name of the column in the `example` dictionary
            that contains the list of messages (e.g., "messages").
        max_length: Maximum sequence length for truncation in tokenization.

    Returns:
        A dictionary containing two keys:
          'src_tokens': A list of token lists. Each inner list contains the
                        token IDs representing the input to the model for a turn.
          'dst_tokens': A list of token lists. Each inner list contains the
                        token IDs representing the target output the model
                        should predict for the corresponding turn.
        An empty list for a key means no valid turns of that type were found.
    """
    messages = example.get(data_column_name, [])
    src_tokens_all_turns = []
    dst_tokens_all_turns = []

    if not isinstance(messages, list):
         # max_logging.log(f"Warning: Expected list for {data_column_name}, got {type(messages)}")
         print(f"Warning: Expected list for {data_column_name}, got {type(messages)}")
         return {'src_tokens': [], 'dst_tokens': []}

    for i in range(len(messages)):
        current_message = messages[i]
        if current_message.get("role") == "assistant":
            if i == 0:
                # Assistant message should not be the first in a conversation.
                # max_logging.log("Warning: Assistant message at index 0, skipping.")
                print("Warning: Assistant message at index 0, skipping.")
                continue

            context_messages = messages[:i]
            target_message = current_message

            # Ensure the message before the assistant's is from the user.
            if not context_messages or context_messages[-1].get("role") != "user":
                # max_logging.log(f"Warning: Assistant message at index {i} not preceded by user message. Skipping turn.")
                print(f"Warning: Assistant message at index {i} not preceded by user message. Skipping turn.")
                continue

            try:
                # --- Source Tokens ---
                # Apply the tokenizer's chat template to the context messages.
                # add_generation_prompt=True adds the necessary tokens to indicate
                # the start of the assistant's turn.
                src_tokens = tokenizer.apply_chat_template(
                    context_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    truncation=True,
                    max_length=max_length,
                )
                src_tokens_all_turns.append(src_tokens)

                # --- Destination Tokens ---
                # Tokenize *only* the content of the assistant's message.
                # Target sequences for language modeling usually end with an EOS token.
                dst_tokens = tokenizer.encode(
                    target_message.get("content", ""),
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False  # Manually add EOS to be explicit.
                )
                # Add EOS token ID.
                if tokenizer.eos_token_id is not None:
                    dst_tokens.append(tokenizer.eos_token_id)

                dst_tokens_all_turns.append(dst_tokens)

            except Exception as e:
                # max_logging.log(f"Error during tokenization for turn at index {i}: {e}")
                print(f"Error during tokenization for turn at index {i}: {e}")
                # Skip this turn on error

    return {
        "src_tokens": src_tokens_all_turns,
        "dst_tokens": dst_tokens_all_turns,
    }
    
class _Tokenize(grain.MapTransform):
  """Tokenize the input."""

  def __init__(self, tokenizer: Any, input_template: dict[str, str]):
    self._tokenizer = tokenizer
    self._input_template = input_template

  def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the input."""
    if "src" in element.keys():  ## MTNT dataset
      src_tokens = self._tokenizer.tokenize(
          element["src"].decode(),
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["dst"].decode(), add_eos=True
      )
    elif "translation" in element.keys():
        ## OPUS-100 dataset
      src_tokens = self._tokenizer.tokenize(
          element["translation"]["en"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["translation"]["fr"], add_eos=True
      )
    else:
      result = tokenize_conversational_sft(self._input_template,self._tokenizer, "messages", 1024)
      src_tokens = result["src_tokens"]
      dst_tokens = result["dst_tokens"]
      
      src_tokens = np.array(src_tokens, dtype=np.int32)
      dst_tokens = np.array(dst_tokens, dtype=np.int32)
      
    return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
    src_tokens, dst_tokens = tokens

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
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return TrainingInput(input_tokens=tokens, input_mask=mask)

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  """Filter out overlength examples."""

  def __init__(self, max_seq_len: int):
    self._max_seq_len = max_seq_len

  def filter(self, element: TrainingInput) -> bool:
    return element.input_tokens.shape[0] <= self._max_seq_len
