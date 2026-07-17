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

"""VTC GSM8K data module for the standard GRPO CLI path (recipe alignment).

Ports the exact prompt construction of the agentic demo
(examples/math_gsm8k/qwen3_grpo_demo.py: VTC_PROMPT_TEMPLATE / build_prompt /
build_gsm8k_dataset) so the standard `grpo_main` path trains on the SAME
raw-text prompt (pre-opened <reasoning>, NO chat template) as the converging
agentic run. Load via:

    data_module="experimental/vtc_data.py"
    apply_chat_template_to_dataset=false

The CLI loader calls `create_dataset(**data_config)` (tunix/cli/utils/
data.py get_dataset_from_module) and then post_init_dataset does the
length-filtering / fraction split / epoch repeat / batching — so this module
returns an UNBATCHED, UNREPEATED grain.MapDataset with fields
{prompts, question, answer}, shuffled with the demo's seed.
"""

from typing import Any

import grain
import tensorflow_datasets as tfds

# For OSS usage: registers the `gsm8k` tfds builder (else tfds.data_source
# raises DatasetNotFoundError). Same as qwen3_grpo_demo.py.
import tensorflow_datasets.text.gsm8k  # noqa: F401  pylint: disable=unused-import

SEED = 42

# Byte-exact copy of qwen3_grpo_demo.py::VTC_PROMPT_TEMPLATE.
VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""


def _as_text(value: Any) -> str:
  # Ported from qwen3_grpo_demo.py::_as_text.
  return value if isinstance(value, str) else value.decode("utf-8")


def _extract_hash_answer(text: str) -> str | None:
  # Ported from qwen3_grpo_demo.py::extract_hash_answer (keeps commas; the
  # VTC reward normalizes commas at comparison time).
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def _build_prompt(question: str) -> str:
  # Ported from qwen3_grpo_demo.py::build_prompt.
  return VTC_PROMPT_TEMPLATE.format(question)


def create_dataset(
    split: str = "train",
    seed: int = SEED,
    data_dir: str | None = None,
    shuffle: bool = True,
    **kwargs,
) -> grain.MapDataset:
  """GSM8K -> grain.MapDataset{prompts, question, answer}, demo-identical.

  Mirrors qwen3_grpo_demo.py::build_gsm8k_dataset minus `.batch()` (the CLI
  post_init_dataset batches) and minus `.repeat()` (num_train_epochs handles
  it).
  """
  del kwargs  # Unused extras from data_config.
  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )

  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)

  dataset = dataset.map(
      lambda x: {
          "prompts": _build_prompt(_as_text(x["question"])),
          "question": _as_text(x["question"]),
          "answer": _extract_hash_answer(_as_text(x["answer"])),
      }
  )
  return dataset
