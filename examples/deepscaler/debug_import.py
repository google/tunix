import sys
import os
import site
import grain

print("--- Python Debug Info ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")

print("\nsys.path:")
for path in sys.path:
    print(f"  - {path}")

print("\nsite-packages:")
for path in site.getsitepackages():
    print(f"  - {path}")

print("--- End Debug Info ---\n")

try:
    print("Attempting import: from tunix.rl.agentic.agents import model_agent")
    from tunix.rl.agentic.agents import model_agent
    print("SUCCESS: Imported model_agent")
except ModuleNotFoundError as e:
    print(f"FAILED to import: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()



DEEPSCALER_DATA_PATH = os.path.join("gs://linchai-bucket-dev/rl/data/", "DeepScaleR-Preview-Dataset/deepscaler.json")

import datasets as datasets_lib
import pandas as pd
import fsspec
import transformers
Dataset = datasets_lib.Dataset


file_open = fsspec.open

AutoTokenizer = transformers.AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
def create_datasets(
    train_ds_path: str = DEEPSCALER_DATA_PATH
):
  def preprocess_fn(example, index):
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  # with file_open(train_ds_path) as train_f, file_open(test_ds_path, 'rb') as test_f:
  with file_open(train_ds_path) as train_f:
    train_df = pd.read_json(train_f)

  train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)
  print("from pandas for train_ds done.")
  
  def process_item(item):
      question = item["question"]
      answer = item["answer"]

      instruction = "Let's think step by step, and put your final answer within \\boxed{}."
      prompt = f"{question} {instruction}"
      prompt = tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False, add_generation_prompt=True)

      return {
          "prompts": prompt,
          "question": question,
          "answer": answer,
      }

  train_ds = grain.MapDataset.source(train_ds).map(process_item)
  return train_ds


train_ds = create_datasets()
print("train_ds example:", train_ds[0])
