import grain
from grain.multiprocessing import MultiprocessingOptions
from transformers import AutoTokenizer


def get_dataset_from_parquet(parquet_path, tokenizer):
  dataset = grain.experimental.ParquetIterDataset(parquet_path)
  dataset = dataset.map(
      lambda x: {
          "prompts": tokenizer.apply_chat_template(
              x["prompt"], tokenize=False, add_generation_prompt=True
          ),
          **{k: v for k, v in x.items() if k != "prompt"},
      }
  )
  return dataset
