"""Build a FIXED workload of REAL SWE-trajectory token sequences for reproducible probing.

Loads N real trajectories from a HF dataset (deepswe's data class), takes the first seq_len
tokens of each (full-length only -> no padding, all real content), saves [N, seq_len] token IDs.
Run ONCE (no TPU needed -- just tokenization):
  python3 build_workload.py --n 128 --seq_len 4096 --out workload.npz

Reusing this fixed workload across all runs = stable, reproducible comparison; for the batch
experiment it is REQUIRED (the target sequence must be identical across batch sizes).
"""
import argparse
import numpy as np


def _row_to_text(row, tok):
  """Prefer the chat-templated messages (faithful to what the model is fed); fall back to text."""
  msgs = row.get("messages")
  if msgs:
    try:
      return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
      import json
      return json.dumps(msgs)
  return row.get("text") or ""


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--dataset", default="SWE-bench/SWE-smith-trajectories")
  p.add_argument("--model_path", default="/mnt/disks/tunix-data/models/Qwen3-32B")
  p.add_argument("--n", type=int, default=128, help="number of trajectory sequences")
  p.add_argument("--seq_len", type=int, default=4096, help="tokens per sequence (real prefix, no padding)")
  p.add_argument("--out", default="workload.npz")
  p.add_argument("--max_scan", type=int, default=4000, help="max rows to scan for full-length trajectories")
  a = p.parse_args()

  from transformers import AutoTokenizer
  from datasets import load_dataset
  tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, trust_remote_code=True)
  ds = load_dataset(a.dataset, streaming=True)
  split = "train" if "train" in ds else list(ds.keys())[0]

  seqs, scanned, short = [], 0, 0
  for row in ds[split]:
    scanned += 1
    ids = tok(_row_to_text(row, tok), return_tensors=None)["input_ids"]
    if len(ids) >= a.seq_len:
      seqs.append(ids[:a.seq_len])          # real prefix, exactly seq_len, no padding
    else:
      short += 1
    if len(seqs) >= a.n or scanned >= a.max_scan:
      break

  if len(seqs) < a.n:
    raise SystemExit(f"only {len(seqs)} trajectories >= {a.seq_len} tokens in {scanned} scanned "
                     f"({short} too short). Lower --seq_len or --n, or raise --max_scan.")
  arr = np.asarray(seqs, dtype=np.int32)     # [n, seq_len]
  np.savez(a.out, tokens=arr)
  print(f"[build_workload] saved {arr.shape} -> {a.out} "
        f"(scanned {scanned} rows, {short} too short, dataset={a.dataset})")


if __name__ == "__main__":
  main()
