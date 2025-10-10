# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""VLM (PaLI-Gemma) data loading and preprocessing for RL/VQA."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Dict, Optional

import json
import os
import numpy as np
from etils import epath
from grain import python as grain
import tensorflow_datasets as tfds

# Optional: PIL for local file images (JSONL mode). If you prefer TF ops, swap in tf.io / tf.image.
try:
  from PIL import Image
  _HAS_PIL = True
except Exception:  # pragma: no cover
  _HAS_PIL = False

# ---- Templates (similar spirit to Gemma’s templates) ----
INPUT_TEMPLATE_VLM = {
    "prefix": "Look at the image and answer the question:\n",
    "suffix": "\n",
}
INPUT_TEMPLATE_VLM_IT = {
    "prefix": (
        "<start_of_turn>user\n"
        "Look at the image and answer the question:\n"
    ),
    "suffix": (
        "\nPlease show your reasoning between <reasoning></reasoning> and "
        "put the final short answer in <answer></answer>."
        "\n<end_of_turn>\n<start_of_turn>model\n"
    ),
}


def _load_jsonl(path: str):
  with epath.Path(path).open("r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      yield json.loads(line)


# =========================
# Public builder functions
# =========================

def create_vqa_loaders(
    *,
    source: str,
    global_batch_size: int,
    image_size: int,
    num_train_epochs: int | None,
    instruct_tuned: bool = True,
    input_template: dict[str, str] | None = None,
    # JSONL-specific:
    jsonl_path: Optional[str] = None,
    image_root: Optional[str] = None,
    # TFDS-specific:
    tfds_name: Optional[str] = None,
    tfds_split_train: str = "train",
    tfds_split_eval: str = "validation",
) -> tuple[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]:
  """Creates train/eval Grain loaders for VLM GRPO.

  Args:
    source: one of {"jsonl", "tfds"}.
    global_batch_size: per-step batch size.
    image_size: resize shorter side (square) to this (e.g., 384 for SigLIP-384).
    num_train_epochs: train epochs; None => repeat indefinitely.
    instruct_tuned: whether to use IT template.
    input_template: override the default template dict.
    jsonl_path, image_root: used when source="jsonl".
      JSONL schema must contain: {"question": str, "image": str, "answer": str}.
      "image" is a relative path under image_root.
    tfds_name: e.g. "coco_captions", "textcaps", "vizwiz" (depends on your TFDS).
    tfds_split_train / tfds_split_eval: splits for TFDS.

  Returns:
    train_loader, eval_loader: each yields dict with:
      {
        "prompts": str,
        "image": np.ndarray [H,W,3] uint8,
        "answer": str,
        "question": str,
      }
  """
  if input_template is None:
    input_template = INPUT_TEMPLATE_VLM_IT if instruct_tuned else INPUT_TEMPLATE_VLM

  if source == "jsonl":
    if not jsonl_path or not image_root:
      raise ValueError("jsonl_path and image_root are required when source='jsonl'")
    train_ds = _JsonlVQASource(jsonl_path, image_root)
    eval_ds = _JsonlVQASource(jsonl_path, image_root)  # simple: same file for eval unless you have a separate one
  elif source == "tfds":
    if not tfds_name:
      raise ValueError("tfds_name is required when source='tfds'")
    train_ds = tfds.data_source(tfds_name, split=tfds_split_train)
    eval_ds = tfds.data_source(tfds_name, split=tfds_split_eval)
  else:
    raise ValueError(f"Unsupported source: {source}")

  train_loader = _build_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      image_size=image_size,
      input_template=input_template,
      source=source,
  )
  eval_loader = _build_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      image_size=image_size,
      input_template=input_template,
      source=source,
  )
  return train_loader, eval_loader


# =========================
# Grain pipeline
# =========================

def _build_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    image_size: int,
    input_template: dict[str, str],
    source: str,
) -> grain.DataLoader:
  ops = []
  if source == "jsonl":
    ops.extend([
        _JsonlLoadAndTemplate(input_template),
        _LoadImageFromPath(image_size),
    ])
  else:  # TFDS path (expects an image array in the TFDS example)
    ops.extend([
        _TFDSMapToVLMTemplate(input_template),
        _EnsureUint8Image(image_size),
    ])
  ops.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=ops,
  )


# =========================
# JSONL pipeline pieces
# =========================

class _JsonlVQASource(grain.RandomAccessDataSource):
  """RandomAccessDataSource wrapping a JSONL file with VQA triplets.

  JSONL entries must have keys: {"question": str, "image": str, "answer": str}
  """

  def __init__(self, jsonl_path: str, image_root: str):
    self._records = list(_load_jsonl(jsonl_path))
    self._image_root = image_root

  def __len__(self) -> int:
    return len(self._records)

  def __getitem__(self, idx: int) -> dict[str, Any]:
    ex = self._records[idx]
    return {
        "question": ex["question"],
        "image_path": os.path.join(self._image_root, ex["image"]),
        "answer": ex.get("answer", ""),
    }


class _JsonlLoadAndTemplate(grain.MapTransform):
  """Builds {'prompts','image_path','answer','question'} using the template."""

  def __init__(self, input_template: dict[str, str]):
    self._tpl = input_template

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    q = element["question"]
    prompt = f"{self._tpl['prefix']}{q}{self._tpl['suffix']}"
    return {
        "prompts": prompt,
        "image_path": element["image_path"],
        "answer": element["answer"],
        "question": q,
    }


class _LoadImageFromPath(grain.MapTransform):
  """Loads image_path -> image np.uint8 [H,W,3], resized square to `image_size`."""

  def __init__(self, image_size: int):
    if not _HAS_PIL:
      raise RuntimeError("Pillow is required for _LoadImageFromPath. Please install pillow.")
    self._size = image_size

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    p = element["image_path"]
    with Image.open(p) as im:
      im = im.convert("RGB")
      im = im.resize((self._size, self._size))
      img = np.asarray(im, dtype=np.uint8)
    out = dict(element)
    out.pop("image_path", None)
    out["image"] = img  # uint8 [H,W,3]
    return out


# =========================
# TFDS pipeline pieces
# =========================

class _TFDSMapToVLMTemplate(grain.MapTransform):
  """Converts TFDS examples to our VLM dict.

  Supported fields:
    - ex['image']: uint8 image
    - ex['captions']['text'] (COCO captions) OR
    - ex['question'] / ex['answer'] (TextCaps/VizWiz style)
  """

  def __init__(self, input_template: dict[str, str]):
    self._tpl = input_template

  def map(self, ex: dict[str, Any]) -> dict[str, Any]:
    # Try to infer a question/answer pair
    if "question" in ex and "answer" in ex:
      q = ex["question"].numpy().decode("utf-8") if hasattr(ex["question"], "numpy") else str(ex["question"])
      a = ex["answer"].numpy().decode("utf-8") if hasattr(ex["answer"], "numpy") else str(ex["answer"])
    elif "captions" in ex and "text" in ex["captions"]:
      # COCO: create a generic question and use first caption as "answer"
      q = "Describe this image in a short sentence."
      cap_list = ex["captions"]["text"]
      if hasattr(cap_list, "numpy"):
        # Eager tensor array -> take first
        a = cap_list[0].numpy().decode("utf-8") if len(cap_list) > 0 else ""
      else:
        a = cap_list[0] if len(cap_list) > 0 else ""
    else:
      # Fallback: generic Q/A
      q = "What is in the image?"
      a = ""

    prompt = f"{self._tpl['prefix']}{q}{self._tpl['suffix']}"
    img = ex["image"]
    # tfds returns a TF tensor or numpy – grain will happily pass numpy
    img = np.array(img)  # ensure numpy

    return {
        "prompts": prompt,
        "image": img,     # expected uint8 [H,W,3]
        "answer": a,
        "question": q,
    }


class _EnsureUint8Image(grain.MapTransform):
  """Ensures image is uint8 and resized to (image_size, image_size)."""

  def __init__(self, image_size: int):
    self._size = image_size

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    img = element["image"]
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
      img = np.clip(img, 0, 255).astype(np.uint8)

    # Resize to square via PIL if available; else keep original (SigLIP can center-crop inside model if you prefer).
    if _HAS_PIL:
      im = Image.fromarray(img, mode="RGB")
      im = im.resize((self._size, self._size))
      img = np.asarray(im, dtype=np.uint8)

    out = dict(element)
    out["image"] = img
    return out
