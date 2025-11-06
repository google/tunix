"""Utilities for tunix scripts."""

from collections.abc import Callable
import json
import logging
import os
import grain

try:
  # This is a g3-only import.
  from GOOGLE_INTERNAL_PACKAGE_PATH.perftools.accelerators.xprof.api.python import xprof_session  # pytype: disable=import-error
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # pytype: disable=import-error

  ENV = 'g3'
except ImportError:
  xprof_session = None
  gfile = None
  ENV = 'oss'

if ENV == 'oss':
  import tensorflow_datasets as tfds
  import fsspec


def get_dataset(
    path: str,
    split: str,
    seed: int,
    system_prompt: str,
    *,
    answer_extractor: Callable[[str], str | None],
    dataset_name: str = 'gsm8k',
) -> grain.MapDataset:
  """Loads the dataset, from CNS in g3 or downloading in OSS."""
  if ENV == 'g3':
    with gfile.Open(path, 'rb') as f:
      data = json.loads(f.read())
  else:  # oss
    if path.startswith('gs://'):
      with fsspec.open(path, 'r') as f:
        data = json.load(f)
    else:
      print(
          f"Downloading {dataset_name.upper()} dataset ('{split}' split) to"
          f" '{path}'..."
      )
      if not os.path.exists(path):
        os.makedirs(path)
      # Using TFDS to download dataset.
      data = tfds.data_source(
          dataset_name,
          split=split,
          data_dir=path,
          builder_kwargs={'file_format': tfds.core.FileFormat.ARRAY_RECORD},
          download=True,
      )

  def _as_text(v):
    return v if isinstance(v, str) else v.decode('utf-8')

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=seed)
      .map(
          lambda x: {
              # passed to model forward pass
              'prompts': system_prompt + _as_text(x['question']),
              # passed to reward functions
              'question': system_prompt + _as_text(x['question']),
              # passed to reward functions
              'answer': answer_extractor(_as_text(x['answer'])),
          }
      )
  )
  return dataset


def get_train_and_eval_datasets(
    data_path: str,
    split: str,
    seed: int,
    system_prompt: str,
    batch_size: int,
    num_batches: int | None,
    train_fraction: float,
    num_epochs: int | None,
    *,
    answer_extractor: Callable[[str], str | None],
    dataset_name: str = 'gsm8k',
) -> tuple[grain.MapDataset, grain.MapDataset | None]:
  """Loads and splits the dataset for training and evaluation."""
  dataset = get_dataset(
      data_path,
      split=split,
      seed=seed,
      system_prompt=system_prompt,
      answer_extractor=answer_extractor,
      dataset_name=dataset_name,
  ).batch(batch_size)

  if num_batches:
    dataset = dataset[:num_batches]

  if train_fraction == 1.0:
    train_dataset = dataset
    if num_epochs:
      train_dataset = train_dataset.repeat(num_epochs)
    val_dataset = None
  else:
    train_size = int(len(dataset) * train_fraction)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    if num_epochs:
      train_dataset = train_dataset.repeat(num_epochs)
      val_dataset = val_dataset.repeat(num_epochs)

  return train_dataset, val_dataset


class profile_and_capture_log:
  """A context manager for profiling and capturing logs.

  This is a no-op in OSS.

  Args:
    tag: A tag for the xprof session.
    enable_profile: Whether to enable profiling.
    device_name: The device name for xprof.
    host_trace_level: The host trace level for xprof.
  """

  def __init__(
      self,
      tag: str = '',
      enable_profile: bool = True,
      device_name: str = 'viperfish',
      host_trace_level: int = 2,
  ):
    self._tag = tag
    self._enable_profile = enable_profile
    self._device_name = device_name
    self._host_trace_level = host_trace_level
    self._xprof = None
    self._log_handler = None

  def __enter__(self):
    if ENV == 'g3' and xprof_session is not None:
      if self._enable_profile:
        self._xprof = xprof_session.XprofSession()
        self._xprof.start_session(
            device_name=self._device_name,
            enable_python_tracer=True,
            host_trace_level=self._host_trace_level,
        )
      self._log_handler = logging.StreamHandler()
      logging.root.addHandler(self._log_handler)

  def __exit__(self, exc_type, exc_value, traceback):
    if self._log_handler:
      logging.root.removeHandler(self._log_handler)
    if self._xprof:
      xprof_url = self._xprof.end_session_and_get_url(
          tag=self._tag, ttl_seconds=60 * 60 * 24 * 365
      )
      print(xprof_url)
