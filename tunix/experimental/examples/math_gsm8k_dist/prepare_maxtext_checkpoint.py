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

"""Prepares a MaxText checkpoint for the distributed GSM8K demo.

The demo workers load MaxText weights through `load_parameters_path`, which
points at the converted Orbax checkpoint directory, typically `<base>/0/items`.
When that checkpoint is missing, this helper invokes MaxText's HF-to-MaxText
converter before any TPU worker process initializes JAX.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _resolve_paths(model_dir: str) -> tuple[str, str]:
  """Returns `(load_parameters_path, base_output_directory)`."""
  normalized_model_dir = model_dir.rstrip("/")
  if normalized_model_dir.endswith("/0/items"):
    first_parent = os.path.dirname(normalized_model_dir)
    return normalized_model_dir, os.path.dirname(first_parent)
  return f"{normalized_model_dir}/0/items", normalized_model_dir


def _checkpoint_exists(path: str) -> bool:
  if path.startswith("gs://"):
    for tool in (("gcloud", "storage", "ls"), ("gsutil", "-q", "ls")):
      if not shutil.which(tool[0]):
        continue
      return (
          subprocess.run(
              [*tool, f"{path.rstrip('/')}/"],
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
              check=False,
          ).returncode
          == 0
      )
    return False

  ckpt_path = Path(path).expanduser()
  if not ckpt_path.is_dir():
    return False
  marker_names = (
      "_CHECKPOINT_METADATA",
      "_METADATA",
      "_sharding",
      "array_metadatas",
      "manifest.ocdbt",
  )
  if any((ckpt_path / marker_name).exists() for marker_name in marker_names):
    return True
  return any(ckpt_path.rglob(".zarray")) or any(ckpt_path.rglob("zarr.json"))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Prepare a MaxText checkpoint for the distributed GSM8K demo."
  )
  parser.add_argument(
      "--model_name", default=os.getenv("MODEL_NAME", "Qwen3-1.7B")
  )
  parser.add_argument("--model_id", default=os.getenv("MODEL_ID", ""))
  parser.add_argument("--model_dir", default=os.getenv("MODEL_DIR", ""))
  parser.add_argument(
      "--maxtext_repo_root", default=os.getenv("MAXTEXT_REPO_ROOT")
  )
  parser.add_argument("--resolved_model_dir_file")
  args = parser.parse_args(sys.argv[1:] if argv is None else argv)
  if not args.model_dir:
    raise ValueError("--model_dir is required for MODEL_SOURCE=maxtext.")

  load_path, base_output_dir = _resolve_paths(args.model_dir)
  print(
      "MaxText checkpoint preparation: "
      f"model_name={args.model_name} load_path={load_path} "
      f"base_output_directory={base_output_dir}",
      file=sys.stderr,
  )

  if _checkpoint_exists(load_path):
    print(f"Found existing MaxText checkpoint: {load_path}", file=sys.stderr)
    if args.resolved_model_dir_file:
      Path(args.resolved_model_dir_file).write_text(load_path, encoding="utf-8")
    return 0

  model_name = args.model_name.strip().lower().replace("_", "-")
  model_name = {
      "qwen3-0p6b": "qwen3-0.6b",
      "qwen3-1p7b": "qwen3-1.7b",
  }.get(model_name, model_name)
  save_dtype = os.getenv("MAXTEXT_DTYPE", "bfloat16")
  if not args.model_id:
    raise ValueError(
        "--model_id is required to convert a missing MaxText checkpoint."
    )

  config_path = None
  if args.maxtext_repo_root:
    candidate = (
        Path(args.maxtext_repo_root).expanduser()
        / "src"
        / "maxtext"
        / "configs"
        / "base.yml"
    )
    if candidate.exists():
      config_path = str(candidate)
  if config_path is None:
    spec = importlib.util.find_spec("maxtext")
    if spec is None or spec.origin is None:
      raise RuntimeError(
          "MaxText is not importable. Install MaxText in this environment or "
          "set MAXTEXT_REPO_ROOT=/path/to/maxtext before launching "
          "MODEL_SOURCE=maxtext."
      )
    config_path = str(Path(spec.origin).parent / "configs" / "base.yml")
    if not Path(config_path).exists():
      raise RuntimeError(f"Could not find MaxText base.yml at {config_path}.")

  env = os.environ.copy()
  env["JAX_PLATFORMS"] = "cpu"
  if args.maxtext_repo_root:
    maxtext_pythonpath = str(Path(args.maxtext_repo_root).expanduser() / "src")
    env["PYTHONPATH"] = (
        maxtext_pythonpath
        if not env.get("PYTHONPATH")
        else f"{maxtext_pythonpath}:{env['PYTHONPATH']}"
    )

  cmd = [
      sys.executable,
      "-m",
      "maxtext.checkpoint_conversion.to_maxtext",
      config_path,
      f"model_name={model_name}",
      f"base_output_directory={base_output_dir}",
      "use_multimodal=false",
      "scan_layers=false",
      "hardware=cpu",
      "skip_jax_distributed_system=true",
      "checkpoint_storage_use_ocdbt=true",
      "checkpoint_storage_use_zarr3=true",
      "--lazy_load_tensors=true",
      "--eager_load_method=safetensors",
      f"--save_dtype={save_dtype}",
      "--simulated_cpu_devices_count=16",
  ]
  cmd.append(f"--hf_model_path={args.model_id}")

  print(
      "No MaxText checkpoint found; converting HuggingFace checkpoint with:\n  "
      + " ".join(cmd),
      file=sys.stderr,
  )
  subprocess.run(cmd, env=env, check=True)

  if not _checkpoint_exists(load_path):
    raise RuntimeError(
        "MaxText conversion completed, but the expected checkpoint was not "
        f"found at {load_path}."
    )

  if args.resolved_model_dir_file:
    Path(args.resolved_model_dir_file).write_text(load_path, encoding="utf-8")
  print(f"Prepared MaxText checkpoint: {load_path}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
