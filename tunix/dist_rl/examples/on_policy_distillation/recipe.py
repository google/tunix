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

"""Pinned Qwen3-8B from Qwen3-32B OPD training and held-out evaluation recipe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import urllib.parse

from tunix.dist_rl.examples.on_policy_distillation import models

DEFAULT_CONFIG = (
    Path(__file__).with_name("configs") / "qwen3_8b_from_qwen3_32b.json"
)
_TRAIN_ENV = (
    "tfds_dataset tfds_split seed shuffle max_steps batch_size mini_batch_size"
    " num_generations train_micro_batch_size max_prompt_length"
    " max_response_length temperature top_p top_k opd_coefficient"
    " opd_task_reward_coefficient opd_teacher_timeout_seconds"
    " opd_max_concurrency beta epsilon learning_rate weight_decay adam_b1"
    " adam_b2 adam_eps max_grad_norm checkpoint_save_interval_steps"
    " checkpoint_max_to_keep"
).split()
_HARDWARE_ENV = (
    "trainer_tpu_chips rollout_tpu_chips trainer_fsdp trainer_tp rollout_fsdp"
    " rollout_tp tpu_chips_per_host_bounds tpu_host_bounds model_dtype"
    " model_load_dtype remat_config"
).split()


def _positive_integer(value: Any, name: str) -> None:
  if type(value) is not int or value <= 0:
    raise ValueError(f"{name} must be a positive integer.")


def _finite_number(value: Any, name: str, *, positive: bool = False) -> None:
  if (
      isinstance(value, bool)
      or not isinstance(value, (float, int))
      or not math.isfinite(value)
      or value < 0
      or (positive and value == 0)
  ):
    raise ValueError(
        f"{name} must be finite and"
        f" {'positive' if positive else 'nonnegative'}."
    )


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
  """Loads a complete profile and rejects mutable identities or unsafe settings."""
  config = json.loads(Path(path).read_text())
  expected_keys = {
      "schema_version",
      "name",
      "student",
      "teacher",
      "training",
      "evaluation",
      "hardware",
      "teacher_service",
  }
  if not isinstance(config, dict) or set(config) != expected_keys:
    raise ValueError("Recipe config has missing or unknown top-level fields.")
  if (
      type(config["schema_version"]) is not int
      or config["schema_version"] != 1
      or not isinstance(config["name"], str)
  ):
    raise ValueError("Unsupported recipe schema or invalid name.")
  for role, repo_id, revision in (
      ("student", models.STUDENT_MODEL_ID, models.STUDENT_REVISION),
      ("teacher", models.TEACHER_MODEL_ID, models.TEACHER_REVISION),
  ):
    if config[role] != {"repo_id": repo_id, "revision": revision}:
      raise ValueError(f"The full recipe fixes the {role} model and revision.")
  train = config["training"]
  if not isinstance(train, dict) or set(train) != set(_TRAIN_ENV):
    raise ValueError("Training config has missing or unknown fields.")
  if train["tfds_dataset"] != "gsm8k:1.0.0" or train["tfds_split"] != "train":
    raise ValueError("Training must use the pinned GSM8K training split.")
  if (
      type(train["seed"]) is not int
      or train["seed"] < 0
      or type(train["shuffle"]) is not bool
  ):
    raise ValueError("Training seed/shuffle must be an integer and boolean.")
  for name in (
      "max_steps",
      "batch_size",
      "mini_batch_size",
      "num_generations",
      "train_micro_batch_size",
      "max_prompt_length",
      "max_response_length",
      "checkpoint_save_interval_steps",
      "checkpoint_max_to_keep",
      "opd_max_concurrency",
  ):
    _positive_integer(train[name], name)
  if train["num_generations"] < 2:
    raise ValueError(
        "The GRPO adapter requires at least two responses per prompt."
    )
  if train["mini_batch_size"] != train["batch_size"]:
    raise ValueError("This recipe uses one optimizer update per rollout batch.")
  trajectories = train["batch_size"] * train["num_generations"]
  microbatch = train["train_micro_batch_size"]
  if trajectories % microbatch or trajectories // microbatch < 2:
    raise ValueError(
        "Use at least two integral gradient-accumulation microbatches."
    )
  if train["max_steps"] * train["batch_size"] > 7473:
    raise ValueError(
        "This recipe does not cycle or silently repeat training prompts."
    )
  if train["max_steps"] % train["checkpoint_save_interval_steps"]:
    raise ValueError("The final step must coincide with a saved checkpoint.")
  for name in ("temperature", "top_p", "beta"):
    _finite_number(train[name], name)
  if (
      train["temperature"] != 1
      or train["top_p"] != 1
      or type(train["top_k"]) is not int
      or train["top_k"] not in (-1, 0)
  ):
    raise ValueError(
        "Sampled-token OPD requires unfiltered temperature-one sampling."
    )
  if train["beta"] != 0:
    raise ValueError("The full recipe uses beta=0 and no reference worker.")
  for name in (
      "opd_coefficient",
      "opd_task_reward_coefficient",
      "opd_teacher_timeout_seconds",
      "learning_rate",
      "weight_decay",
      "epsilon",
      "adam_b1",
      "adam_b2",
      "adam_eps",
      "max_grad_norm",
  ):
    _finite_number(
        train[name],
        name,
        positive=name
        in {
            "opd_coefficient",
            "opd_teacher_timeout_seconds",
            "learning_rate",
            "adam_eps",
            "max_grad_norm",
        },
    )
  if not 0 <= train["adam_b1"] < 1 or not 0 <= train["adam_b2"] < 1:
    raise ValueError("Adam moment coefficients must lie in [0, 1).")
  hardware = config["hardware"]
  if not isinstance(hardware, dict) or set(hardware) != set(_HARDWARE_ENV):
    raise ValueError("Hardware config has missing or unknown fields.")
  if (
      hardware["model_dtype"] != "bfloat16"
      or hardware["model_load_dtype"] != "bfloat16"
  ):
    raise ValueError("The full recipe uses bfloat16 model/load dtypes.")
  if hardware["remat_config"] != "DECODER":
    raise ValueError("The full recipe requires decoder rematerialization.")
  for role in ("trainer", "rollout"):
    chips = str(hardware[f"{role}_tpu_chips"]).split(",")
    if (
        not chips
        or any(not c.isdecimal() for c in chips)
        or any(str(int(c)) != c for c in chips)
        or len(set(chips)) != len(chips)
    ):
      raise ValueError(f"Invalid {role} TPU chip list.")
    for axis in ("fsdp", "tp"):
      _positive_integer(hardware[f"{role}_{axis}"], f"{role}_{axis}")
    if hardware[f"{role}_fsdp"] * hardware[f"{role}_tp"] != len(chips):
      raise ValueError(f"The {role} mesh must cover its chip list.")
  if set(str(hardware["trainer_tpu_chips"]).split(",")) & set(
      str(hardware["rollout_tpu_chips"]).split(",")
  ):
    raise ValueError("Trainer and rollout chip sets must be disjoint.")
  for name in ("tpu_chips_per_host_bounds", "tpu_host_bounds"):
    dimensions = str(hardware[name]).split(",")
    if len(dimensions) != 3 or any(
        not item.isdecimal() or int(item) < 1 for item in dimensions
    ):
      raise ValueError(
          f"{name} must contain three positive integer dimensions."
      )
  if hardware["tpu_host_bounds"] != "1,1,1":
    raise ValueError("This launcher profile supports one TPU host.")
  local_chips = math.prod(
      int(item) for item in hardware["tpu_chips_per_host_bounds"].split(",")
  )
  if any(
      len(hardware[f"{role}_tpu_chips"].split(",")) != local_chips
      for role in ("trainer", "rollout")
  ):
    raise ValueError("TPU bounds must match each worker process's chip count.")
  from tunix.dist_rl.examples.on_policy_distillation.evaluation import EvaluationConfig  # pylint: disable=g-import-not-at-top

  eval_config = EvaluationConfig.from_dict(config["evaluation"])
  if eval_config.split != "test" or eval_config.max_examples != 0:
    raise ValueError(
        "The full recipe evaluates the entire official held-out test split."
    )
  if eval_config.max_response_length > train["max_response_length"]:
    raise ValueError(
        "Evaluation must fit the live rollout worker response budget."
    )
  teacher = config["teacher_service"]
  if not isinstance(teacher, dict) or set(teacher) != {
      "host",
      "port",
      "tensor_parallel_size",
      "mem_fraction_static",
      "context_length",
  }:
    raise ValueError("Teacher service config has missing or unknown fields.")
  for name in ("port", "tensor_parallel_size", "context_length"):
    _positive_integer(teacher[name], f"teacher {name}")
  if (
      teacher["port"] > 65535
      or not isinstance(teacher["host"], str)
      or not teacher["host"]
  ):
    raise ValueError("Invalid teacher host/port.")
  _finite_number(
      teacher["mem_fraction_static"], "teacher memory fraction", positive=True
  )
  if teacher["mem_fraction_static"] >= 1:
    raise ValueError("Teacher memory fraction must be below one.")
  if (
      teacher["context_length"]
      < train["max_prompt_length"] + train["max_response_length"]
  ):
    raise ValueError(
        "Teacher context must cover the complete training trajectory."
    )
  return config


def config_digest(config: dict[str, Any]) -> str:
  return hashlib.sha256(
      json.dumps(config, sort_keys=True, allow_nan=False).encode()
  ).hexdigest()


def _value(value: Any) -> str:
  return str(value).lower() if isinstance(value, bool) else str(value)


def build_environment(
    config: dict[str, Any],
    prepared: dict[str, Any],
    *,
    models_manifest: Path,
    run_dir: Path,
    data_dir: Path,
    teacher_url: str,
    eval_only: bool = False,
    require_resume: bool = False,
) -> dict[str, str]:
  """Resolves explicit launcher overrides without inheriting model/smoke defaults."""
  train = config["training"]
  environment = {name.upper(): _value(train[name]) for name in _TRAIN_ENV}
  environment.update({
      name.upper(): _value(value) for name, value in config["hardware"].items()
  })
  environment.update({
      "MODEL_NAME": models.STUDENT_MODEL_NAME,
      "GENERATION_SEED": str(train["seed"]),
      "MODEL_ID": models.STUDENT_MODEL_ID,
      "MODEL_REVISION": models.STUDENT_REVISION,
      "MODEL_DIR": prepared["student"]["snapshot_path"],
      "TOKENIZER_PATH": prepared["student"]["tokenizer_path"],
      "OPD_TEACHER_URL": teacher_url,
      "OPD_TEACHER_TOKENIZER_PATH": prepared["teacher"]["tokenizer_path"],
      "OPD_TEACHER_REVISION": prepared["teacher"]["declared_revision"],
      "OPD_MODELS_MANIFEST": str(models_manifest),
      "OPD_EVAL_CONFIG": str(run_dir / "evaluation.json"),
      "OPD_EVAL_OUTPUT_DIR": str(run_dir / "evaluation"),
      "OPD_EVAL_ONLY": _value(eval_only),
      "OPD_REQUIRE_RESUME": _value(require_resume or eval_only),
      "ALLOW_MODEL_DOWNLOAD": "false",
      "TRAINER_BACKEND": "tunix",
      "SAMPLER": "inprocess_vllm",
      "WEIGHT_SYNC_MODE": "raiden",
      "MAX_STALENESS": "0",
      "IN_FLIGHT_WEIGHT_UPDATES": "false",
      "USE_ROLLOUT_LOGPS": "true",
      "USE_LORA": "false",
      "RUN_INFERENCE_NODE": "0",
      "INFERENCE_ADDR": "",
      "CHAT_PARSER": "raw",
      "EOS_TOKENS": "151645,151643",
      "SCHEDULE_TYPE": "",
      "OPT_CHAIN_TYPE": "clip_by_global_norm",
      "MAX_SEQ_TOKEN_PER_TPU": "",
      "MAX_SEGMENTS_PER_PACKED_ROW": "",
      "SEGMENT_ALIGNMENT_BOUNDARY": "",
      "MAXTEXT_MODEL_NAME": "",
      "MAXTEXT_CKPT": "",
      "MAXTEXT_ATTENTION": "",
      "TRAINER_PATHWAYS": "0",
      "ARTIFACT_ROOT": str(run_dir / "artifacts"),
      "LOG_ROOT": str(run_dir / "logs"),
      "LOG_DIR": str(run_dir / "metrics"),
      "TRAJECTORY_LOG_DIR": str(run_dir / "trajectories"),
      "CHECKPOINT_ROOT_DIRECTORY": str(run_dir / "checkpoints"),
      "TFDS_DATA_DIR": str(data_dir),
      "RUN_ID": "opd-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:16],
      "WANDB_PROJECT": "tunix-opd-gsm8k",
      "WAIT_TIMEOUT_SECS": "1800",
      "EVAL_EVERY_N_STEPS": str(train["max_steps"] + 1),
  })
  return environment


def _atomic_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
  )
  temporary.replace(path)


def _launch(args: argparse.Namespace, config: dict[str, Any]) -> int:
  run_dir = Path(args.run_dir).expanduser().resolve()
  if run_dir.exists() and not run_dir.is_dir():
    raise ValueError("Run path must be a directory.")
  models_path = Path(args.models_manifest).expanduser().resolve()
  prepared = models.load_manifest(models_path)
  existing = run_dir / "run_manifest.json"
  recorded = json.loads(existing.read_text()) if existing.is_file() else None
  if args.command == "eval" and not args.teacher_url and recorded is not None:
    args.teacher_url = recorded["environment"]["OPD_TEACHER_URL"]
  if not args.teacher_url:
    raise ValueError(
        "Training requires a teacher URL; evaluation needs the recorded run"
        " URL."
    )
  parsed = urllib.parse.urlsplit(args.teacher_url)
  if (
      parsed.scheme not in ("http", "https")
      or not parsed.netloc
      or parsed.username
      or parsed.password
      or parsed.query
      or parsed.fragment
  ):
    raise ValueError(
        "Use a teacher HTTP(S) base URL without credentials, query or fragment."
    )
  data_dir = (
      Path(args.data_dir or run_dir.parent / "gsm8k_data")
      .expanduser()
      .resolve()
  )
  environment = build_environment(
      config,
      prepared,
      models_manifest=models_path,
      run_dir=run_dir,
      data_dir=data_dir,
      teacher_url=args.teacher_url,
      eval_only=args.command == "eval",
      require_resume=args.resume,
  )
  launcher = (
      Path(__file__).resolve().parents[1] / "math_gsm8k_dist" / "launcher.sh"
  )
  plan = {
      "config_sha256": config_digest(config),
      "profile": config,
      "models": prepared,
      "launcher": ["bash", str(launcher)],
      "environment": environment,
  }
  if args.dry_run:
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0
  continuing = args.command == "eval" or args.resume
  if continuing:
    if not existing.is_file():
      raise ValueError(
          "Resume/evaluation requires an existing run_manifest.json."
      )
    old = recorded
    if (
        old["config_sha256"] != plan["config_sha256"]
        or old["models"] != prepared
    ):
      raise ValueError(
          "Run configuration/model manifest changed; use a new run directory."
      )
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir() or not any(
        path.is_dir() and path.name.isdecimal() and int(path.name) > 0
        for path in checkpoint_dir.iterdir()
    ):
      raise ValueError(
          "Resume/evaluation requires an existing numeric checkpoint step."
      )
  elif run_dir.exists() and any(run_dir.iterdir()):
    raise ValueError(
        "A fresh run requires an empty directory; use --resume explicitly."
    )
  remote = None
  if args.command != "eval":
    from tunix.dist_rl.examples.on_policy_distillation import serve_teacher  # pylint: disable=g-import-not-at-top

    remote = asyncio.run(
        serve_teacher.verify_teacher_service(
            args.teacher_url,
            prepared,
            timeout_seconds=30.0,
            min_context_length=config["training"]["max_prompt_length"]
            + config["training"]["max_response_length"],
        )
    )
  run_dir.mkdir(parents=True, exist_ok=True)
  if not existing.exists():
    versions = {}
    for package in ("google-tunix", "jax", "flax", "transformers", "vllm"):
      try:
        versions[package] = importlib.metadata.version(package)
      except importlib.metadata.PackageNotFoundError:
        versions[package] = None
    _atomic_json(existing, {**plan, "package_versions": versions})
  _atomic_json(run_dir / "evaluation.json", config["evaluation"])
  if remote is not None:
    _atomic_json(run_dir / "teacher_preflight.json", remote)
  process_environment = os.environ.copy()
  process_environment.update(environment)
  process_environment["PYTHON_BIN"] = sys.executable
  return subprocess.run(
      plan["launcher"], env=process_environment, check=False
  ).returncode


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", default=str(DEFAULT_CONFIG))
  commands = parser.add_subparsers(dest="command", required=True)
  show = commands.add_parser(
      "show",
      help="Validate and print the frozen recipe without downloads or workers.",
  )
  show.add_argument("--config", default=argparse.SUPPRESS)
  teacher = commands.add_parser(
      "teacher",
      help="Launch the pinned teacher using the profile's resource settings.",
  )
  teacher.add_argument("--config", default=argparse.SUPPRESS)
  teacher.add_argument("--models-manifest", required=True)
  teacher.add_argument("--cuda-visible-devices")
  teacher.add_argument("--dry-run", action="store_true")
  for name in ("train", "eval"):
    command = commands.add_parser(name)
    command.add_argument("--config", default=argparse.SUPPRESS)
    command.add_argument("--models-manifest", required=True)
    command.add_argument("--run-dir", required=True)
    command.add_argument("--teacher-url", required=name == "train")
    command.add_argument("--data-dir")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--resume", action="store_true")
  args = parser.parse_args(argv)
  config = load_config(args.config)
  if args.command == "show":
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0
  if args.command == "teacher":
    arguments = [
        "--models-manifest",
        str(Path(args.models_manifest).expanduser().resolve()),
    ]
    for name, value in config["teacher_service"].items():
      arguments.extend(["--" + name.replace("_", "-"), str(value)])
    if args.cuda_visible_devices is not None:
      arguments.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.dry_run:
      print(
          json.dumps(
              [
                  sys.executable,
                  "-m",
                  "tunix.dist_rl.examples.on_policy_distillation.serve_teacher",
                  *arguments,
              ],
              indent=2,
          )
      )
      return 0
    from tunix.dist_rl.examples.on_policy_distillation import serve_teacher  # pylint: disable=g-import-not-at-top

    serve_teacher.main(arguments)
    return 0
  return _launch(args, config)


if __name__ == "__main__":
  raise SystemExit(main())
