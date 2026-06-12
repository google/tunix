"""Builds and optionally runs ablation sweeps for the GSM8K VTC demo.

The presets live in ``qwen3_grpo_gsm8k_vtc_demo.py`` via ``--ablation_preset``.
This runner groups them into a small screening stage plus a few drill-down
stages so we can identify which bundle most strongly affects convergence.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEMO_SCRIPT = os.path.join(SCRIPT_DIR, "qwen3_grpo_gsm8k_vtc_demo.py")
ABLATION_ARTIFACT_DIR = os.path.join(
    REPO_ROOT, "artifacts", "qwen3_grpo_gsm8k_vtc", "ablations"
)
WANDB_URL_PATTERN = re.compile(r"https?://[^\s]*wandb\.ai[^\s]*")

STAGES = {
    "screening": [
        "final",
        "oldish_full",
        "revert_prompt_reward",
        "revert_rollout_runtime",
        "revert_model_bundle",
        "revert_old_logps_to_rollout",
        "revert_kl",
    ],
    "prompt_drilldown": [
        "final",
        "revert_prompt_reward",
        "revert_parser_only",
        "revert_reward_only",
    ],
    "runtime_drilldown": [
        "final",
        "revert_rollout_runtime",
        "revert_runtime_async_only",
        "revert_runtime_prefix_only",
        "revert_runtime_concurrency_only",
        "revert_runtime_inflight_only",
    ],
    "recompute_drilldown": [
        "final",
        "revert_old_logps_to_rollout",
    ],
    "model_drilldown": [
        "final",
        "revert_model_bundle",
        "revert_actor_dtype_only",
        "revert_flash_only",
    ],
    "rootcause_drilldown": [
        "revert_parser_only",
        "revert_reward_only",
        "revert_actor_dtype_only",
        "revert_flash_only",
    ],
    "dtype_drilldown": [
        "final",
        "revert_actor_dtype_only",
    ],
    "all": [
        "final",
        "oldish_full",
        "revert_prompt_reward",
        "revert_parser_only",
        "revert_reward_only",
        "revert_rollout_runtime",
        "revert_runtime_async_only",
        "revert_runtime_prefix_only",
        "revert_runtime_concurrency_only",
        "revert_runtime_inflight_only",
        "revert_model_bundle",
        "revert_actor_dtype_only",
        "revert_flash_only",
        "revert_old_logps_to_rollout",
        "revert_kl",
    ],
}


def build_command(
    *,
    preset: str,
    max_steps: int,
    extra_args: list[str],
    python_bin: str,
    stage: str,
) -> list[str]:
  tag = f"{stage}_{preset}"
  return [
      python_bin,
      DEMO_SCRIPT,
      "--ablation_preset",
      preset,
      "--max_steps",
      str(max_steps),
      "--experiment_tag",
      tag,
      *extra_args,
  ]


def _append_wandb_url(
    *,
    record_file: str,
    preset: str,
    experiment_tag: str,
    url: str,
) -> None:
  os.makedirs(os.path.dirname(record_file), exist_ok=True)
  with open(record_file, "a", encoding="utf-8") as f:
    f.write(f"{experiment_tag}\t{preset}\t{url}\n")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run named ablation stages for the GSM8K VTC demo."
  )
  parser.add_argument(
      "--stage",
      type=str,
      default="screening",
      choices=sorted(STAGES),
      help="Experiment bundle to generate/run.",
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      default=200,
      help="Override max_steps for every run in the sweep.",
  )
  parser.add_argument(
      "--python_bin",
      type=str,
      default=sys.executable,
      help="Python interpreter used to launch the demo.",
  )
  parser.add_argument(
      "--execute",
      action="store_true",
      help="Actually run the commands sequentially. Default is dry-run.",
  )
  parser.add_argument(
      "--extra_args",
      type=str,
      default="",
      help="Extra raw args appended to every demo invocation.",
  )
  parser.add_argument(
      "--wandb_urls_file",
      type=str,
      default=os.path.join(ABLATION_ARTIFACT_DIR, "wandb_urls.tsv"),
      help="TSV file where discovered W&B run URLs are appended.",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.path.join(ABLATION_ARTIFACT_DIR, "logs"),
      help="Directory for per-run stdout/stderr logs when executing.",
  )
  args = parser.parse_args()

  extra_args = shlex.split(args.extra_args)
  presets = STAGES[args.stage]

  print(f"Stage: {args.stage}")
  print("Presets:")
  for preset in presets:
    print(f"  - {preset}")
  print()

  commands = [
      build_command(
          preset=preset,
          max_steps=args.max_steps,
          extra_args=extra_args,
          python_bin=args.python_bin,
          stage=args.stage,
      )
      for preset in presets
  ]

  for cmd in commands:
    print(shlex.join(cmd))

  if not args.execute:
    return

  env = os.environ.copy()
  os.makedirs(args.log_dir, exist_ok=True)
  os.makedirs(os.path.dirname(args.wandb_urls_file), exist_ok=True)
  if not os.path.exists(args.wandb_urls_file):
    with open(args.wandb_urls_file, "w", encoding="utf-8") as f:
      f.write("experiment_tag\tpreset\twandb_url\n")

  print()
  print(f"W&B URL record file: {args.wandb_urls_file}")
  print(f"Per-run logs: {args.log_dir}")
  for cmd in commands:
    experiment_tag = cmd[cmd.index("--experiment_tag") + 1]
    preset = cmd[cmd.index("--ablation_preset") + 1]
    log_path = os.path.join(args.log_dir, f"{experiment_tag}.log")
    seen_urls = set()

    print()
    print(f"[ablation] running: {shlex.join(cmd)}")
    print(f"[ablation] log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
      process = subprocess.Popen(
          cmd,
          env=env,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,
      )
      assert process.stdout is not None
      for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        for url in WANDB_URL_PATTERN.findall(line):
          if url in seen_urls:
            continue
          seen_urls.add(url)
          _append_wandb_url(
              record_file=args.wandb_urls_file,
              preset=preset,
              experiment_tag=experiment_tag,
              url=url,
          )
          print(f"[ablation] recorded wandb url for {experiment_tag}: {url}")
      return_code = process.wait()
      if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


if __name__ == "__main__":
  main()
