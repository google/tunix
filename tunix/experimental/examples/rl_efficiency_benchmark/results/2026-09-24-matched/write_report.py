"""Render the chip- and microbatch-matched English benchmark report."""

from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def read(path):
  return json.loads(path.read_text())


def num(value, digits=3):
  return "—" if value is None else f"{value:,.{digits}f}"


def row(*cells):
  return "| " + " | ".join(map(str, cells)) + " |"


def percentile(values, fraction):
  ordered = sorted(values)
  index = (len(ordered) - 1) * fraction
  low = int(index)
  return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def comparison_table(comparison, mapping):
  lines = [
      "| Metric | Agentic 2+2 | Distributed 2+2 | Dist − agentic | Relative difference |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for key, label in mapping:
    metric = comparison["metrics"][key]
    lines.append(row(
        label, num(metric["agentic"]), num(metric["dist"]),
        num(metric["dist_minus_agentic"]),
        num(metric["dist_relative_to_agentic_percent"], 2) + "%",
    ))
  return "\n".join(lines)


def event_supplement(directory, warmup, total_steps):
  events = [
      json.loads(line)
      for path in sorted(directory.glob("events*.jsonl"))
      for line in path.read_text().splitlines()
  ]
  generation = [e for e in events if e["kind"] == "generation" and warmup <= e["step"] < total_steps]
  trajectory = [e for e in events if e["kind"] == "rollout_trajectory" and warmup <= e["step"] < total_steps]
  steady_count = total_steps - warmup
  assert len(trajectory) == 512 * steady_count
  assert all(e["ok"] for e in generation)
  assert all(e["ok"] and e["exact_token_continuity"] for e in trajectory)
  assert all(e["conversation_tokens"] == e["generated_tokens"] + e["environment_tokens"] for e in trajectory)
  return {
      "Generation calls": len(generation),
      "Generation calls/step": len(generation) / steady_count,
      "Generation call latency, mean (s)": statistics.mean(e["seconds"] for e in generation),
      "Generation call latency, median (s)": statistics.median(e["seconds"] for e in generation),
      "Generation call prompt tokens, mean": statistics.mean(e["prompt_tokens"] for e in generation),
      "Generation call output tokens, mean": statistics.mean(e["generated_tokens"] for e in generation),
      "Generation output tokens, total": sum(e["generated_tokens"] for e in generation),
      "Trajectory events": len(trajectory),
      "Trajectory latency, median (s)": statistics.median(e["seconds"] for e in trajectory),
      "Initial prompt tokens/trajectory": statistics.mean(e["prompt_tokens"] for e in trajectory),
      "Environment tokens/trajectory": statistics.mean(e["environment_tokens"] for e in trajectory),
      "New conversation tokens/trajectory": statistics.mean(e["conversation_tokens"] for e in trajectory),
      "Environment tokens, total": sum(e["environment_tokens"] for e in trajectory),
      "New conversation tokens, total": sum(e["conversation_tokens"] for e in trajectory),
      "Environment time/trajectory, median (s)": statistics.median(e["environment_seconds"] for e in trajectory),
      "Reward sum": sum(e["reward"] for e in trajectory),
      "SUCCEEDED with reward=1": sum(e["status"] == "SUCCEEDED" and e["reward"] == 1 for e in trajectory),
      "SUCCEEDED with reward=0": sum(e["status"] == "SUCCEEDED" and e["reward"] == 0 for e in trajectory),
      "Reward timing field, nonzero count": sum(e["reward_seconds"] != 0 for e in trajectory),
      "Generation calls by step": [sum(e["step"] == step for e in generation) for step in range(warmup, total_steps)],
  }


def main():
  comparison = read(ROOT / "stages-comparison.json")
  analysis = read(ROOT / "stages-analysis.json")
  agentic = read(ROOT / "stages-agentic" / "summary.json")
  dist = read(ROOT / "stages-dist" / "summary.json")
  ma = read(ROOT / "stages-agentic" / "manifest.json")
  md = read(ROOT / "stages-dist" / "manifest.json")
  assert comparison["timing_mode"] == "stages"
  assert all(comparison["parity_checks"].values())
  assert comparison["run_count"] == {"agentic": 1, "dist": 1}
  assert ma["agentic_chip_layout"] == md["agentic_chip_layout"] == "split2x2"
  assert ma["revision"] == md["revision"] and ma["diff_sha256"] == md["diff_sha256"]
  assert ma["workload"] == md["workload"]
  assert ma["model_metadata_sha256"] == md["model_metadata_sha256"]
  assert ma["model_shards"] == md["model_shards"]
  assert ma["training_contract"] == md["training_contract"]
  assert read(ROOT / "stages-agentic" / "dataset.json") == read(ROOT / "stages-dist" / "dataset.json")
  workload = ma["workload"]
  agentic_config = read(ROOT / "stages-agentic" / "agentic.json")
  dist_env = md["environment"]
  assert agentic_config["model_config"]["model_id"] == dist_env["MODEL_ID"]
  assert agentic_config["env_kwargs"]["is_slippery"] is False
  assert dist_env["IS_SLIPPERY"] == "0"
  assert agentic_config["rollout_config"]["temperature"] == float(dist_env["TEMPERATURE"])
  assert agentic_config["rollout_config"]["top_p"] == float(dist_env["TOP_P"])
  assert agentic_config["rollout_config"]["top_k"] == int(dist_env["TOP_K"])
  assert agentic_config["rl_training_config"]["actor_optimizer_config"]["learning_rate"] == float(dist_env["LEARNING_RATE"])
  optimizer = agentic_config["rl_training_config"]["actor_optimizer_config"]
  for key, env_key in (("b1", "ADAM_B1"), ("b2", "ADAM_B2")):
    assert optimizer[key] == float(dist_env[env_key])
  assert optimizer["chain_kwargs"]["max_norm"] == float(dist_env["MAX_GRAD_NORM"])
  objective = agentic_config["agentic_grpo_config"]
  for key, env_key in (("loss_algo", "LOSS_ALGO"), ("advantage_estimator", "ADVANTAGE_ESTIMATOR")):
    assert objective[key] == dist_env[env_key]
  for key, env_key in (("beta", "BETA"), ("epsilon", "EPSILON"), ("epsilon_high", "EPSILON_HIGH")):
    assert objective[key] == float(dist_env[env_key])
  assert objective["off_policy_steps"] == int(dist_env["OFF_POLICY_STEPS"])
  assert agentic_config["vllm_config"]["max_num_seqs"] == int(dist_env["VLLM_MAX_NUM_SEQS"])
  assert agentic_config["vllm_config"]["max_num_batched_tokens"] == int(dist_env["VLLM_MAX_NUM_BATCHED_TOKENS"])
  assert int(dist_env["TRAIN_MICRO_BATCH_SIZE"]) == workload["micro_groups"] * workload["generations"]
  assert int(dist_env["COMPUTE_LOGPS_MICRO_BATCH_SIZE"]) == workload["micro_groups"] * workload["generations"]
  warmup = workload["warmup"]
  total_steps = workload["steps"]
  steady_count = total_steps - warmup
  assert warmup == 1 and steady_count == 20
  steady_label = f"{warmup}–{total_steps - 1}"
  microbatch_size = workload["micro_groups"] * workload["generations"]
  training_calls_per_step = workload["batch"] // workload["micro_groups"]
  a = analysis["agentic"]
  d = analysis["dist"]
  aa, dd = a["mean_per_step"], d["mean_per_step"]
  ta, td = a["mean_post_rollout"], d["mean_post_rollout"]
  metrics = comparison["metrics"]
  overlap = read(ROOT / "overlap-analysis.json")
  timeline = read(ROOT / "training_timeline_summary.json")
  steady_overlap = {
      mode: [item for item in overlap[mode] if item["step"] > 0]
      for mode in ("agentic", "dist")
  }
  microbatch = {
      mode: [duration for item in items for duration in item["training_microbatch_durations_seconds"]]
      for mode, items in steady_overlap.items()
  }
  nonfinal_microbatch = {
      mode: [duration for item in items for duration in item["nonfinal_training_microbatch_durations_seconds"]]
      for mode, items in steady_overlap.items()
  }
  for mode, stage_mean in (("agentic", aa["training"]["seconds"]), ("dist", dd["training"]["seconds"])):
    assert len(microbatch[mode]) == training_calls_per_step * len(steady_overlap[mode])
    assert abs(statistics.mean(microbatch[mode]) * training_calls_per_step - stage_mean) < 1e-5
    assert abs(
        statistics.mean(item["rollout_collection_seconds"] for item in steady_overlap[mode])
        - metrics["rollout_collection_time_seconds"][mode]
    ) < 1e-5
    assert abs(timeline[mode]["averages"]["training_seconds"] - stage_mean) < 1e-5

  def overlap_mean(mode, key):
    return statistics.mean(item[key] for item in steady_overlap[mode])

  lines = [
      "# Agentic vs Distributed Performance Benchmark",
      "",
      "Date: 24 September 2026 (UTC). One four-chip TPU v5p host. "
      "The agentic actor uses FSDP2 on chips 0–1 and its vLLM rollout uses "
      "TP2 on chips 2–3; distributed uses the same two-plus-two allocation. "
      "The agentic rollout device indexes [2, 3] were observed in the "
      "vLLM initialization log.",
      "",
      "## Answer",
      "",
      "**Observed E2E gap.** Across steps 1–20, excluding warmup step 0, "
      "distributed averaged **293.18 s/step** versus agentic's "
      "**303.33 s/step**: **10.14 s/step (3.34%) faster**. It was faster "
      "on **15/20** steps, but the last **10** averaged **288.91** versus "
      "**288.82 s/step** (distributed then agentic), effectively equal.",
      "",
      "**Where the time goes.** Distributed saved **39.64 s/step** in "
      "rollout collection but spent **29.48 s/step more** from the last "
      "trajectory to weight-sync completion. About **0.01 s/step** of "
      "rollout-start delay accounts for the remainder of the net "
      "**10.14 s/step** gap. These are sequential segments of the step "
      "wall clock; completed trainer API times overlap rollout and cannot "
      "be added as separate E2E savings.",
      "",
      "**Why the longer tail.** Distributed's completed training calls "
      "took **76.20 versus 84.91 s/step** and actor log-prob "
      "**20.81 versus 75.41 s/step** (distributed versus agentic). Yet "
      "only **66.03% versus 96.19%** of training time overlapped rollout. "
      "After the last trajectory, **25.89 versus 3.24 s/step** of training "
      "remained; weight sync itself was **19.02 versus 18.43 s/step**. "
      "Thus the measured tail is dominated by outstanding trainer-side "
      "work rather than weight-sync duration.",
      "",
      "**Scope of the evidence.** The 2+2 chips and 8-trajectory "
      "microbatches are aligned, so this run locates its observed timing "
      "gap. Online outcomes differ: distributed produced **969.35 versus "
      "842.01 output tokens/trajectory** and its success rate was "
      "**48.75% versus 56.56%** (distributed versus agentic). With one "
      "sequential run per stack in stage mode, fixed-input trainer speed, "
      "output-controlled rollout speed, run-to-run uncertainty, and "
      "pipeline-mode throughput remain untested.",
      "",
      "The comparison matches chip allocation, model checkpoint, tokenizer, "
      "dataset order, total trajectories, sequence padding, precision, seed, "
      "timing mode, and actual train/log-prob microbatch shapes. Both stacks "
      f"handle {microbatch_size} trajectories per call and make {training_calls_per_step} calls per step. The "
      "implementations still use different local versus RPC and weight-sync "
      "paths, and sampled rollouts need not contain identical tokens.",
      "",
      "A matched 16-trajectory attempt failed on dist's first trainer "
      "forward/backward call: XLA needed 75.81 GiB while 56.77 GiB was "
      "available. It contributes no speed result. We therefore used one "
      "complete eight-generation prompt group per microbatch on both stacks. "
      "The failed attempt and this change are recorded in "
      "[`alignment_audit.md`](alignment_audit.md).",
      "",
      "## Requested timing breakdown",
      "",
      f"| Metric, steady steps {steady_label} | Agentic 2+2 | Distributed 2+2 | Dist − agentic | Dist / agentic |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  timing_rows = [
      ("Full step (s/step)", metrics["end_to_end_step_time_seconds"]["agentic"], metrics["end_to_end_step_time_seconds"]["dist"]),
      ("Rollout collection, first to last trajectory (s/step)", metrics["rollout_collection_time_seconds"]["agentic"], metrics["rollout_collection_time_seconds"]["dist"]),
      ("Actor log-prob (s/step)", aa["actor_log_probs"]["seconds"], dd["actor_log_probs"]["seconds"]),
      ("Sampler/trainer agreement (s/step)", aa["sampler_trainer_agreement"]["seconds"], dd["sampler_trainer_agreement"]["seconds"]),
      (f"Training, all {training_calls_per_step} microbatches plus update (s/step)", aa["training"]["seconds"], dd["training"]["seconds"]),
      ("One training microbatch, mean (s)", statistics.mean(microbatch["agentic"]), statistics.mean(microbatch["dist"])),
      ("One training microbatch, median (s)", statistics.median(microbatch["agentic"]), statistics.median(microbatch["dist"])),
      ("One training microbatch, P90 (s)", percentile(microbatch["agentic"], .9), percentile(microbatch["dist"], .9)),
      ("Nonfinal training microbatch, mean (s)", statistics.mean(nonfinal_microbatch["agentic"]), statistics.mean(nonfinal_microbatch["dist"])),
      ("Final training microbatch with update, mean (s)", overlap_mean("agentic", "final_training_microbatch_with_update_seconds"), overlap_mean("dist", "final_training_microbatch_with_update_seconds")),
      ("Last trajectory to weight-sync end (s/step)", overlap_mean("agentic", "last_trajectory_to_sync_end_seconds"), overlap_mean("dist", "last_trajectory_to_sync_end_seconds")),
      ("Weight sync (s/step)", aa["weight_sync"]["seconds"], dd["weight_sync"]["seconds"]),
  ]
  for label, av, dv in timing_rows:
    lines.append(row(label, num(av), num(dv), num(dv-av), num(dv/av,2)+"×"))
  lines += [
      "",
      "All durations are synchronized host wall times. Rollout and training "
      "overlap, so their totals must not be added to estimate full-step time. "
      "The last-trajectory-to-sync-end interval equals the post-rollout "
      "tail in this trace because the step boundary is sync completion. "
      f"Each microbatch contains {microbatch_size} trajectories. The per-microbatch sample "
      f"has {training_calls_per_step * len(steady_overlap['agentic'])} calls per stack across {steady_count} measured steps; the final call "
      "in each step also includes the optimizer update. Distributed has a separate "
      "update RPC span, while agentic's optimizer update is fused inside "
      "that final JAX training call, so this report does not claim an "
      "optimizer-only speed comparison.",
      "",
      "## Completed stages",
      "",
      "| Stage, mean seconds per step | Agentic 2+2 | Distributed 2+2 | Dist − agentic | Dist / agentic |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for key, label in (
      ("actor_log_probs", "Actor log-prob"),
      ("sampler_trainer_agreement", "Sampler/trainer agreement"),
      ("training", "Forward/backward and update"),
      ("weight_sync", "Weight synchronization"),
  ):
    av, dv = aa[key]["seconds"], dd[key]["seconds"]
    lines.append(row(label, num(av), num(dv), num(dv-av), num(dv/av,2)+"×"))
  for key, label in (
      ("rollout_collection_time_seconds", "Rollout collection"),
      ("post_rollout_tail_seconds", "Post-rollout tail"),
      ("end_to_end_step_time_seconds", "Full synchronized step"),
  ):
    v = metrics[key]
    av, dv = v["agentic"], v["dist"]
    lines.append(row(label,num(av),num(dv),num(dv-av),num(dv/av,2)+"×"))
  lines += [
      "",
      "The stage spans are device-fenced host wall times. Training begins "
      "after a microbatch is ready and overlaps ongoing rollout collection; "
      "it does not wait for all 512 trajectories. Nested RPCs must not be "
      "added to the full-step time. "
      "Weight-sync host time includes readiness and installation, not solely "
      "network transfer.",
      "",
      "## Post-rollout critical path",
      "",
      "| Mean interval per step | Agentic 2+2 | Distributed 2+2 | Dist − agentic |",
      "| --- | ---: | ---: | ---: |",
  ]
  for label, av, dv in (
      ("Final trajectory to step end (s)",ta["tail_seconds"],td["tail_seconds"]),
      ("Final trajectory to sync start (s)",ta["pre_sync_seconds"],td["pre_sync_seconds"]),
      ("Weight sync (s)",ta["sync_seconds"],td["sync_seconds"]),
      ("Training within pre-sync window (s)",ta["pre_sync_span_seconds"]["training"],td["pre_sync_span_seconds"]["training"]),
      ("Log-prob within pre-sync window (s)",ta["pre_sync_span_seconds"]["actor_log_probs"],td["pre_sync_span_seconds"]["actor_log_probs"]),
      ("Agreement within pre-sync window (s)",ta["pre_sync_span_seconds"]["sampler_trainer_agreement"],td["pre_sync_span_seconds"]["sampler_trainer_agreement"]),
      ("Log-prob calls started after final trajectory",ta["late_logprob_calls"],td["late_logprob_calls"]),
  ):
    lines.append(row(label,num(av),num(dv),num(dv-av)))
  lines += [
      "",
      "These operation spans are clipped to the pre-sync interval. "
      "The last rollout completion does not imply that all queued "
      "microbatches have finished. The post-rollout tail measures the "
      "remaining work after the final trajectory, not the entire trainer cost.",
      "",
      "## Rollout and training overlap",
      "",
      "| Step | Agentic trajectories ready at first train | Dist trajectories ready at first train | Agentic first-train lead before last trajectory (s) | Dist first-train lead before last trajectory (s) |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for agentic_step, dist_step in zip(overlap["agentic"], overlap["dist"]):
    assert agentic_step["step"] == dist_step["step"]
    if agentic_step["step"] == 0:
      continue
    lines.append(row(
        agentic_step["step"],
        agentic_step["trajectories_completed_at_first_training"],
        dist_step["trajectories_completed_at_first_training"],
        num(agentic_step["seconds_between_first_training_and_last_trajectory"]),
        num(dist_step["seconds_between_first_training_and_last_trajectory"]),
    ))
  lines += [
      "",
      "Each step has 512 trajectories. The first training call begins once "
      f"one {microbatch_size}-trajectory microbatch is ready, while other trajectories continue "
      f"to finish. Counts can exceed {microbatch_size} because production continues during "
      "preprocessing and scheduling. Both implementations finish the step's "
      "gradient accumulation before optimizer update and weight sync.",
      "",
      "## Average stages and measured overlap",
      "",
      "### Mean-duration diagram",
      "",
      f"![Average pipeline stages and overlap across {steady_count} measured steps](training_average.png)",
      "",
      "Mean rollout and weight-sync bars occupy separate rows on the "
      "step-relative clock. Training and actor log-prob durations are split "
      "into work during and after rollout. For a compact average, those "
      "durations are packed against the mean rollout boundary; their "
      "horizontal placement is schematic, not actual call timestamps. The "
      "dashed line marks that boundary. Concurrent stage durations must not "
      "be added to full-step time.",
      "",
      f"### Detailed {steady_count}-step timeline",
      "",
      f"![Actual rollout, log-prob, training and weight-sync spans for all {steady_count} measured steps](training_timeline.png)",
      "",
      "Each thin stripe is one measured step aligned to its own start. Blue "
      "marks rollout collection; purple and orange mark completed log-prob "
      "and training calls; green marks weight sync. Unlike the mean diagram, "
      "these horizontal positions are actual observed timestamps. The dashed "
      "line marks mean last-trajectory completion.",
      "",
      f"| Overlap metric, steady steps {steady_label} | Agentic 2+2 | Distributed 2+2 |",
      "| --- | ---: | ---: |",
  ]
  for key, label in (
      ("training_during_rollout_seconds", "Training within rollout window (s/step)"),
      ("training_after_rollout_seconds", "Training after rollout window (s/step)"),
      ("training_overlap_share_percent", "Training within rollout window (% of training)"),
      ("first_training_after_rollout_start_seconds", "First training after rollout start (s)"),
  ):
    av = timeline["agentic"]["averages"][key]
    dv = timeline["dist"]["averages"][key]
    lines.append(row(label, num(av, 2), num(dv, 2)))
  lines += [
      "",
      "The overlap seconds clip each training-call interval to its step's "
      f"rollout collection window, then average the {steady_count} steps. The fraction "
      "is overlap seconds divided by all completed training-call seconds; "
      "it does not measure chip utilization. Per-step intervals and derived "
      "values are in [`training_timeline_summary.json`](training_timeline_summary.json).",
      "",
      "## Per-step results",
      "",
      "A = agentic; D = distributed. Speedup is "
      "(agentic E2E / distributed E2E − 1) × 100%; positive values favor "
      "distributed. All durations are seconds.",
      "",
      "| Step | E2E A (s) | E2E D (s) | D speedup (%) | Rollout A (s) | Rollout D (s) | Tail A (s) | Tail D (s) |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for ar, dr, ao, do in zip(a["per_step"], d["per_step"], steady_overlap["agentic"], steady_overlap["dist"]):
    assert ar["step"] == dr["step"] == ao["step"] == do["step"]
    lines.append(row(
        ar["step"], num(ar["end_to_end_seconds"], 2), num(dr["end_to_end_seconds"], 2),
        f"{100 * (ar['end_to_end_seconds'] / dr['end_to_end_seconds'] - 1):+.2f}%",
        num(ao["rollout_collection_seconds"], 2), num(do["rollout_collection_seconds"], 2),
        num(ao["last_trajectory_to_sync_end_seconds"], 2),
        num(do["last_trajectory_to_sync_end_seconds"], 2),
    ))
  lines += [
      "",
      "| Step | Train A (s) | Train D (s) | Log-prob A (s) | Log-prob D (s) | Microbatch A (s) | Microbatch D (s) |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for ar, dr, ao, do in zip(a["per_step"], d["per_step"], steady_overlap["agentic"], steady_overlap["dist"]):
    lines.append(row(
        ar["step"], num(ar["training"]["seconds"], 2), num(dr["training"]["seconds"], 2),
        num(ar["actor_log_probs"]["seconds"], 2), num(dr["actor_log_probs"]["seconds"], 2),
        num(statistics.mean(ao["training_microbatch_durations_seconds"]), 3),
        num(statistics.mean(do["training_microbatch_durations_seconds"]), 3),
    ))
  lines += [
      "",
      f"Each stack ran once, in distributed then agentic order. {steady_count} "
      "consecutive steady steps describe within-run variation and do not "
      "establish run-to-run confidence intervals.",
      "",
      "## Distributed RPC versus completed worker time",
      "",
      "| Operation | Calls/step | Outer RPC (s/step) | Completed worker (s/step) | Outside worker (s/step) |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for key,label in (
      ("worker_rpc_per_token_logps","Actor log-prob"),
      ("worker_rpc_fwd_bwd","Forward/backward"),
      ("worker_rpc_update","Optimizer update"),
  ):
    v=d["rpc_worker_pairs"][key]
    lines.append(row(label,num(v["calls"]/steady_count,0),num(v["rpc_seconds_per_step"]),num(v["worker_seconds_per_step"]),num(v["outside_worker_seconds_per_step"])))
  lines += [
      "",
      "The worker spans include Python/JAX work, result materialization, and "
      "device-completion waits; they are not pure chip kernel times. Every "
      "worker span is nested in exactly one RPC and has completion evidence.",
      "",
      "## Workload and measurement controls",
      "",
      "| Item | Agentic 2+2 | Distributed 2+2 |",
      "| --- | ---: | ---: |",
  ]
  for label,av,dv in (
      ("Trajectories/step",agentic["training_input"]["trajectories_per_step"],dist["training_input"]["trajectories_per_step"]),
      ("Training trajectories/call",agentic["training_input"]["trajectories_per_call"],dist["training_input"]["trajectories_per_call"]),
      ("Training calls/step",aa["training"]["calls"],dd["training"]["calls"]),
      ("Actor log-prob trajectories/call",agentic["actor_log_probs_input"]["trajectories_per_call"],dist["actor_log_probs_input"]["trajectories_per_call"]),
      ("Actor log-prob calls/step",aa["actor_log_probs"]["calls"],dd["actor_log_probs"]["calls"]),
      ("Padded token slots/step",agentic["training_input"]["padded_token_slots_per_step"],dist["training_input"]["padded_token_slots_per_step"]),
      ("Output tokens/trajectory",agentic["rollout"]["average_output_tokens_per_trajectory"],dist["rollout"]["average_output_tokens_per_trajectory"]),
      ("Reward/trajectory",agentic["rollout"]["average_reward_per_trajectory"],dist["rollout"]["average_reward_per_trajectory"]),
  ):
    lines.append(row(label,num(av,2),num(dv,2)))
  lines += [
      "",
      "All comparison parity checks passed: training and log-prob "
      "sequence lengths, trajectories per step, padded token slots per "
      "step, per-call trajectory counts, full tensor shapes, and calls per "
      "step match. Both stacks use BF16 compute, FP32 checkpoint loading, "
      "and exact token continuity. Prompt and completion tensors have "
      "length 2048 each. Sampled output tokens and reward are online outcomes "
      "and can differ across stacks. Both runs exited successfully.",
      "",
      "## Complete aggregate metrics",
      "",
      comparison_table(comparison, [
          ("end_to_end_step_time_seconds","Full step (s/step)"),
          ("rollout_collection_time_seconds","Rollout collection (s/step)"),
          ("trajectory_latency_mean_seconds","Trajectory latency mean (s)"),
          ("trajectory_latency_p90_seconds","Trajectory latency P90 (s)"),
          ("rollout_start_delay_seconds","Rollout start delay (s/step)"),
          ("post_rollout_tail_seconds","Post-rollout tail (s/step)"),
          ("average_output_tokens_per_trajectory","Output tokens/trajectory"),
          ("average_prompt_tokens_per_model_call","Prompt tokens/model call"),
          ("average_turns_per_trajectory","Turns/trajectory"),
          ("average_reward_per_trajectory","Reward/trajectory"),
          ("model_generation_time_per_trajectory_seconds","Generation API time/trajectory (s)"),
          ("environment_time_per_trajectory_seconds","Environment time/trajectory (s)"),
      ]),
      "",
      "| Outcome rate | Agentic 2+2 | Distributed 2+2 | Dist − agentic |",
      "| --- | ---: | ---: | ---: |",
  ]
  for key in ("SUCCEEDED","MAX_CONTEXT_LIMIT_REACHED"):
    v=comparison["trajectory_outcome_percent"][key]
    lines.append(row(key,num(v["agentic"]),num(v["dist"]),num(v["dist_minus_agentic"])+" pp"))
  lines += [
      "",
      "`SUCCEEDED` records normal episode completion; it does not by "
      "itself mean reward=1. Use reward/trajectory for task outcome quality.",
      "",
      "## Complete recorded API counts and host spans",
      "",
      "| API span | Agentic calls/step | Dist calls/step | Agentic host s/step | Dist host s/step |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for key,v in sorted(comparison["api_calls"].items()):
    calls=v["calls_per_step"]
    duration=v["host_time_per_step_seconds"]
    lines.append(row(
        f"`{key}`",num(calls["agentic"],2),num(calls["dist"],2),
        num(duration["agentic"],4),num(duration["dist"],4),
    ))
  ra=read(ROOT / "stages-agentic" / "result.json")
  rd=read(ROOT / "stages-dist" / "result.json")
  lines += [
      "",
      "These host spans can nest. `rollout_poll` largely records waiting and "
      "must not be added to trainer or full-step durations.",
      "",
      "## Process completion",
      "",
      "| Item | Agentic 2+2 | Distributed 2+2 |",
      "| --- | ---: | ---: |",
      row("Exit code",ra["returncode"],rd["returncode"]),
      row("Process runtime including startup/shutdown (s)",num(ra["total_run_time_seconds"],2),num(rd["total_run_time_seconds"],2)),
      row("Warmup step 0 (s)",num(agentic["end_to_end"]["warmup_step_time_seconds"]["mean"],2),num(dist["end_to_end"]["warmup_step_time_seconds"]["mean"],2)),
      row("Steady step mean (s)",num(agentic["end_to_end"]["steady_state_step_time_seconds"]["mean"],2),num(dist["end_to_end"]["steady_state_step_time_seconds"]["mean"],2)),
      row("Steady step median (s)",num(agentic["end_to_end"]["steady_state_step_time_seconds"]["median"],2),num(dist["end_to_end"]["steady_state_step_time_seconds"]["median"],2)),
      row("Steady step sample SD (s)",num(statistics.stdev(agentic["end_to_end"]["steady_state_step_time_seconds"]["per_step"]),2),num(statistics.stdev(dist["end_to_end"]["steady_state_step_time_seconds"]["per_step"]),2)),
      row("Trajectories/full-step second",num(512/agentic["end_to_end"]["steady_state_step_time_seconds"]["mean"],3),num(512/dist["end_to_end"]["steady_state_step_time_seconds"]["mean"],3)),
      "",
      "Process runtime includes time outside the training-step boundaries "
      "and is not the steady-step metric.",
  ]
  supplements = ROOT / "event-supplements.json"
  if supplements.exists():
    ea, ed = (read(supplements)[mode] for mode in ("agentic", "dist"))
  else:
    ea = event_supplement(ROOT / "stages-agentic", warmup, total_steps)
    ed = event_supplement(ROOT / "stages-dist", warmup, total_steps)
  lines += [
      "",
      "## Raw event supplements",
      "",
      f"| Event metric, steady steps {steady_label} | Agentic 2+2 | Distributed 2+2 |",
      "| --- | ---: | ---: |",
  ]
  for key in ea:
    if key == "Generation calls by step":
      continue
    av,dv=ea[key],ed[key]
    digits = 3 if any(isinstance(v,float) and not v.is_integer() for v in (av,dv)) else 0
    lines.append(row(key,num(av,digits),num(dv,digits)))
  lines += [
      "",
      "| Step | Agentic generation calls | Distributed generation calls |",
      "| --- | ---: | ---: |",
  ]
  for step in range(warmup,total_steps):
    lines.append(row(step,ea["Generation calls by step"][step-warmup],ed["Generation calls by step"][step-warmup]))
  lines += [
      "",
      "A trajectory can contain multiple generation calls. Every measured "
      "generation and trajectory has `ok=true`; every trajectory has "
      "`exact_token_continuity=true` and its conversation-token count equals "
      "generated plus environment tokens. Zero reward-timing fields indicate "
      "unavailable timing measurements, not zero reward computation cost. "
      "The environment-time field includes executor queue and wall waiting, "
      "not just FrozenLake compute.",
  ]
  lines += [
      "",
      "## Reproducibility",
      "",
      f"Source revision: `{ma['revision']}` with matching working-tree diff "
      "hashes in both manifests. The benchmark command used "
      "`--agentic-chip-layout split2x2 --match-training-microbatch "
      f"--timing-mode stages --steps {total_steps} "
      "--warmup 1 --batch 64 --generations 8 --micro-groups 1 --prompt 2048 "
      "--response 2048 --turns 8 --concurrency 512 --seed 42`. Exact commands, "
      "environment, source and model hashes are in "
      "[`stages-agentic/manifest.json`](stages-agentic/manifest.json) and "
      "[`stages-dist/manifest.json`](stages-dist/manifest.json). Raw logs and "
      "events remain in the original run directories, outside this snapshot. "
      "Unrounded values and all API "
      "spans are in [`stages-comparison.json`](stages-comparison.json) and "
      "[`stages-analysis.json`](stages-analysis.json), and "
      "[`overlap-analysis.json`](overlap-analysis.json). "
      "Configuration and residual differences are documented in "
      "[`alignment_audit.md`](alignment_audit.md). "
      "The benchmark configuration code is "
      "[`frozenlake.py`](../../frozenlake.py).",
      "",
  ]
  # Lead with the result and its interpretation; retain the full measurements
  # below for auditability. The old stage table and prose interpretation repeat
  # values already present in the detailed tables.
  def section(heading):
    start = lines.index("## " + heading)
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return lines[start:end]

  def subsection(heading, new_heading=None):
    result = section(heading).copy()
    result[0] = "### " + (new_heading or heading)
    return ["#" + line if line.startswith("### ") and line != result[0] else line
            for line in result]

  timeline_section = section("Average stages and measured overlap")
  detailed_start = next(i for i, line in enumerate(timeline_section)
                        if line.startswith("### Detailed"))
  detail_timeline = ["### Measured overlap and 20-step timeline", ""] + [
      "#" + line if line.startswith("### ") else line
      for line in timeline_section[detailed_start:]
  ]

  key_rows = [
      ("Full step (s/step)", metrics["end_to_end_step_time_seconds"]["agentic"], metrics["end_to_end_step_time_seconds"]["dist"], 2),
      ("Rollout collection (s/step)", metrics["rollout_collection_time_seconds"]["agentic"], metrics["rollout_collection_time_seconds"]["dist"], 2),
      ("Post-rollout tail (s/step)", metrics["post_rollout_tail_seconds"]["agentic"], metrics["post_rollout_tail_seconds"]["dist"], 2),
      ("Training calls incl. update (s/step)", aa["training"]["seconds"], dd["training"]["seconds"], 2),
      ("One training microbatch (s)", statistics.mean(microbatch["agentic"]), statistics.mean(microbatch["dist"]), 3),
      ("Training after last trajectory (s/step)", timeline["agentic"]["averages"]["training_after_rollout_seconds"], timeline["dist"]["averages"]["training_after_rollout_seconds"], 2),
      ("Actor log-prob (s/step)", aa["actor_log_probs"]["seconds"], dd["actor_log_probs"]["seconds"], 2),
      ("Weight sync (s/step)", aa["weight_sync"]["seconds"], dd["weight_sync"]["seconds"], 2),
  ]
  key_table = [
      "| Mean over 20 measured steps | Agentic 2+2 | Distributed 2+2 | Dist − agentic |",
      "| --- | ---: | ---: | ---: |",
  ] + [row(label, num(av, digits), num(dv, digits), num(dv-av, digits))
       for label, av, dv, digits in key_rows]
  reward = metrics["average_reward_per_trajectory"]
  key_table.append(row("Reward/trajectory", num(reward["agentic"]),
                       num(reward["dist"]),
                       num(reward["dist_minus_agentic"])))

  lines = [
      "# Agentic vs Distributed Performance Benchmark",
      "",
      "24 September 2026 · TPU v5p · 2 trainer + 2 rollout chips per stack · "
      "20 measured steps after one warmup · 512 trajectories/step · "
      "8 trajectories/microbatch.",
      "",
      "## Key findings",
      "",
      "Distributed averaged **293.18 s/step** versus **303.33 s/step** for "
      "agentic: **10.14 s (3.34%) less time per step**, equivalent to "
      f"**{100 * (metrics['end_to_end_step_time_seconds']['agentic'] / metrics['end_to_end_step_time_seconds']['dist'] - 1):.2f}% higher step throughput**. "
      "It was faster on 15/20 steps, but the last ten averaged **288.91 "
      "versus 288.82 s/step** (distributed versus agentic): effectively equal.",
      "",
      *key_table,
      "",
      "## Where the gap occurs",
      "",
      "The E2E difference is **−39.64 s/step** in rollout collection, "
      "**+29.48 s/step** after the last trajectory, and about **+0.01 "
      "s/step** in rollout-start delay (distributed minus agentic). "
      "Distributed's training calls were faster, but only **66.03%** of "
      "their time overlapped rollout versus **96.19%** for agentic. "
      "It had **25.89 versus 3.24 s/step** of training left after rollout; "
      "weight sync differed by just **0.58 s/step**. The longer tail is "
      "primarily unfinished trainer-side work, not slower weight sync.",
      "",
      "## Average timeline",
      "",
      "![Average pipeline stages and overlap across 20 measured steps](training_average.png)",
      "",
      "[Download the average chart as SVG](training_average.svg).",
      "",
      "Bars show mean durations and overlap; the placement of training and "
      "log-prob bars is schematic. Rollout and training overlap, so their "
      "durations must not be added to reconstruct E2E time.",
      "",
      "## Experimental setup",
      "",
      "| Parameter | Shared setting |",
      "| --- | --- |",
      "| Hardware | TPU v5p-4; trainer chips 0–1 (FSDP2), rollout chips 2–3 (TP2) |",
      "| Model / precision | google/gemma-4-E2B-it; shared checkpoint and tokenizer; BF16 compute, FP32 load |",
      "| Task / dataset | FrozenLake, non-slippery; 1,344 matched prompts |",
      "| Run | 21 steps: 1 warmup + 20 measured; seed 42; distributed then agentic |",
      "| Batch | 64 prompts/step × 8 generations = 512 trajectories/step |",
      "| Microbatch | 8 trajectories/call; 64 training + 64 log-prob calls/step; 1 optimizer update/step |",
      "| Context / turns | 2,048 prompt + 2,048 response tokens; up to 8 turns |",
      "| Sampling | temperature 0.7; top_p 1.0; top_k 0 |",
      "| vLLM scheduling | concurrency 512; max_num_seqs 32; max_num_batched_tokens 8,192; HBM utilization 0.2 |",
      "| Objective | GSPO-token; RLOO; beta 0; epsilon 0.003/0.005; off-policy steps 0 |",
      "| Optimizer | AdamW; learning rate 1e-6; b1 0.9; b2 0.95; global-norm clip 100 |",
      "| Timing | stages mode; synchronized host wall time |",
      "",
      "## Comparison limits",
      "",
      "Chip allocation, checkpoint, prompt order, tensor shapes and call "
      "counts are aligned; sampled trainer tokens are not identical. "
      "Online trajectories differ: distributed generated **969.35 versus "
      "842.01 output tokens/trajectory** and had mean reward **0.479 "
      "versus 0.547** (distributed versus agentic). The timing comparison "
      "therefore does not establish speed at equal task quality or isolate "
      "inference kernels. Each stack ran once (distributed first) in stage "
      "mode; fixed-input "
      "trainer speed, run-to-run uncertainty and pipeline throughput remain "
      "unmeasured.",
      "",
      "## Detailed measurements",
      "",
      *subsection("Requested timing breakdown", "Stage timings"),
      *subsection("Post-rollout critical path"),
      *detail_timeline,
      *subsection("Workload and measurement controls"),
      *subsection("Complete aggregate metrics"),
      *subsection("Distributed RPC versus completed worker time"),
      "## Appendix: step-level and raw records", "",
      *subsection("Per-step results"),
      *subsection("Rollout and training overlap"),
      *subsection("Complete recorded API counts and host spans"),
      *subsection("Process completion"),
      *subsection("Raw event supplements"),
      "Agentic's first and last ten steps averaged **317.83 and 288.82 "
      "s/step**; distributed averaged **297.45 and 288.91 s/step**. "
      "The earlier four-step run found a **39.42 s/step (12.21%)** "
      "distributed advantage. The longer-run estimate above supersedes "
      "that short-run gap.",
      "",
      *subsection("Reproducibility"),
  ]
  lines = [line.replace("Agentic 2+2", "Agentic")
           .replace("Distributed 2+2", "Distributed") for line in lines]
  output=ROOT / "report.md"
  output.write_text("\n".join(lines).rstrip() + "\n")
  print(output)


if __name__ == "__main__":
  main()
