"""Validate the completed 20-step matched benchmark and derived visualization."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path):
  return json.loads(path.read_text())


def main():
  comparison = load(ROOT / "stages-comparison.json")
  stage = load(ROOT / "stages-analysis.json")
  timeline = load(ROOT / "training_timeline_summary.json")
  average_chart = load(ROOT / "training_average_summary.json")
  assert comparison["timing_mode"] == "stages"
  assert comparison["run_count"] == {"agentic": 1, "dist": 1}
  assert len(comparison["parity_checks"]) == 12
  assert all(comparison["parity_checks"].values())
  manifests = {}
  for mode in ("agentic", "dist"):
    directory = ROOT / f"stages-{mode}"
    manifest = load(directory / "manifest.json")
    result = load(directory / "result.json")
    summary = load(directory / "summary.json")
    manifests[mode] = manifest
    assert result["returncode"] == 0, (mode, result)
    assert manifest["workload"]["steps"] == 21
    assert manifest["workload"]["warmup"] == 1
    assert manifest["workload"]["micro_groups"] == 1
    assert manifest["agentic_chip_layout"] == "split2x2"
    assert summary["training_input"]["trajectories_per_call"] == 8
    assert summary["actor_log_probs_input"]["trajectories_per_call"] == 8
    assert summary["training_input"]["trajectories_per_step"] == 512
    assert len(stage[mode]["per_step"]) == 20
    assert [item["step"] for item in stage[mode]["per_step"]] == list(range(1, 21))
    assert len(timeline[mode]["steps"]) == 20
    assert stage[mode]["mean_per_step"]["training"]["calls"] == 64
    assert stage[mode]["mean_per_step"]["actor_log_probs"]["calls"] == 64
    averages = timeline[mode]["averages"]
    chart_means = average_chart[mode]
    assert abs(averages["training_seconds"] - stage[mode]["mean_per_step"]["training"]["seconds"]) < 1e-5
    assert abs(averages["full_step_seconds"] - comparison["metrics"]["end_to_end_step_time_seconds"][mode]) < 1e-5
    assert abs(averages["rollout_seconds"] - comparison["metrics"]["rollout_collection_time_seconds"][mode]) < 1e-5
    assert abs(averages["training_during_rollout_seconds"] + averages["training_after_rollout_seconds"] - averages["training_seconds"]) < 1e-5
    assert chart_means["measured_steps"] == 20
    assert abs(sum(chart_means[key] for key in (
        "through_last_trajectory_seconds", "before_sync_seconds",
        "weight_sync_seconds")) - averages["full_step_seconds"]) < 1e-5
    assert abs(chart_means["training_during_rollout_seconds"]
               + chart_means["training_after_rollout_seconds"]
               - averages["training_seconds"]) < 1e-5
    assert abs(chart_means["logprob_during_rollout_seconds"]
               + chart_means["logprob_after_rollout_seconds"]
               - averages["actor_log_probs_seconds"]) < 1e-5
    for item in timeline[mode]["steps"]:
      assert len(item["intervals_from_step_start_seconds"]["training"]) == 64
      assert len(item["intervals_from_step_start_seconds"]["actor_log_probs"]) == 64
      assert len(item["intervals_from_step_start_seconds"]["weight_sync"]) == 1
    print(mode, "step mean", round(averages["full_step_seconds"], 3),
          "rollout", round(averages["rollout_seconds"], 3),
          "training", round(averages["training_seconds"], 3),
          "overlap", round(averages["training_during_rollout_seconds"], 3))
  assert manifests["agentic"]["workload"] == manifests["dist"]["workload"]
  assert manifests["agentic"]["revision"] == manifests["dist"]["revision"]
  assert manifests["agentic"]["diff_sha256"] == manifests["dist"]["diff_sha256"]
  assert (ROOT / "training_timeline.svg").stat().st_size > 10_000
  assert (ROOT / "training_timeline.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
  assert (ROOT / "training_average.svg").stat().st_size > 3_000
  assert (ROOT / "training_average.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
  print("All 20-step validation checks passed")


if __name__ == "__main__":
  main()
