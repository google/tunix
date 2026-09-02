#!/usr/bin/env python3
"""WandB Run Data & Telemetry Exporter.

Extracts complete non-downsampled scalar training histories, hyperparameter configs,
and full console stdout/stderr logs from Weights & Biases experiment runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

DEFAULT_ENTITY = "yuxzhang-google"
DEFAULT_PROJECT = "zero-tim-p57-frozenlake-tim"


def get_wandb_api_key() -> str:
    """Retrieve WandB API key from environment, netrc, or live cluster pod."""
    # 1. Environment variables
    for env_k in ["WANDB_API_KEY", "INJECTED_WANDB_API_KEY"]:
        val = os.environ.get(env_k)
        if val:
            return val.strip()

    # 2. Check ~/.netrc
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.isfile(netrc_path):
        try:
            with open(netrc_path, "r", encoding="utf-8") as f:
                content = f.read()
                m = re.search(r"machine\s+api\.wandb\.ai.*?password\s+([^\s]+)", content, re.DOTALL)
                if m:
                    return m.group(1).strip()
        except Exception:
            pass

    # 3. Fallback: inspect live cluster head pod
    try:
        cmd = [
            "kubectl", "get", "pods", "-l", "jobset.sigs.k8s.io/jobset-name",
            "-o", "jsonpath={.items[?(@.metadata.labels.jobset\\.sigs\\.k8s\\.io/role=='head')].metadata.name}"
        ]
        pods_out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        if pods_out:
            head_pod = pods_out.split()[0]
            env_cmd = ["kubectl", "exec", head_pod, "-c", "jax-tpu", "--", "env"]
            env_out = subprocess.run(env_cmd, capture_output=True, text=True, timeout=10).stdout
            for line in env_out.splitlines():
                if line.startswith("INJECTED_WANDB_API_KEY=") or line.startswith("WANDB_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        pass

    raise RuntimeError("WandB API key not found. Please set WANDB_API_KEY environment variable.")


def parse_run_target(target: str, default_entity: str = DEFAULT_ENTITY, default_project: str = DEFAULT_PROJECT) -> Tuple[str, str, str]:
    """Parse a WandB URL or Run ID into (entity, project, run_id)."""
    target = target.strip()
    # e.g. https://wandb.ai/yuxzhang-google/zero-tim-p57-frozenlake-tim/runs/o3k6jvb4/logs?nw=...
    url_m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([a-zA-Z0-9_-]+)", target)
    if url_m:
        return url_m.group(1), url_m.group(2), url_m.group(3)

    # e.g. entity/project/run_id
    parts = target.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return default_entity, parts[0], parts[1]
    else:
        return default_entity, default_project, target


def fetch_console_logs_graphql(api_key: str, entity: str, project: str, run_id: str) -> List[str]:
    """Fetch raw console stdout logs via WandB GraphQL API with cursor pagination."""
    query = """
    query ModelLogs($entity: String, $project: String, $name: String, $after: String) {
      model(name: $project, entityName: $entity) {
        bucket(name: $name) {
          logLines(first: 5000, after: $after) {
            pageInfo {
              hasNextPage
              endCursor
            }
            edges {
              node {
                line
                number
              }
            }
          }
        }
      }
    }
    """
    lines: List[str] = []
    cursor: Optional[str] = None
    has_next = True

    while has_next:
        variables: Dict[str, Any] = {
            "entity": entity,
            "project": project,
            "name": run_id,
        }
        if cursor:
            variables["after"] = cursor

        resp = requests.post(
            "https://api.wandb.ai/graphql",
            auth=("api", api_key),
            json={"query": query, "variables": variables},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  ⚠️ Warning: GraphQL log fetch returned HTTP {resp.status_code}: {resp.text[:100]}")
            break

        data = resp.json()
        if "errors" in data and data["errors"]:
            print(f"  ⚠️ Warning: GraphQL errors: {data['errors']}")
            break

        bucket = (data.get("data") or {}).get("model", {}).get("bucket")
        if not bucket or not bucket.get("logLines"):
            break

        log_data = bucket["logLines"]
        edges = log_data.get("edges", [])
        for edge in edges:
            node = edge.get("node", {})
            if "line" in node:
                lines.append(node["line"])

        page_info = log_data.get("pageInfo", {})
        has_next = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

        if not edges:
            break

    return lines


def export_single_run(
    api: Any,
    api_key: str,
    entity: str,
    project: str,
    run_id: str,
    output_base_dir: str,
    include_logs: bool = True,
) -> Dict[str, Any]:
    """Export history, config, summary, and logs for a single WandB run."""
    print(f"\n🚀 Exporting Run: {entity}/{project}/{run_id}")
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", run.name or run_id)
    run_dir_name = f"{run_id}_{clean_name}"
    run_dir = os.path.join(output_base_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_dict = dict(run.summary)
    config_dict = dict(run.config)

    # 1. Export Config
    config_yaml_path = os.path.join(run_dir, "config.yaml")
    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    # 2. Export Summary
    summary_json_path = os.path.join(run_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)

    # 3. Export History (non-downsampled via scan_history)
    print("  📊 Pulling full-resolution history metrics...")
    history_records: List[Dict[str, Any]] = []
    all_keys = set()
    for row in run.scan_history():
        history_records.append(row)
        all_keys.update(row.keys())

    # Sort keys for consistent CSV columns
    priority_keys = ["_step", "_runtime", "_timestamp", "rewards/train/solve_ratio", "rewards/train/advantage/mean", "perf/train/global_step_time"]
    remaining_keys = sorted(list(all_keys - set(priority_keys)))
    ordered_keys = [k for k in priority_keys if k in all_keys] + remaining_keys

    # Write CSV
    history_csv_path = os.path.join(run_dir, "history.csv")
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for row in history_records:
            writer.writerow(row)

    # Write JSONL
    history_jsonl_path = os.path.join(run_dir, "history.jsonl")
    with open(history_jsonl_path, "w", encoding="utf-8") as f:
        for row in history_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  ✅ Saved {len(history_records)} history steps -> {history_csv_path}")

    # 4. Export Console Logs
    log_lines_count = 0
    if include_logs:
        print("  📜 Fetching console logs via GraphQL...")
        logs = fetch_console_logs_graphql(api_key, entity, project, run_id)
        log_path = os.path.join(run_dir, "console.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for line in logs:
                f.write(line + "\n")
        log_lines_count = len(logs)
        print(f"  ✅ Saved {log_lines_count} log lines -> {log_path}")

    # Compute key benchmark summary metrics
    solve_ratios = [r.get("rewards/train/solve_ratio") for r in history_records if r.get("rewards/train/solve_ratio") is not None]
    peak_solve = max(solve_ratios) if solve_ratios else summary_dict.get("rewards/train/solve_ratio", 0.0)
    final_solve = solve_ratios[-1] if solve_ratios else summary_dict.get("rewards/train/solve_ratio", 0.0)

    step_times = [r.get("perf/train/global_step_time") for r in history_records if r.get("perf/train/global_step_time") is not None]
    avg_step_time = (sum(step_times) / len(step_times)) if step_times else 0.0

    return {
        "run_id": run_id,
        "name": run.name,
        "state": run.state,
        "total_steps": len(history_records),
        "peak_solve_ratio": f"{float(peak_solve)*100:.2f}%" if peak_solve is not None else "N/A",
        "final_solve_ratio": f"{float(final_solve)*100:.2f}%" if final_solve is not None else "N/A",
        "avg_step_time_sec": f"{avg_step_time:.1f}s" if avg_step_time > 0 else "N/A",
        "runtime_hours": f"{float(run.summary.get('_runtime', 0))/3600:.2f}h",
        "log_lines": log_lines_count,
        "run_dir": run_dir,
    }


def generate_ablation_summary(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate comparative summary CSV and Markdown reports."""
    summary_csv_path = os.path.join(output_dir, "ablation_summary.csv")
    fields = ["run_id", "name", "state", "total_steps", "peak_solve_ratio", "final_solve_ratio", "avg_step_time_sec", "runtime_hours", "log_lines"]

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for res in results:
            writer.writerow({k: res.get(k, "") for k in fields})

    summary_md_path = os.path.join(output_dir, "ablation_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# 📊 WandB Multi-Run Ablation Benchmark Summary\n\n")
        f.write("| Run ID | Experiment Name | State | Steps | Peak Solve | Final Solve | Avg Step Time | Total Runtime | Log Lines |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for res in results:
            f.write(
                f"| `{res['run_id']}` | **{res['name']}** | {res['state']} | {res['total_steps']} | "
                f"**{res['peak_solve_ratio']}** | {res['final_solve_ratio']} | {res['avg_step_time_sec']} | {res['runtime_hours']} | {res['log_lines']} |\n"
            )
        f.write("\n")

    print(f"\n✨ Generated Ablation Summary Table -> {summary_md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete WandB training data & console logs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of WandB URLs, run paths, or run IDs to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="/usr/local/google/home/yuxzhang/yuxuan_dev/tunix_code_rl/wandb_exports",
        help="Directory where exported datasets and logs will be saved.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip downloading console stdout logs.",
    )
    args = parser.parse_args()

    api_key = get_wandb_api_key()
    os.environ["WANDB_API_KEY"] = api_key

    # Import wandb safely
    try:
        import wandb
    except ImportError:
        print("❌ Error: wandb package is not installed in the local Python environment.")
        print("👉 You can run this inside the cluster pod with: kubectl exec -it <pod> -- python3 export_wandb.py ...")
        sys.exit(1)

    api = wandb.Api(api_key=api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for target in args.runs:
        entity, project, run_id = parse_run_target(target)
        try:
            res = export_single_run(
                api=api,
                api_key=api_key,
                entity=entity,
                project=project,
                run_id=run_id,
                output_base_dir=args.output_dir,
                include_logs=not args.no_logs,
            )
            results.append(res)
        except Exception as e:
            print(f"❌ Failed to export {target}: {e}")

    if results:
        generate_ablation_summary(results, args.output_dir)
        print("\n🎉 All requested runs exported successfully!")


if __name__ == "__main__":
    main()
