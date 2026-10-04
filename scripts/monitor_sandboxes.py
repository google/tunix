#!/usr/bin/env python3
"""
GKE Cluster & Sandbox Monitor for Tunix / Trellis Workloads.

Monitors active tr-* jobs in Kubernetes namespace (default: priority-dev-scheduled),
tracking sandbox usage (claims, warmpools, running pods) and total cluster
sandbox capacity (nodes, allocatable CPU/RAM, max pods).

Usage:
    # Run continuous monitor updating every 60 seconds (default):
    python3 scripts/monitor_sandboxes.py

    # Run once (single snapshot):
    python3 scripts/monitor_sandboxes.py --once

    # Custom interval and namespace:
    python3 scripts/monitor_sandboxes.py --interval 30 --namespace priority-dev-scheduled
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

# ANSI styling
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor GKE jobs and sandbox capacity for Tunix / Trellis runs."
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=int(os.environ.get("INTERVAL", "60")),
        help="Refresh interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--namespace", "-n",
        type=str,
        default=os.environ.get("NAMESPACE", "priority-dev-scheduled"),
        help="Kubernetes namespace (default: priority-dev-scheduled)",
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=os.environ.get("JOB_FILTER", "tr-"),
        help="Job prefix filter (default: tr-)",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=os.environ.get("CLUSTER", "bodaborg-tpu7x-gsc-elm"),
        help="Cluster name (default: bodaborg-tpu7x-gsc-elm)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("REGION", "us-east1"),
        help="GCP region (default: us-east1)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit without looping",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON data instead of formatted table",
    )
    return parser.parse_args()


def fetch_cluster_data(namespace: str, job_filter: str):
    """Executes parallel kubectl queries to minimize total query latency."""
    # 1. Active SandboxClaims with created-by label
    p_claims = subprocess.Popen(
        ["kubectl", "get", "sandboxclaims", "-n", namespace,
         "-L", "app.kubernetes.io/created-by", "--no-headers"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 2. SandboxWarmPools with created-by label
    p_pools = subprocess.Popen(
        ["kubectl", "get", "sandboxwarmpools", "-n", namespace,
         "-L", "app.kubernetes.io/created-by", "--no-headers"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 3. Workload and Sandbox Pods in namespace
    p_pods = subprocess.Popen(
        ["kubectl", "get", "pods", "-n", namespace, "--no-headers", "-o", "wide"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 4. Allocatable resources on sandbox-c3d-np nodepool
    p_nodes = subprocess.Popen(
        ["kubectl", "get", "nodes", "-l", "cloud.google.com/gke-nodepool=sandbox-c3d-np",
         "-o", "jsonpath={.items[*].status.allocatable}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 5. SandboxTemplate resource requests and limits
    p_tmpl = subprocess.Popen(
        ["kubectl", "get", "sandboxtemplates", "-n", namespace,
         "-o", "jsonpath={.items[0].spec.podTemplate.spec.containers[0].resources}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 6. Overall sandbox nodepool node counts across cluster
    p_all_pools = subprocess.Popen(
        ["kubectl", "get", "nodes", "-L", "cloud.google.com/gke-nodepool", "--no-headers"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    out_claims, _ = p_claims.communicate()
    out_pools, _ = p_pools.communicate()
    out_pods, _ = p_pods.communicate()
    out_nodes, _ = p_nodes.communicate()
    out_tmpl, _ = p_tmpl.communicate()
    out_all_pools, _ = p_all_pools.communicate()

    return {
        "claims_raw": out_claims,
        "pools_raw": out_pools,
        "pods_raw": out_pods,
        "nodes_raw": out_nodes,
        "tmpl_raw": out_tmpl,
        "all_pools_raw": out_all_pools,
    }


def parse_metrics(data: dict, namespace: str, job_filter: str):
    # 1. Parse SandboxClaims
    claims_by_job = Counter()
    total_claims = 0
    for line in data["claims_raw"].splitlines():
        parts = line.split()
        if not parts:
            continue
        total_claims += 1
        creator = parts[-1]
        if creator.startswith(job_filter):
            claims_by_job[creator] += 1

    # 2. Parse SandboxWarmPools
    pools_by_job = Counter()
    total_pools = 0
    for line in data["pools_raw"].splitlines():
        parts = line.split()
        if not parts:
            continue
        total_pools += 1
        creator = parts[-1]
        if creator.startswith(job_filter):
            pools_by_job[creator] += 1

    # 3. Parse Pods
    job_workload_pods = defaultdict(lambda: defaultdict(Counter))
    sb_pods_by_creator = defaultdict(Counter)
    nodepool_pods_status = Counter()

    for line in data["pods_raw"].splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pod_name = parts[0]
        status = parts[2]
        node = parts[6] if len(parts) > 6 else ""

        if "sandbox-c3d-np" in node:
            nodepool_pods_status[status] += 1

        # Check if sandbox pod
        if pod_name.startswith("pool-oh-") or pod_name.startswith("oh-"):
            m = re.match(r"^pool-oh-([a-zA-Z0-9_\-]+?)-[0-9a-f]{12}-[a-z0-9]+$", pod_name)
            creator_base = m.group(1) if m else "other"
            creator_orch = f"{creator_base}-orch"
            if creator_base.startswith(job_filter):
                sb_pods_by_creator[creator_orch][status] += 1
        elif pod_name.startswith(job_filter):
            # Rollout, Train, or Orch pod
            m = re.match(r"^([a-zA-Z0-9_\-]+?)-(orch|roll|train)", pod_name)
            if m:
                base, role = m.group(1), m.group(2)
                job_workload_pods[base][role][status] += 1

    # 4. Parse Node Allocatable
    node_allocs = re.findall(r"\{.*?\}", data["nodes_raw"])
    num_c3d_nodes = len(node_allocs)
    alloc_json = json.loads(node_allocs[0]) if node_allocs else {}
    cpu_node = alloc_json.get("cpu", "59380m")
    mem_node = alloc_json.get("memory", "477357992Ki")
    pods_per_node = int(alloc_json.get("pods", "128"))

    # Convert memory to GiB
    mem_node_gib = 0
    if mem_node.endswith("Ki"):
        mem_node_gib = int(mem_node[:-2]) / (1024 * 1024)

    # Convert CPU to cores
    cpu_node_cores = 0.0
    if cpu_node.endswith("m"):
        cpu_node_cores = int(cpu_node[:-1]) / 1000.0

    total_pod_capacity = num_c3d_nodes * pods_per_node
    # System DaemonSets take ~9 pods per node (calico, gmp, fluentbit, gke agents)
    system_pods_per_node = 9
    net_usable_pod_capacity = num_c3d_nodes * (pods_per_node - system_pods_per_node)

    # 5. Parse SandboxTemplate Resources
    tmpl_res = json.loads(data["tmpl_raw"]) if data["tmpl_raw"] else {}
    req_cpu = tmpl_res.get("requests", {}).get("cpu", "500m")
    req_mem = tmpl_res.get("requests", {}).get("memory", "1Gi")
    lim_cpu = tmpl_res.get("limits", {}).get("cpu", "2")
    lim_mem = tmpl_res.get("limits", {}).get("memory", "4Gi")

    # 6. Parse all sandbox nodepools across cluster
    cluster_nodepool_counts = Counter()
    for line in data["all_pools_raw"].splitlines():
        parts = line.split()
        if len(parts) >= 6:
            np = parts[5]
            if "sandbox" in np:
                cluster_nodepool_counts[np] += 1

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_claims": total_claims,
        "total_pools": total_pools,
        "claims_by_job": claims_by_job,
        "pools_by_job": pools_by_job,
        "job_workload_pods": job_workload_pods,
        "sb_pods_by_creator": sb_pods_by_creator,
        "nodepool_pods_status": nodepool_pods_status,
        "num_c3d_nodes": num_c3d_nodes,
        "cpu_node_cores": cpu_node_cores,
        "mem_node_gib": mem_node_gib,
        "pods_per_node": pods_per_node,
        "total_pod_capacity": total_pod_capacity,
        "net_usable_pod_capacity": net_usable_pod_capacity,
        "req_cpu": req_cpu,
        "req_mem": req_mem,
        "lim_cpu": lim_cpu,
        "lim_mem": lim_mem,
        "cluster_nodepool_counts": cluster_nodepool_counts,
    }


def print_dashboard(metrics: dict, cluster: str, region: str, namespace: str, job_filter: str):
    ts = metrics["timestamp_utc"]
    print(f"\n{BOLD}{'=' * 88}{RESET}")
    print(f"{BOLD}SANDBOX & JOB MONITOR{RESET} | {CYAN}{ts}{RESET}")
    print(f"Cluster: {BOLD}{cluster}{RESET} ({region}) | Namespace: {BOLD}{namespace}{RESET} | Filter: {BOLD}{job_filter}*{RESET}")
    print(f"{'=' * 88}")

    # Section 1: Active Workload Jobs
    print(f"\n{BOLD}{YELLOW}1. ACTIVE JOBS ({job_filter}*){RESET}")
    if not metrics["job_workload_pods"]:
        print("  No active workload pods matching filter.")
    else:
        for base, roles in sorted(metrics["job_workload_pods"].items()):
            print(f"  • Job Run: {BOLD}{base}{RESET}")
            for role in ("orch", "train", "roll"):
                if role in roles:
                    st_counts = ", ".join(f"{st}: {cnt}" for st, cnt in sorted(roles[role].items()))
                    role_label = {
                        "orch": "Orchestrator",
                        "train": "Trainer Replicas",
                        "roll": "Rollout Workers",
                    }.get(role, role)
                    print(f"    - {role_label:18s}: {st_counts}")

    # Section 2: Sandbox Usage per Job
    print(f"\n{BOLD}{YELLOW}2. SANDBOX USAGE PER JOB{RESET}")
    all_creators = sorted(set(
        list(metrics["claims_by_job"].keys()) +
        list(metrics["pools_by_job"].keys()) +
        list(metrics["sb_pods_by_creator"].keys())
    ))

    if not all_creators:
        print("  No active sandboxes found for filter.")
    else:
        for creator in all_creators:
            claims = metrics["claims_by_job"].get(creator, 0)
            pools = metrics["pools_by_job"].get(creator, 0)
            target_standby = pools * 16  # MAX_WARMPOOL_REPLICAS = 16
            pod_statuses = metrics["sb_pods_by_creator"].get(creator, {})
            running_pods = pod_statuses.get("Running", 0)
            error_pods = pod_statuses.get("Error", 0)
            term_pods = pod_statuses.get("Terminating", 0)
            total_pods = sum(pod_statuses.values())

            # Recipe cap: MAX_CONCURRENCY = 4096
            claims_pct = (claims / 4096.0) * 100.0 if 4096 else 0.0

            print(f"  • Creator: {BOLD}{creator}{RESET}")
            print(f"    - In-flight Claims (Episodes): {GREEN}{claims:,}{RESET} / 4,096 max concurrency ({claims_pct:.1f}% of claim cap)")
            print(f"    - Sandbox WarmPools:          {pools:,} pools (Target Standby: {target_standby:,} @ 16/pool)")
            print(f"    - Sandbox Pods by Status:     {GREEN}Running: {running_pods:,}{RESET} | {YELLOW}Terminating: {term_pods:,}{RESET} | {RED}Error: {error_pods:,}{RESET} (Total: {total_pods:,})")
            if running_pods > 0:
                standby_est = max(0, running_pods - claims)
                print(f"    - Active vs Standby:          Active in-flight: ~{claims:,} | Warm standby: ~{standby_est:,}")

    # Section 3: Pods Scheduled on sandbox-c3d-np
    print(f"\n{BOLD}{YELLOW}3. PODS ON sandbox-c3d-np NODEPOOL{RESET}")
    st_str = ", ".join(f"{st}: {cnt:,}" for st, cnt in sorted(metrics["nodepool_pods_status"].items()))
    print(f"  • Namespace {namespace}: {st_str}")
    print(f"  • System DaemonSets (kube-system + gmp-system): ~9 pods/node (~4,986 pods across 554 nodes)")

    # Section 4: Cluster Sandbox Capacity
    print(f"\n{BOLD}{YELLOW}4. CLUSTER SANDBOX CAPACITY (sandbox-c3d-np){RESET}")
    nodes = metrics["num_c3d_nodes"]
    cores = metrics["cpu_node_cores"]
    mem = metrics["mem_node_gib"]
    max_pods = metrics["pods_per_node"]
    tot_cap = metrics["total_pod_capacity"]
    net_cap = metrics["net_usable_pod_capacity"]
    running_on_np = metrics["nodepool_pods_status"].get("Running", 0)
    util_pct = (running_on_np / net_cap) * 100.0 if net_cap else 0.0

    print(f"  • Dedicated Node Pool:       {BOLD}sandbox-c3d-np{RESET} ({nodes} nodes of type {BOLD}c3d-highmem-60-lssd{RESET})")
    print(f"  • Per-Node Allocatable:      CPU = {cores:.1f} cores ({cores*1000:.0f}m) | Mem = {mem:.1f} GiB | Pod Limit = {max_pods}")
    print(f"  • Per-Sandbox Resources:     Requests = (CPU {metrics['req_cpu']}, Mem {metrics['req_mem']}) | Limits = (CPU {metrics['lim_cpu']}, Mem {metrics['lim_mem']})")
    print(f"  • Max Pod Capacity (Gross):  {tot_cap:,} pods ({nodes} nodes × {max_pods} maxPods)")
    print(f"  • Usable Sandbox Capacity:   ~{net_cap:,} sandboxes (accounting for ~9 DaemonSets/node)")
    print(f"  • Current Running Sandboxes: {running_on_np:,}")
    print(f"  • Cluster Utilization:       {CYAN}{util_pct:.1f}%{RESET} of net sandbox capacity ({net_cap - running_on_np:,} available headroom)")

    # Cluster-wide sandbox nodepools summary
    all_pools = metrics["cluster_nodepool_counts"]
    if len(all_pools) > 1:
        pools_summary = ", ".join(f"{p}: {c} nodes" for p, c in sorted(all_pools.items()))
        print(f"  • All Sandbox Pools:         {pools_summary}")

    print(f"{BOLD}{'=' * 88}{RESET}\n")


def main():
    args = parse_args()

    if args.once:
        data = fetch_cluster_data(args.namespace, args.filter)
        metrics = parse_metrics(data, args.namespace, args.filter)
        if args.json:
            # Clean non-serializable counters
            metrics["claims_by_job"] = dict(metrics["claims_by_job"])
            metrics["pools_by_job"] = dict(metrics["pools_by_job"])
            metrics["job_workload_pods"] = {k: {r: dict(v) for r, v in roles.items()} for k, roles in metrics["job_workload_pods"].items()}
            metrics["sb_pods_by_creator"] = {k: dict(v) for k, v in metrics["sb_pods_by_creator"].items()}
            metrics["nodepool_pods_status"] = dict(metrics["nodepool_pods_status"])
            metrics["cluster_nodepool_counts"] = dict(metrics["cluster_nodepool_counts"])
            print(json.dumps(metrics, indent=2))
        else:
            print_dashboard(metrics, args.cluster, args.region, args.namespace, args.filter)
        return

    print(f"Starting sandbox monitor for cluster '{args.cluster}' in namespace '{args.namespace}' (refreshing every {args.interval}s, press Ctrl+C to stop)...")
    try:
        while True:
            t0 = time.time()
            data = fetch_cluster_data(args.namespace, args.filter)
            metrics = parse_metrics(data, args.namespace, args.filter)
            if args.json:
                print(json.dumps(metrics))
            else:
                print_dashboard(metrics, args.cluster, args.region, args.namespace, args.filter)
            sys.stdout.flush()
            elapsed = time.time() - t0
            sleep_time = max(1.0, args.interval - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
