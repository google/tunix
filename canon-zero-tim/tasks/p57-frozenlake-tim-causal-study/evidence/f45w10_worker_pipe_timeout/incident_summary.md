# Incident Summary: FrozenLake P45 Full Wave 10 Step-63 Worker-to-Worker Pipe Timeout

- **JobSet**: `canon-p57-fl-zero-f45w10-96544812`
- **Workload**: FrozenLake P45 Full Wave 10 (Qwen3-8B, 64 TPU v5p, DP8xTP8)
- **Failure Timestamp**: 2026-08-28T03:32:51Z
- **Progress Reached**: Step 63 / 300 (Solve Rate: 44.5%, Step duration ~2.9m)
- **Failure Mode**: `DEADLINE_EXCEEDED_WORKER_PIPE_TIMEOUT`

---

## 1. Error Trace & Log Excerpts

Reported by `worker-0-13` (`gke-tpu-3a97861b-1vp4`) and `worker-0-1` (`gke-tpu-3a97861b-rf71`):

```text
W0828 03:32:48.947930 1488 pipe_endpoint.cc:834] Client pipe canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-2.canon-p57-fl-zero-f45w10-96544812:29000/worker_to_worker id=17973490401862667062: saw idle time 5.43636304s where timeout is 10s
W0828 03:32:49.948070 1506 pipe_endpoint.cc:834] Client pipe canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-2.canon-p57-fl-zero-f45w10-96544812:29000/worker_to_worker id=17973490401862667062: saw idle time 6.436503863s where timeout is 10s
W0828 03:32:50.948218 1488 pipe_endpoint.cc:834] Client pipe canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-2.canon-p57-fl-zero-f45w10-96544812:29000/worker_to_worker id=17973490401862667062: saw idle time 7.436651489s where timeout is 10s
I0828 03:32:51.916048  816 worker_message_distributor.cc:2308] Waiting for all peers to exit distributor_6408541556418313203_1
E0828 03:32:51.916271  816 quick_restart.cc:19] Job cancelled due to 6408541556418313203 gke-tpu-3a97861b-1vp4 exiting [reported by ServerConfig{ job_info=JobInfo{ name=6408541556418313203 job_instance_id=6408541556418313203 task_index=13 num_tasks=16 addr=canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-13.canon-p57-fl-zero-f45w10-96544812:29000 [hostname='gke-tpu-3a97861b-1vp4']} }] with error INTERNAL: Client pipe canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-2.canon-p57-fl-zero-f45w10-96544812:29000/worker_to_worker id=14969354806911437370 broke with error: DEADLINE_EXCEEDED: lost connection to peer at http://machine/gke-tpu-3a97861b-qbx9/events#srcs=borg%2Bcoroner since 10.047667564s ago; this usually means that the peer has unexpectedly gone away (peer stopped sending messages since 2026-08-28T03:32:41.865463105+00:00; timeout is 10s and the current machine is http://machine/gke-tpu-3a97861b-1vp4/events#srcs=borg%2Bcoroner)
	Client pipe canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-2.canon-p57-fl-zero-f45w10-96544812:29000/worker_to_worker id=14969354806911437370
=== Source Location Trace: === 
third_party/pathways/data_parallel/pipe.cc:307

=== Source Location Trace: ===
third_party/pathways/data_parallel/worker_message_distributor.cc:1842
F0828 03:32:51.916364  816 quick_restart.cc:40] (Quick-restart requested but unsupported in Cloud/OSS.)
```

---

## 2. Root Cause Analysis

1. **Worker-to-Worker Communication Interruption**:
   - During gradient accumulation in Step 63, Pathways worker 13 (`gke-tpu-3a97861b-1vp4`) observed `saw idle time > 10s` across its direct client pipe to worker 2 (`gke-tpu-3a97861b-qbx9:29000`).
   - When the deadline was exceeded (`DEADLINE_EXCEEDED: lost connection to peer`), Pathways data parallel message distributor initiated fail-closed teardown.
2. **Cloud/OSS Quick Restart Unsupported**:
   - Pathways quick-restart is unsupported in Cloud/OSS environments (`quick_restart.cc:40`), causing the worker process to abort (exit code 1) and transitioning the JobSet into `Failed`.
3. **Physical Node Status**:
   - Physical inspection of node `gke-tpu-3a97861b-qbx9` shows it returned to `Ready` without hardware faults or memory pressure.

---

## 3. Log Artifacts in this Directory

- `worker_0_1_error.log`: Full stdout/stderr from worker 1 (`gke-tpu-3a97861b-rf71`)
- `worker_0_13_error.log`: Full stdout/stderr from worker 13 (`gke-tpu-3a97861b-1vp4`)
- `canon-p57-fl-zero-f45w10-96544812-pathways-worker-0-*.log`: Complete logs across all 14 non-terminated worker pods (`worker-0-1`, `worker-0-3` through `worker-0-15`).
