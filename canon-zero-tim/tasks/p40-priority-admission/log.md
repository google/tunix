# Log

## 2026-08-11 UTC — P40.1: Bind the cross-workload priority contract

- Type: decision
- Fact: Source commit `458dcd2c` added `priorityClassName: very-high` to both head and worker Pod specs in the 64-chip and 256-chip base JobSets, but neither renderer nor its tests reject later drift.
- Action: Bound a separate P40 task for the GSM8K/FrozenLake/DeepSWE scheduling contract; no cluster resource was created or modified.
- Command: `git diff bbf3527a..458dcd2c -- canon-zero-tim/cluster/jobset-64chip.yaml canon-zero-tim/cluster/jobset-256cluster-64chip.yaml`
- Result: Four Pod-level priority fields are present; renderer enforcement remains pending.
- Files/artifacts: `cluster/jobset-64chip.yaml`, `cluster/jobset-256cluster-64chip.yaml`
- Rollback: Remove only the P40 renderer assertions, tests, and runbook text; the already-published base-template priority change is independent.
- Next: Add fail-closed priority assertions to P33 and P34 renderers.

## 2026-08-11 UTC — P40.3: Close the local priority admission gate

- Type: code change
- Fact: P33 renders both GSM8K and FrozenLake from the 64-chip base; P34 renders DeepSWE from the 256-chip base. Both bases already contain `very-high` on head and worker Pods.
- Action: Added renderer assertions that reject missing or mismatched head/worker priority, positive coverage for all three workloads, negative controls for both Pod roles, and read-only PriorityClass preflight instructions in both runbooks.
- Command: `sudo docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v /home/yuxuan/code_rl_repro/sequence_packing/p39_deepswe_production_contract_0810:/workspace:ro -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 bash canon-zero-tim/tests/p33_workloads/run_cpu.sh`
- Command: `bash canon-zero-tim/tests/p34_deepswe/run_static.sh`
- Command: `python3 -m py_compile canon-zero-tim/cluster/render_p33_jobsets.py canon-zero-tim/cluster/render_p34_jobset.py && git diff --check`
- Result: P33 ended `CPU_GATE PASS workloads=2`; P34 ended `P34_STATIC_PASS suites=10`; focused P33 renderer tests passed 11/11 and focused P34 renderer tests passed 10/10; syntax and diff checks passed.
- Files/artifacts: `cluster/render_p33_jobsets.py`, `cluster/render_p34_jobset.py`, `tests/p33_workloads/test_render_p33_jobsets.py`, `tests/p34_deepswe/test_render_p34_jobset.py`, `cluster/P33_QUEUE.md`, `cluster/P34_DEEPSWE_RUNBOOK.md`
- Rollback: Revert only the two `_PRIORITY_CLASS` validation blocks, their tests, and the two runbook preflight sections. Do not alter the published base templates or any workload numerical policy.
- Next: Review the diff; commit/push only after explicit approval. Before any target launch, run the documented `kubectl get priorityclass very-high` check in that cluster.
