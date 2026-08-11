# State

- Status: complete
- Objective: Require `priorityClassName: very-high` on every rendered GSM8K, FrozenLake, and DeepSWE Pathways head and worker Pod.
- Definition of done: P33 and P34 renderers reject a missing or mismatched priority class; focused renderer tests and the P33/P34 static suites pass; both operator runbooks contain a read-only cluster preflight.
- Task directory: `canon-zero-tim/tasks/p40-priority-admission`
- Directory state: tracked
- Current phase: P40.3 passed
- Last verified fact: P33 frozen-image CPU gate ended `CPU_GATE PASS workloads=2`; P34 static gate ended `P34_STATIC_PASS suites=10`; focused priority positive/negative tests passed.
- Next action: Obtain explicit approval before committing or pushing the P40 change.
- Blockers: none
- Key artifacts: `cluster/render_p33_jobsets.py`, `cluster/render_p34_jobset.py`, `cluster/P33_QUEUE.md`, `cluster/P34_DEEPSWE_RUNBOOK.md`
- Updated: 2026-08-11 UTC
