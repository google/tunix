# P58.37 — K29 profiler-free production full training

Status: `LOCAL CONSTRUCTION PASS / K30 TARGET NOT RUN`

## Incident boundary

K29 proved P58.36 on the real 128-device target. Step 1 returned all 128
trajectories across eight prompt groups, completed Rescore B, and passed
pre-alignment with exact A=B=C over 412,449 action tokens. The process then
failed before any Step-1 backward work when the update-entry XProf hook passed
the head-local path
`/mnt/disks/linchai_data/.../xprof-update` to Pathways. Pathways requires a
`gs://` log directory and raised `ValueError`.

This is an observer admission failure, not a rollout, alignment, backward,
optimizer, or checkpoint failure. The immutable incident package remains at
`canon-zero-tim/evidence/p58_k29_xprof_gcs_path_incident/`. Its proposal to
move production XProf to GCS is historical analysis; P58.37 records the later
operator decision to remove profiling from the production training carrier.

## Production contract

The 1,000-update P58 Zero-HP full workload is profiler-free:

- all `CANON_XPROF_*` and `CANON_PERF_TRACE_*` variables are absent, not set
  to zero or an empty path;
- no update-entry XProf hook, XProf label wrapper, Perfetto exporter, GCS
  restore, XPlane, trace JSON, or Perfetto artifact is admitted;
- resolved-environment, Python startup, and postflight classifier boundaries
  all reject profiler reinjection;
- the shared runner skips XProf restore only for the exact P58 high-performance
  full identity. Other V1 high-performance workloads remain fail-closed if
  their required XProf path disappears.

The production numerical and training bundle is unchanged: P59 rank-parallel
backward, P71 scan, fixed-head, TiTO, B8xG16, 16K response, DP8xTP8 rollout and
DP8xTP8 trainer, TPU-resident AdamW, 1,000 updates, disabled checkpointing, and
the existing finite A-B warning policy all remain active.

## Diagnostic separation

P58.37 does not delete XProf support. The independent one-host XProf and P59
diagnostic carriers retain their own explicit paths, workload identities, and
capture gates. A trace request must use one of those carriers; it must not be
smuggled into the long production run.

## K30 gate

K29 cannot resume because P58 checkpoints are disabled. A future K30 must be
rendered fresh from the final clean remote-read source SHA and a matching
digest-pinned image after separate publication and launch approval. Before
application, require that the rendered environment contains none of:

```text
CANON_XPROF_DIR CANON_XPROF_PHASE CANON_XPROF_SKIP_STEPS CANON_XPROF_STEPS
CANON_XPROF_PYTHON_TRACER CANON_XPROF_HOST_TRACER
CANON_XPROF_TPU_TRACE_MODE CANON_XPROF_LABELS
CANON_PERF_TRACE_DIR CANON_PERF_TRACE_EXPORT_STEP
```

K30 must reproduce the complete Step-1 128-row batch, cross update entry
without a profiler call, complete all sixteen reverse groups, pass the first
update/optimizer transaction and outer synchronization gates, then continue
into Step 2. Absence of profiling alone is not training success.

## Construction evidence

- focused profile contract: 12/12 PASS;
- profiler-free full classifier: 8/8 PASS;
- resolved-environment and exact runner behavior: PASS inside the pinned
  dependency image;
- complete digest-pinned gate: `P58_EXACT_IMAGE_CPU_PASS`;
- P34 static: `P34_STATIC_PASS suites=10`;
- flag registry: `FLAG_AUDIT_PASS`, 409 declared/actual/unique and no new
  flag name;
- Python/Bash syntax and `git diff --check`: PASS.

No Kubernetes workload, TPU target, image publication, commit, or push was
performed for this repair. Construction evidence cannot prove K30.
