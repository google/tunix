# threads/ — per-thread run directories (target layout, first use forward)

New runs land here, one immutable directory per launch, packaged by
`../scripts/package_run.sh` and checked by `../scripts/check_run_dir.sh`:

```
threads/<thread>/runs/<run-id>/
```

Thread names match `../THREADS.md`: `zero-tim-carrier`, `perf`, `frozenlake-train`,
`deepswe-eval`, `deepswe-train`, `delivery-docs`.

Rules:
- every launch gets a directory here, bootstrap failures included (`verdict.json`
  = INCONCLUSIVE); run directories are write-once;
- legacy evidence stays where it is (`tasks/*/evidence/`, `debug_logs/`) and is
  indexed by `../EVIDENCE.md` — do not migrate old runs;
- a new run is registered in `../EVIDENCE.md` in the same CL that adds it.
