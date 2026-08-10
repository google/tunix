# Log

## 2026-08-10 UTC — P38.1: bind the decode-prefill carrier investigation

- Type: decision
- Fact: r35 proved B-C bitwise on both production workloads but stopped before backward because
  A-B differed by 2 bytes on GSM8K and 70 bytes on FrozenLake. The FrozenLake sampler diagnostic
  measured `logp_diff_max=0.10390`; the sparse boundary cannot be treated as a one-ULP fact.
- Hypothesis: The remaining carrier is in the Pathways serving decode-versus-prefill envelope;
  proxy-flag causality and the first numerical divergence site remain unverified.
- Action: Bound P38 and started evidence hardening without changing loss, precision, optimizer,
  or hard-gate semantics.
- Command: omitted
- Result: P38.1 active; no new TPU numerical result exists.
- Files/artifacts: `state.md`, `plan.md`, `phases/p38-1-evidence-hardening.md`
- Rollback: Omit the isolated P38 task records and evidence-only code changes; existing r35 raw
  artifacts and P36/P37 work remain untouched.
- Next: Run local alignment and runner negative controls after implementation.

## 2026-08-10 UTC — P38.1: evidence hardening passed locally

- Type: implementation and verification
- Fact: a pre-backward mismatch now records its original coordinate, token id, exact scalar bits,
  XOR, differing byte offsets, ULP distance, numerical delta, and maximum-absolute mismatch. At
  most 1024 records per boundary are emitted and truncation is explicit.
- Fact: the JSONL is flushed and fsynced before a hard-gate exception. The complete strict-JSON
  record and report SHA are printed to stdout, and `90_run.sh` repeats the persisted record after
  a nonzero workload exit.
- Fact: nonfinite values are encoded explicitly as `nan`, `inf`, or `-inf`; they cannot crash the
  evidence serializer after the numerical gate has already found a red boundary.
- Action: Corrected r35 byte-versus-token wording and added one-ULP, signed-zero, high-amplitude,
  nonfinite, invalid-shape, bounded-output, and failed-workload controls. Precision, loss,
  sampling, gradient, optimizer, and hard-gate behavior were not changed.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'python3 -m unittest discover -s tests/rl -p alignment_test.py'`
- Result: PASS, 18 tests.
- Command: `sudo docker run --rm --entrypoint bash -e PYTHONDONTWRITEBYTECODE=1 -e JAX_PLATFORMS=cpu -v "$PWD:/workspace:ro" -w /workspace tunix_frozenlake_image:vllm-tpu0.25.0 -lc 'bash canon-zero-tim/tests/p33_workloads/run_cpu.sh'`
- Result: PASS. The final suite completed 63 + 18 + 10 + 14 + 31 + 2 + 1 registered tests and printed
  `[P38.EVIDENCE] FAILED_REPORT_STDOUT_PASS` and `[P33.WORKLOAD] CPU_GATE PASS`.
- Command: `git diff --check && bash -n canon-zero-tim/cluster/steps/90_run.sh canon-zero-tim/tests/p33_workloads/run_cpu.sh && python3 -m py_compile tunix/rl/alignment.py tests/rl/alignment_test.py`
- Result: PASS.
- Files/artifacts: `tunix/rl/alignment.py`, `tests/rl/alignment_test.py`,
  `../../tests/p33_workloads/run_cpu.sh`, `../../cluster/steps/90_run.sh`,
  `../../debug_logs/README.md`, `HANDOFF.md`
- Rollback: Revert only the P38 evidence fields, stdout artifact block, tests, and P38 records. The
  original r35 raw logs and all unrelated dirty P36/P37 files remain untouched.
- Next: Publish only after explicit approval, then run one strict Attempt-0 GSM8K
  `alignment-short` reproduction. Do not queue full training.

## 2026-08-10 UTC — P38.2 preparation: GSM8K-first no-commit diagnostic

- Type: decision and implementation
- Fact: the existing `gsm8k-full` manifest was not a safe diagnostic substitute. If its sparse
  A-B carrier were not sampled, the manifest could continue into committing training.
- Action: Added a dedicated `gsm8k-alignment-short` JobSet. It preserves the signed GSM8K shape
  (`32` prompts, `8` generations, response limit `1024`, local M256), sets `max_steps=1`, and
  requires `CANON_P33_NO_COMMIT=1`.
- Result: The full P33 CPU gate passed with five isolated strict JobSets. Renderer parity checks
  prove the generated command is byte-for-byte the frozen `dp_workloads` command.
- Rollback: Remove only the GSM8K alignment-short spec and its tests. Existing full workload
  recipes are unchanged.
- Next: Run GSM8K alignment-short first. If it is exact, treat it as a non-reproduction and run
  FrozenLake alignment-short; do not call it a fix.
