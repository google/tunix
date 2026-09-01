# P58.33 — K24 precomputed-gradient checkpoint admission

Status: `LOCAL CONSTRUCTION PASS / TARGET NOT RUN`

K25 later emitted the required disabled-checkpoint marker and crossed this
phase boundary. Its subsequent precheck-only environment admission failure is
tracked in P58.34; K25 still made zero optimizer commits and is not resumable.

## Incident

K24 `canon-p58-ds4b-zero-hp-full-k24` ran on 128 TPU v5p devices as
rollout DP8xTP8 plus trainer DP8xTP8. It returned all 128 trajectories, solved
6 (4.6875%), retained 46 rows with nonzero final advantage, and passed strict
A=B=C over 388,328 action tokens. All 16 forward groups and the complete
36-layer Pallas VJP ran. Reverse group 1/16 completed with exact replicas and
a finite nonzero gradient.

The run then stopped before accumulator mutation and optimizer commit:

```text
ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted
```

This was not checkpoint I/O. The P58 renderer inherited P34's checkpoint
directory and save cadence, while the precomputed-gradient checkpoint schema
is admitted only for P45 FrozenLake. K24 made zero optimizer commits and has
no resumable trainer checkpoint.

Immutable evidence:
`canon-zero-tim/evidence/p58_k24_precomputed_checkpoint_contract_incident/`.

## Repair and admission

Every P58 arm and stage now uses exactly:

```text
--ckpt_dir=none
```

`--save_interval_steps` and `--max_to_keep` are absent. Checkpoint mode is a
shared paired-recipe field, not a treatment. The renderer, `00_env.sh`, and
the Python CLI contract reject any other value. The Python entrypoint must
emit this receipt before model initialization:

```text
[P58.CHECKPOINT] PASS mode=disabled cli=none resume=unsupported
```

The downside is explicit: no P58 trainer/optimizer checkpoint exists, and a
stopped 1,000-update campaign cannot resume its trainer state. Durable
trajectory journals, alignment/update reports, W&B, and debug artifacts are
unchanged and remain available for diagnosis.

## Alignment and safety boundary

P58.32 remains active for the exact Zero-HP/full production identity:
`CANON_DEEPSWE_ALIGNMENT_WARN_ONLY=1` is renderer-derived and admits only a
finite decode-vs-prefill A-B difference plus its direct ratio/clip/TIS
consequences. B-C, T_old-current, nonfinite, shape, gradient, replica,
transaction, optimizer, OOM, and evidence failures remain fatal. This is
`convergence-only / alignment-degraded`, never a Zero-TIM certification.

## Historical next-target instruction

This instruction was consumed by K25. K25 used a fresh run ID and crossed the
checkpoint boundary before failing in P58.34 scope. Before model
initialization, require the checkpoint-disabled receipt. Then require all 16
reverse-group receipts, finite nonzero gradients, exact replicas, and the
first valid TPU-resident optimizer transaction. No checkpoint receipt is
expected. K24 cannot be resumed. Image publication and Kubernetes launch
remain separately approval-gated.

## Construction evidence

- focused renderer/script/postflight contracts: 57/57 PASS;
- P34 static: `P34_STATIC_PASS suites=10`;
- Python compilation, Bash syntax, and diff hygiene: PASS;
- flag registry: `declared=409 actual=409 unique=409`, PASS;
- complete digest-pinned image gate:
  `P58_EXACT_IMAGE_CPU_PASS ... alignment_policy=1 zero_hp_full=1
  checked_vma_diagnostic=1 coarse_seam=1 ...`.

These are construction receipts only. No TPU target, optimizer commit, image
publication, or Kubernetes mutation occurred.
