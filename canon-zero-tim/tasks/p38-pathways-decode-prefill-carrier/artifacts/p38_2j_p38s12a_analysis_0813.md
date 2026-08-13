# P38s12a analysis-level receipt

- Evidence publication: `23bb2a3c`.
- Runtime source: `e4d442bcc654938b5fcf437d901f6691265cb050`.
- Scientific arm: known-red stock, DP16xTP4, 32 prompts x 8 generations,
  `max_concurrency=256`; the submitted `p38s12b` label is not the arm identity.
- Core integrity: capsule and serving archive re-extracted from `head.full.log`
  byte-for-byte; classifier rerun matched `serving-classification.json`.
- Numerical result: A-B 46/44,818 elements and 74/179,272 bytes red,
  `max_abs=0.1039161682`; B-C 0 bytes.
- Capture result: four pre/post records, 136 request-journal records, all eight
  selected rows joined; nine source rows were red and row 255 was omitted by
  the old cap of eight.
- Admission boundary: expected pre-backward evidence exists, but outer
  postflight rejected `rc=137`; full Kubernetes/Pathways artifacts are absent;
  the checksum manifest's self-entry is stale. Verdict:
  `CORE_EVIDENCE_COMPLETE / RUN_ADMISSION_INCOMPLETE`.
- Next selected row: source row 231, capsule index 3.
