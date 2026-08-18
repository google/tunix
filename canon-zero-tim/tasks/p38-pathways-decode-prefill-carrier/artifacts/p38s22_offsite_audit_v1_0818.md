# P38s22 offsite audit v1 receipt review

- Evidence commit: `0b86ef5caeb80d1c7a9d4a3e7cdec308dc3ed14a`.
- Analysis tool commit: `180fc2ff048c2225392581f5adef9539ac98202b`.
- Return integrity: all 10 entries in the returned `SHA256SUMS` verify.
- Tool/contract identity: auditor, archive helper, wrapper, and contract SHAs
  match the published analysis source.
- Verdict: `INCONCLUSIVE / OFFSITE_EVIDENCE_AUDIT_FAILED`.
- Direct failure: root `SHA256SUMS` was unavailable. Root `COLLECTED.json` and
  `COMPLETE.json` were also not returned.
- Preserved round receipts: 3/3 completion markers are
  `sealed-and-verified`; 3/3 manifest SHAs and logical count 10 match their
  marker. Every manifest is sorted and names the preregistered capsule SHA.
- Limitation: the v1 root-first auditor exited before verifying the actual
  round tar bytes or recomputing the numerical totals. The returned package is
  a valid sealed failure receipt, not a signed P38s22 PASS.
- Decision: do not relaunch P38s22. P38.2w2 audits the three independent round
  archives first and reports missing root postflight as a separate claim.
