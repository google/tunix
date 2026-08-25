# P60-2F — evidence-ledger finalization

## Status

Latest-tip local and pinned exact-image gates pass on an uncommitted additive
tree. No target rerun is authorized by this phase.

## Trigger

The clean-SHA Zero-HP run
`v1_zero-hp_p60_readable_zero_p60_2e_clean_20260825_r1` completed its runtime
and classifier gates, but exposed a packaging-order defect. The old runner
generated `SHA256SUMS` and only afterward appended its terminal GREEN marker
to `driver.log`. Consequently the manifest recorded driver SHA-256
`f025bf90b76d668b37311ab420571e90ffaca3b2f10248bf082aaa2b259cd8d7`,
while the immutable final file hashes to
`d946011e6fa704f2db20fa75e170b20373296420335c33c91d043c1470769a8b`.
The hash of `driver.log` with its last line removed equals the recorded value,
which mechanically localizes the defect.

The run therefore has the claim ceiling `CORE TARGET GATES PASS / EVIDENCE
PACKAGING RED`; its terminal GREEN line alone is not an acceptance receipt.
The preserved run root must not be repaired in place.

## Contract

The common runner now uses one sourced finalization helper and must execute
this order exactly:

1. determine the single terminal GREEN or RED marker and its execution return
   code;
2. reject a pre-existing terminal marker, then append the selected marker to
   `driver.log` exactly once;
3. require every manifest input to be a readable regular file and reject
   duplicate paths;
4. generate a temporary SHA manifest and atomically rename it to
   `SHA256SUMS`;
5. immediately run `sha256sum -c SHA256SUMS`;
6. emit `SHA_LEDGER_PASS` only after that verification succeeds;
7. after verification, write only to stdout/stderr and return the already
   selected GREEN=0 or RED=1 execution status.

Manifest construction or verification failure emits `SHA_LEDGER_RED` and
returns 98. A post-manifest tamper must therefore fail even when the already
hashed `driver.log` contains an execution GREEN marker.

`raw.log` and `driver.log` are mandatory manifest inputs. Census,
classification, hierarchy, and alignment outputs are added when they exist so
that an early Docker RED still receives a self-consistent ledger; their
absence remains visible to the RED marker/classifier and never relaxes GREEN
acceptance.

## Mechanical controls

- GREEN: one GREEN marker in `driver.log`, ledger verifies, runner status 0.
- RED: one RED marker in `driver.log`, ledger verifies, runner status 1.
- Post-manifest tamper: mutate a hashed fixture only after manifest creation;
  require `sha256sum -c` failure, `SHA_LEDGER_RED`, no `SHA_LEDGER_PASS`, and
  runner status 98.
- Static runner order: select precedes terminal write, terminal write precedes
  verification, verification precedes `SHA_LEDGER_PASS`, and no later source
  text appends to `driver.log`.

These controls are evidence-integrity tests only. They launch no TPU and
change no training, hierarchy, census, flag, numerical, or synchronization
behavior.

## Exit gate

P60-2F may be called local PASS only after the task CPU suite, complete pinned
exact-image suite, 372/372 flag audit, branch preflight, shell syntax,
`git diff --check`, secret scan, and no-new-sync audit all pass. A true target
PASS additionally requires a fresh clean committed-tree Zero-HP run whose
wrapper returns 0, prints `SHA_LEDGER_PASS`, and whose immutable root passes an
independent `sha256sum -c SHA256SUMS`.

## Result

- Task CPU suite: 11/11 PASS, including GREEN/exit 0, RED/exit 1,
  post-manifest tamper/exit 98, and duplicate-terminal-marker rejection.
- Document gate: `P60_2_DOCSET_PASS files=14 phase=p60-2f`.
- Pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
  complete P63-inclusive ladder PASS, followed by the 11/11 task suite,
  labels-off/on P59 numerical controls, one-ULP negative, and
  `P60_XPROF_ANNOTATION_API_PASS ... micro_steps=0..15 ... xplane=1 trace=1`;
  final marker `P60_2B_EXACT_IMAGE_PASS ... tpu_devices=0`.
- Static gates: 372/372 flag registry PASS, branch preflight PASS on the
  expected local branch, shell/Python syntax PASS, `git diff --check` PASS,
  changed-file secret scan PASS, and no production training/numerical source
  diff.
- Latest-tip integration: fetched operator tip
  `53876c15f407435dbd44680ad18f5f8e88f3c255`, rebased the already committed
  P60-2E implementation without conflict as local commit
  `d0c6c67474d836664bab69eed665d96d6ff53a25`, restored this additive P60-2F
  tree, and reran the 11/11 task suite, 372/372 audit, branch preflight, diff
  check, and complete pinned exact-image ladder. All passed; no TPU ran.

P60-2F is local/exact-image PASS and target not rerun. Its additive commit and
any fresh one-host launch require separate user approvals.
