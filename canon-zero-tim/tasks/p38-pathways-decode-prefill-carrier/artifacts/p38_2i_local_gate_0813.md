# P38.2i local gate — 2026-08-13

## Scope

This record covers the P38s12a request-journal construction rebased on parent
`df46a880426460e96f2b160aef73a532b2bfe58b`. No cluster run, backward,
optimizer commit, or training launch occurred.

## Results

- serving-capture classifier: 30 tests PASS;
- serving JobSet renderer: seven tests PASS, including stock-only output;
- outer postflight: PASS, including red/U/capture-error/missing-coverage and
  marker-present-but-journal-file-missing negative controls;
- pinned Qwen3-1.7B overlay: 23 tests PASS, 29/29 manifest entries;
- pinned Qwen3-8B overlay: 23 tests PASS, 29/29 manifest entries;
- exact-image terminal marker:
  `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 serving_capture_cases=13 overlays=2`;
- complete pinned-image P33 CPU/adjacent gate: PASS with terminal marker
  `[P33.WORKLOAD] CPU_GATE PASS`;
- installed patched runner SHA-256:
  `3a219b251020894ade2002e480aa8b3fef90ea62a70794116b143bad89b36b17`;
- Python compilation, shell syntax, executable-source ASCII scan, and
  credential-pattern scan: PASS. Ordinary source whitespace checks pass;
  patch 13 retains the unified-diff format's required one-space blank context
  markers and is instead validated by patch application plus exact-image
  manifest identity.

## Proven and not proven

Proven locally: patch 13 applies to both pinned model overlays; the journal is
default-off, host-only, archived, and required by postflight; flattened block
tables and multiple unique row joins are accepted while ambiguous joins fail;
all selected capsule rows must have a journal join.

Not proven: that P38s12a reproduces the carrier, that it captures every target
row on Pathways, that observation generations equal allocator generations,
that KV contents are equal or stale, or that RoPE/RPA/residual/logits is the
cause. P38s12a and P38s12b are NOT RUN.
