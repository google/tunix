# Goal

Produce a matched one-host Qwen3-1.7B GSM8K XProf pair that compares the
untreated Native numerical/trainer program with the strict optimized Zero-TIM
V1 bundle. Both arms must consume the same profiled training batch and produce
a device-plane update capture containing backward work.

This replaces neither the historical P58 DeepSWE pair nor the 64-chip GSM8K
target. It is a four-chip DP4×TP1 operation-attribution and launch-risk proxy.

Current outcome: the standalone backward-capture requirement is satisfied for
both arms. The stricter same-profiled-batch pairing requirement is not: the
two inference programs sampled different completions despite identical seeds,
so the current pair is correctly classified inconclusive for causal timing.

## Active follow-up: P60-2 readability

Make the already-complete Zero-TIM/P59 update capture navigable as one bounded
training/update/group/transaction hierarchy without changing its numerical
program, synchronization, fixed DP reduction, optimizer transaction, or the
official semantic Perfetto vocabulary. The revised carrier must emit the same
host `StepTraceAnnotation("train", step_num=1)` contract as Native and retain
non-empty `Steps` rows on all eight TPU device planes. A future, separately
approved one-host Zero-HP capture must also retain 3/3 commits, 51/51 strict
  alignment PASS, and complete backward on 8/8 TensorCore planes. This follow-up
  does not repair the input mismatch and does not authorize a timing ratio.

Acceptance also requires fail-closed evidence packaging: freeze one terminal
marker in `driver.log`, generate `SHA256SUMS` only after all hashed files are
immutable, verify it immediately, and emit `SHA_LEDGER_PASS` before returning
success. A runtime/classifier GREEN with an invalid ledger is not TARGET PASS.
