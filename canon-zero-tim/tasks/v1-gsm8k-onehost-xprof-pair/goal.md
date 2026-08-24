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
