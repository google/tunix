# State

- Status: IMPLEMENTED / HOST PASS / BOTH ONE-HOST ARMS PASS / PAIR INPUT-MISMATCH.
- Worktree: `/home/yuxuan/code_rl_repro/worktrees/p60_gsm8k_native_zero_xprof_0824`.
- Branch: `local/p60-gsm8k-native-zero-xprof-0824`.
- No commit, push, image publication, or Kubernetes launch is authorized.
- Evidence grade: direct-TPU development/analysis-grade because the source tree
  is intentionally dirty. It proves the carriers work but is not a clean-SHA
  signed release receipt.
- One-host result: Native and Zero-HP each completed 3/3 optimizer updates and
  passed the all-plane backward/decode-absence census. Zero-HP additionally
  passed 51/51 strict alignment records with zero FAIL.
- Pair result: `INCONCLUSIVE_INPUT_MISMATCH`; source/image/model/topology/window
  and prompt hashes match, while sampled completions and derived advantages do
  not. No causal timing claim is allowed from this pair.
- Next optional gate: a separately designed frozen-train-batch replay if a
  causal Native-vs-Zero backward timing comparison is still required.
