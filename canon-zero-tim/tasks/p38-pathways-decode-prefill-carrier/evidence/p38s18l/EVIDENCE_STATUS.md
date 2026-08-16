# P38s18l Evidence Status

- **Status**: Complete & Verified
- **Run ID**: `p38s18l` (JobSet `canon-p38-fl-stock-p38s18l-9a834574`)
- **Commit**: `9a83457417fc995079a4beaf7c0c1694f4da605f`
- **Topology**: 64 TPU chips (`DP16xTP4`, concurrency 256)
- **Diagnostic Mode**: `--seam-mode layer` (All 36 Transformer layers intermediate state capture)
- **Key Findings**:
  1. **B-C Boundary (`S_prefill` vs `T_old`)**: `0` mismatches (STRICT EXACT 0 DIFF across all rounds).
  2. **Layer Seam Observer**: All 36 Transformer layers (`layer_input`, `layer_output`) and `final_norm` are 100% bitwise identical between decode and prefill for all joined mismatch positions (`All-36-Layers-Equal = 20, Divergent Signatures = {}`).
  3. **Root Cause Localization**: The hidden representation chain is bitwise exact. The minor A-B logprob divergence originates strictly in the tail `lm_head` / log-softmax reduction normalizer.
