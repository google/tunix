# P36.2 — Target Pathways way-count measurement

- Status: active

## Finding

- Confirmed: The historical flag-off way-count probe used the same core replicated, stock-AR and
  F4 computation for widths 2 and 4.
- Confirmed: The current unified runner has additional probes before and after P1, so the
  historical run is not a perfect same-source control.
- Hypothesis: A remote proxy flag-on run will remove or materially reduce the replicated
  cross-program drift.

## Execution

1. Publish the reviewed P36 source without changing the pinned proxy image.
2. Render one `envon1` JobSet with a unique run ID and record its SHA-256.
3. Confirm live Pod YAML contains no raw excess-precision proxy argument, exactly one proxy
   `XLA_FLAGS=--xla_allow_excess_precision=false` entry, and the expected proxy image digest.
4. Run Attempt 0 in `gate-only`; do not load a model or initialize W&B.
5. Archive the complete T1 log, live Pod YAML, proxy/RM/worker logs and Kubernetes events.
6. Validate the expected way-count row count before reading any magnitude.

## Exit gate

- Pass: Every registered width/depth/arm row is present and the replicated flag-on arm is bitwise
  exact. Proceed to a P35 envelope-only recheck.
- Partial: The replicated drift materially decreases but is nonzero. Add a matched current-source
  flag-off control before attributing the change or selecting the next carrier.
- Unchanged: Do not call the hypothesis falsified from the historical comparison alone. Add a
  matched current-source flag-off control; only that pair can close the causal claim.
- Infrastructure: Unknown proxy flag, missing row, retry, disconnect or missing proxy evidence is
  inconclusive and has no numerical verdict.

## Result

Attempt `flagon1` was inconclusive: the proxy rejected the raw command-line argument before any
way-count row ran. The corrected `envon1` attempt is pending target execution.
