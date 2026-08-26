# Phase D: wide DP8xTP8 first-red observer

## Purpose

The one-host ladder is exhausted without reproducing the target red. The next
experiment therefore stays on the known-red DP8xTP8 M15 workload and carries
the complete first-red instrumentation in the same run. It is not another
blind APC retry and it does not test a proposed repair.

The immutable endpoint contract remains:

```text
A = APC-on production decode, prompt_logprobs=None, logprobs=1
B = independent full-reset serving prefill rescore of A's action IDs
C = trainer old-policy forward

A - B = 0 bytes
B - C = 0 bytes
```

## Why this run is wide

`CANON_CONTINUE_DECODE=8` means most action tokens are created inside the
device loop and cannot have a standard-path seam record without changing the
program under test. The observer therefore does not claim full action-token
coverage. It records every standard-path source token in positions 960..4096
and joins exact A/B records to red action coordinates after the run.
Completion-position-zero mismatches are especially valuable: their source is
the final prompt token, which executes on the standard prefill boundary.

Layer mode records:

1. layer input and output for all 36 transformer layers;
2. final RMSNorm output;
3. raw/processed target logit and log normalizer;
4. observer and production target logprob;
5. request, call, position, token-prefix hash, and replay-ledger page geometry.

Full mode is a second, conditional run only after layer mode selects one layer.
It records these 15 checkpoints for that layer:

```text
layer_input -> input_norm -> q_proj/k_proj/v_proj
-> q_norm/k_norm -> q_post_rope/k_post_rope
-> rpa_output -> o_proj -> attention_residual
-> post_attention_norm -> mlp_output -> layer_output
```

## Single-variable geometry

- workload: FrozenLake M15/main;
- model: Qwen3-8B;
- topology: DP8xTP8 on 64 TPU chips;
- batch: 32 prompts x 8 generations = 256 trajectories;
- context: prompt 4096, response 8192, 15 turns;
- serving: max concurrency 256, scheduler M256, `continue_decode=8`;
- training: precheck only, zero backward, zero optimizer commits;
- control: APC off;
- treatment: APC on;
- observer: identical on both arms.

No production full profile enables APC. The experiment-only profile remains
the only admitted target.

## Layer-run gates

Construction gates:

- renderer creates exactly off/on layer manifests;
- `00_env.sh` resolves exact profile, topology, 960..4096 bounds, 8-GiB seam
  ceiling, 256-MiB tail ceiling, and no KV/terminal-discriminator observer;
- every raw JSON has a paired SHA-verified NPZ;
- observer records contain both A and B;
- APC-off is exact; B-C is zero on both arms;
- APC-on red has at least one exact completion-position-zero join;
- compact evidence bundle verifies its internal `SHA256SUMS`.

Accepted classifications:

| Result | Meaning | Next action |
|---|---|---|
| `M15_LAYER_FIRST_RED_LOCALIZED` | trunk first-red layer found | render `full` at `selected_layer` |
| `M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED` | all layer/final-hidden fingerprints exact; terminal tail red | localize only the reported tail interval |
| `M15_OBSERVER_TREATMENT_EXACT` | observer run did not reproduce target red | retain as one target observation; do not claim repair |
| APC-off red or B-C red | carrier/shared contract invalid | hard stop; no APC conclusion |
| no completion-position-zero join | insufficient standard-path anchor | hard stop; inspect records before redesign |

Fingerprint equality is a coarse localization receipt, not full-tensor byte
equality. A red fingerprint is decisive for an interval; an exact fingerprint
only narrows what the next observer must materialize.

## Full-run gate

The full run must use the layer selected by the layer classifier, not a human
guess. `FIRST_RED_LOCALIZED` must contain:

- exact request/token/cache coordinate;
- last exact checkpoint;
- first red checkpoint;
- source `file:line` anchors;
- A/B selected raw record identities;
- all remaining unobserved continue-decode red points counted explicitly.

Only after this gate may Phase E change one localized degree of freedom.

## Observer neutrality and negatives

- Existing P38 seam/tail implementation is observer-neutral on its certified
  carriers; this phase adds no model observer arithmetic.
- Host tests exercise one-bit fingerprint red, B-C red, missing first-action
  anchor, wrong mode/layer, wrong bounds, and exact control.
- The fresh target supplies the strongest adjacent neutrality check: APC-off
  must stay A-B/B-C exact under the identical observer.
- Dirty-page negative remains required after a repair; it is not part of this
  localization run.

## Evidence boundary

The original observer archive can be several GiB and remains in the registered
attempt. Postflight also creates `m15_wide_seam_bundle.tar` with only selected
raw A/B records, capsule, pre-alignment, replay ledger, classification,
receipt, and manifest. The bundle contains real token material and is not
automatically added to the GCS upload set without separate authorization.

## Stop conditions

- Any real A-B/B-C red remains hard red; no tolerance is admitted.
- No `FIRST_RED_LOCALIZED` means no numerical repair.
- No target run implies no root-cause claim.
- This phase never enables production APC.
