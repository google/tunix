# P36 Pathways proxy XLA-flag handoff

## Scope

P36 tests whether the remote Pathways proxy compiler, not only the JAX client process, receives
`--xla_allow_excess_precision=false`. It is a topology gate. It does not load a model, start
training, initialize W&B, or change declared model dtypes.

## Confirmed before the experiment

- The client profile exports the flag.
- The existing client preflight checks only that client environment value.
- The checked-in Pathways proxy args omit the flag.
- Existing 64-chip results remain valid measurements of the flag-off remote compiler path. They
  are not evidence that the canonical flag reached that compiler.

## Local gate

```bash
bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh
```

The renderer must reject missing, duplicate and `true` proxy-environment controls, a raw proxy
argument, and placement on the resource manager or worker. A successful local gate proves YAML
delivery and fail-closed behavior only. It is not a numerical result.

The initial 6/6 local gate validated a raw-argument contract that the target proxy subsequently
rejected. That historical result is preserved in P36.1 but is superseded. The corrected
proxy-`XLA_FLAGS` contract passes 7/7 locally; the adjacent P35 renderer passes 7/7. This host has
no `kubectl` binary or configured GKE context, so no corrected target JobSet was created here.

## Delivery correction after `flagon1`

Target Attempt `flagon1` ended before numerical execution:

```text
ERROR: Unknown command line flag 'xla_allow_excess_precision'
```

The corrected manifest must contain:

```yaml
env:
- name: XLA_FLAGS
  value: --xla_allow_excess_precision=false
```

in `pathways-proxy` only. Its `args` list must contain no
`--xla_allow_excess_precision=...` item. This is a new delivery attempt, not a retry or a rewrite
of `flagon1`.

## Direct-attached sensitivity control

A paired four-device v5p run used the same frozen image and probe in both arms. Adding
`--xla_allow_excess_precision=false` reduced differing bytes by 88.04% to 93.80% across depths
4 through 24. The ON arm retained a 2.71% to 3.22% differing-byte fraction, so the flag is a
strong carrier in this generic probe but not its only carrier. This does not replace the P36.2
Pathways measurement and does not predict a bitwise target result.

## Target render

Render only from a published 40-character source commit:

```bash
python3 canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py \
  --source-commit <published-40-character-sha> \
  --run-id envon1 \
  --output /tmp/p36-proxy-xla-envon1.yaml
```

Before applying it, inspect the rendered proxy args and env and record the manifest SHA-256. The
JobSet must be Attempt 0. Archive the head log, `p36_waycount.raw.log`, live Pod YAML, proxy log,
resource-manager log, worker logs and Kubernetes events before deleting the JobSet. Proxy startup
without the previous unknown-flag error proves acceptance only; the complete way-count table is
required to show that the remote compiler consumed a load-bearing setting.

## Registered target verdict

The complete way-count table is required. The replicated arm is the primary discriminator.
The historical flag-off run used the same core P1 computation for widths 2 and 4, but the current
unified runner contains additional probes. It is a high-value screening baseline, not a perfect
same-source causal control.

| Observation | Verdict | Next action |
|---|---|---|
| Replicated drift becomes bitwise zero | Proxy flag is a load-bearing carrier | Run one P35 envelope-only A/B/C gate with the same proxy contract |
| Replicated drift materially decreases but remains nonzero | The flag is a strong carrier candidate | Add one matched current-source flag-off control before causal promotion |
| Replicated drift is effectively unchanged | The screening run does not support the hypothesis | Add one matched current-source flag-off control before declaring it falsified |
| Proxy rejects the argument or exits | Delivery contract failure | Fix the argument form; do not report a numerical FAIL |

P36 must not promote the flag into shared P33/P34 workload defaults until the target result is
reviewed. P35.3 replay socket failures remain a separate infrastructure line.
