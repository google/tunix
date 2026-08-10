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

The renderer must reject missing, duplicate and `true` proxy flag controls. A successful local
gate proves manifest delivery and fail-closed behavior only. It is not a numerical result.

Reviewed local result: PASS, 6/6. The adjacent P35 renderer passed 7/7 and the P33 renderer passed
6/6 in the pinned frozenlake image. This host has no `kubectl` binary or configured GKE context,
so no target JobSet was created by the implementation host.

## Target render

Render only from a published 40-character source commit:

```bash
python3 canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py \
  --source-commit <published-40-character-sha> \
  --run-id flagon1 \
  --output /tmp/p36-proxy-xla-flagon1.yaml
```

Before applying it, inspect the rendered proxy args and record the manifest SHA-256. The JobSet
must be Attempt 0. Archive the head log, `p36_waycount.raw.log`, live Pod YAML, proxy log,
resource-manager log, worker logs and Kubernetes events before deleting the JobSet.

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
