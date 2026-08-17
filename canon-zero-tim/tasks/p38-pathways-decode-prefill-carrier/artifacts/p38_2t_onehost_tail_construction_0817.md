# P38.2t one-host canonical-tail construction result

Date: 2026-08-17 UTC

Command:

```bash
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_tail_construction.sh targetjoin0817
```

Result:

```text
backend=tpu
device_count=4
shape=[256,151936]
dtype=float32
same_python_callable=true
outer_programs=2
differing_elements=0 / 38,895,616
negative_control_differing_elements=1
verdict=PASS_CONSTRUCTION_ONLY
```

Evidence receipts:

```text
fb3db128c66301368bf9ab1461396586c68fa9df1059062e69922d5795d83981  p38_tail_targetjoin0817.raw.log
4b5b27daf313223974f7428004ea49a903e14a6a501602c841e232f057b550f1  p38_tail_targetjoin0817.result.json
6a839b7bb559b033fcf362f530f04080d2ebe2136f61ceffca073f82accb3df6  run_p38_tail_construction.sh
```

The raw log records source commit
`0eb7049e821b0b02f046b4cf2f88239ef67c353c`, pinned image
`tunix_frozenlake_image:vllm-tpu0.25.0`, probe SHA
`9a375b9e10cf1ecf5cf884ea12873f204864ba3481405f0d8b8e99e1023c196e`,
and canonical-logsoftmax SHA
`ad023b2720c54f87d8dbca7ddc9c87246d7c3cdc57b8df36abc69240f7839b92`.

Interpretation: identical production-shaped logits remain bitwise identical
through the canonical log-softmax even when embedded in two distinct outer
programs, and the negative control is live. This rejects only the bounded
same-input one-host construction failure. It does not prove that production A
and B supply identical full logits or that 64-chip Pathways uses an identical
executable envelope.
