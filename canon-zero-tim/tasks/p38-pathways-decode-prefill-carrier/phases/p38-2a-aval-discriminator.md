# P38.2a — Model-free tail aval and sharding discriminator

- Status: active

## Question

Does the production tail compile and execute under distinct global aval and
placement contracts even though rollout and rescore resolve the same Python
callable and the same per-rank canonical log-softmax?

## Confirmed facts

- `_make_canonical_compute_and_gather` accepts both global M256 and global
  M4096 in the DP16 contract. These are distinct JAX compilation cache keys.
- r35 recorded decode `sample` at global M16, decode score at padded global
  M256, and prompt transform/score at global M4096. The inner per-rank
  log-softmax is local M256 in both score arms, but the outer executable avals
  are not equal.
- The direct-attached DP1 result used prompt M256 and decode score M256 and was
  exact across 11,340 action tokens. It cannot establish DP16 Pathways
  equality.
- FrozenLake's r35 maximum absolute A-B difference was `0.10390`. No
  one-ULP-only assumption is admitted.

## Deliverable

Add one default-off, no-model probe that runs the live sampling transform and
the live shared canonical scorer at the registered shapes:

| Topology | Transform arms | Score arms |
|---|---|---|
| direct-attached DP1xTP4 control | M16 vs M256 | M256 vs M256 |
| target DP16xTP4 | M16 vs M4096 | M256 vs M4096 |

Both arms receive identical real-row logits, metadata, and forced target IDs.
The report must contain global shapes, dtypes, named-sharding specs, device
order, sampled-token equality, processed-target equality, target-logprob
equality, implied-normalizer equality, measurement count, and a one-bit
negative control. It must state that model-free equality cannot prove the
production model boundary.

## Pre-registered interpretation

| First red observation | Meaning | Next repair target |
|---|---|---|
| processed target | sampling transform / outer sample executable | normalize the transform's complete global aval before considering score changes |
| processed target exact, target logprob red | canonical scorer's outer aval or gather envelope | normalize the score aval and preserve the existing Pallas local-M contract |
| target logprob exact, implied normalizer red | invalid instrument | reject the run because `normalizer = processed_target - logprob` is algebraic |
| every model-free field exact | carrier not reproduced without the model | do not claim a fix; continue to both production probes |

The GSM8K production probe tests the tail hypothesis. The FrozenLake probe is
independent and must include multi-turn coordinates. Neither workload is a
fallback for the other.

## Exit gate

- CPU tests reject bad topology, missing measurements, mismatched real-row
  inputs, and a negative control that is not observed.
- A direct-attached four-chip run completes with the expected DP1 shape table.
- The target renderer is source-pinned, Attempt-0, no-model, no-checkpoint,
  no-backward, no-optimizer, and defaults the new probe off everywhere else.
- A red numerical comparison is a completed diagnostic result, not a process
  failure. Missing measurements or a red negative-control contract exits
  nonzero and taints all later probes.

## Rollback

Leave `CANON_RUN_P38_AVAL` unset. The unified admission runner excludes the
probe and all production paths remain unchanged.

## Direct-attached result

- Status: local control passed; target Pathways arm remains pending.
- Command: `bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_aval_onehost.sh 0810_r1`
- Result: `MODEL_FREE_NOT_REPRODUCED`. All five comparisons were exact and the
  injected one-bit negative control reported one differing element.
- The sample transform compiled to distinct HLO at M16 and M256, while both
  M256 scorer arms had the same HLO digest. Distinct transform programs did
  not cause a numerical difference on direct-attached DP1xTP4.
- Artifact: `../../debug_logs/p38_aval_0810_r1.result.json`
  (`sha256=f3c783a6d2d29dac0f1700b474f404f70831924fd946f5ac830cc8851cbf595f`).
- Raw log: `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.raw.log`
  (`sha256=f14101994a166a9639f3a87ee992be5f36f056c3fbf913c317fb9267eaf233a1`).
- Claim ceiling: this is not a DP16 Pathways result and does not explain the
  GSM8K or FrozenLake production boundaries.
