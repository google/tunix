# P38.2w — LM-head program discriminator

Status: host implementation and one-host gates complete. Review, commit, push,
and the target run each require separate user approval.

## Entering evidence

P38s21/source `f7f9fee6256f1f31f99c2794c4f346e9196b010c` completed and
sealed diagnostic rounds 0 and 1. Round 2 exceeded the 4-GiB local observer
bound, so the run has no third round, controlled exit, root `COLLECTED`, or
root `COMPLETE`. Its correct run-level verdict is
`ANALYSIS_GRADE_PARTIAL_2_OF_3`.

The two admitted rounds report:

```text
round 0: N_action=45,276; A-B=47 elements / 76 bytes; max_abs=0.162704...
round 1: N_action=44,695; A-B= 7 elements /  9 bytes; max_abs=0.230720...
round 0/1: B-C=0 elements / 0 bytes
```

The committed classifier joins all 54 red points. For those selected points,
the complete captured `final_hidden_rows` are byte-exact and the first
measured red interval is `lm_head_logits`. This localizes the carrier to the
program interval between final hidden and raw-logit evidence. It does **not**
prove a particular GEMM reduction order, full-vocabulary logit equality, or
backbone equality for unselected rows.

Source archaeology establishes the relevant executable split:

- Qwen3-8B's seven transformer projections are intercepted by the canonical
  Pallas projection shim.
- `lm_head` is a separate `JaxLmHead` with equation `TD,DV->TV`; it is not one
  of those seven registered Pallas sites.
- Decode enters lm_head at local `M=16`; prefill/rescore enters at local
  `M=256`. `CANON_LOGPROB_M=256` starts after logits and does not unify this
  lm-head input geometry.
- The reduction dimension is hidden K=4096. Vocabulary V=151,936 is an output
  dimension; describing the root cause as a "sum over vocabulary" is wrong.

## Historical negative that constrains this phase

P19 already implemented `CANON_MM_ALGO=1` with
`DotAlgorithmPreset.BF16_BF16_F32`. It removed isolated M=1/M=256 dot drift,
but a PATHTRACE-confirmed historical M=16/M=2048 end-to-end arm had exactly no
effect, and the isolated real M=16/M=2048 dot pair was already exact. Therefore
the preset is a candidate discriminator, not a known repair.

The present experiment is still justified because P38s21 newly isolates the
first measured interval to lm_head and the actual current geometry is
M=16/M=256. The historical negative remains in the decision table and may not
be rewritten as a success.

## Deliverable

1. A real-Qwen3-8B-weight one-host operator screen compares the same first 16
   BF16 hidden rows under M=16 and M=256 for both default einsum and
   `BF16_BF16_F32`. It uses the production K=4096, V=151,936 and TP4 vocab
   sharding, four deterministic input seeds, plus a one-bit negative control.
2. The P38 renderer gains one explicit `--lm-head-algo` arm. It sets the
   existing `CANON_MM_ALGO=1` and fixed preset only under stock,
   concurrency-256, canonical-Pallas-projection geometry. It rejects U,
   concurrency-32, seam, tail, and terminal-observer combinations.
3. The off path leaves both algorithm env variables absent. No model default,
   production profile, prefix-cache setting, backward path, or optimizer path
   changes.
4. One slim P38s22 target run executes three frozen precheck-only rounds. It
   reuses existing bounded incident/KV capture but does not recapture the
   multi-gigabyte terminal corpus.

## Local gates

```bash
set -euo pipefail
JAX_PLATFORMS=cpu python3 -m unittest \
  canon-zero-tim/tests/p38_serving/test_lm_head_probe.py \
  canon-zero-tim/tests/p38_serving/test_render_p38_serving_jobsets.py -v
bash -n \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_lm_head_onehost.sh
python3 -m py_compile \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/probe_p38_lm_head.py \
  canon-zero-tim/cluster/render_p38_serving_jobsets.py
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_lm_head_onehost.sh \
  p38_2w_<unique>
```

The one-host verdict is one of:

| Verdict | Meaning |
|---|---|
| `ALGORITHM_ELIMINATES_OPERATOR_DRIFT` | candidate strengthened; target remains mandatory |
| `BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE` | random one-host inputs did not expose the carrier; target is still allowed |
| `ALGORITHM_NOT_SUFFICIENT` | do not launch this target arm; move to a dedicated fixed-tile lm_head |
| `FAIL_NEGATIVE_CONTROL` | probe invalid |

## Target gate: P38s22

Render from one clean, published full SHA:

```bash
set -euo pipefail
RUN_ID=p38s22
OUT="/tmp/p38-serving-$RUN_ID"
test ! -e "$OUT"
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --max-concurrency 256 \
  --lm-head-algo
YAML="$OUT/jobset-p38-serving-stock.yaml"
grep -Fq 'name: CANON_MM_ALGO' "$YAML"
grep -Fq 'value: BF16_BF16_F32' "$YAML"
kubectl apply --dry-run=server -f "$YAML"
kubectl apply -f "$YAML"
```

Admission requires Attempt 0, three frozen round records, exact B-C in every
round, `CANON_MM_ALGO` PATHTRACE evidence, zero backward, zero optimizer
commits, controlled exit 42, and complete durability markers. Any missing
round/marker is `INCONCLUSIVE`.

## Decision table

| P38s22 result | Decision |
|---|---|
| A-B exact in all three rounds; B-C exact | preset is a causal repair candidate for this program interval; run strict backward-no-commit before promotion |
| A-B remains red; B-C exact | `CANON_MM_ALGO` is rejected again; build a dedicated fixed-tile Pallas lm_head and do not tune more generic precision flags |
| B-C red or canonical projection attestation missing | intervention scope invalid; reject the run |
| infrastructure/transport incomplete | `INCONCLUSIVE`; preserve Attempt 0 and repair only the failed infrastructure contract |

## Claim ceiling and rollback

P38.2w can establish only whether the existing pinned dot algorithm changes
the current at-scale lm-head carrier. A green one-host operator screen is not
Pathways evidence. A green P38s22 does not by itself restore zero-TIM training;
strict backward-no-commit and then full-workload gates remain mandatory.

Rollback is to omit `--lm-head-algo`, leaving `CANON_MM_ALGO` and
`CANON_MM_ALGO_PRESET` unset. The experiment never changes the default path.
