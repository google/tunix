# P38.2l Qwen3-8B one-host incident-capture rehearsal

Date: 2026-08-14 UTC

## Scope

This gate validates instrumentation reachability and neutrality on one real
v5p host with Qwen3-8B DP1xTP4. It does not reproduce or repair the production
carrier. The local envelope uses two prompts, two generations, three
frozen-weight rounds, no prefix cache, no backward, and no optimizer commit.
The incident lower bound is zero locally only to guarantee hook exercise; the
target remains pinned to `[1400, 3072)`.

## Admitted runs

Capture on:

- label: `p38_2l_dress_0814g_on`
- result: `PASS mode=on rounds=3 backward=0 optimizer_commits=0`
- exact-call ledger: 729 records, 2,118,899 bytes
- raw log SHA-256:
  `2528cc1a197352de5938ae9c5c966db80b709d0cfd854823b7c79b187dbb7737`
- pre-alignment SHA-256:
  `1647cca9abb2fee78c9d91aab540167b0a23e55c97c1cfe0fb42b5d9cd84a738`
- incident-ledger SHA-256:
  `08041d01646da70eb6b57595d0296d99f37b7665cd366e80092c4d2084ca6f09`

Capture off:

- label: `p38_2l_dress_0814h_off`
- result: `PASS mode=off rounds=3 backward=0 optimizer_commits=0`
- raw log SHA-256:
  `25c53c21edef7611d376e2121db754e7d7ed90a15a574c0dd88191b192a0a001`
- pre-alignment SHA-256:
  `ed7e8e4d57fca5a3fd5778b34f13fc2cd7861be54bb5f2cb5f52939f473654e7`

The rehearsal script SHA-256 printed by both runs is
`11a278503abb92796ed6f53cc891dde94c166aa7473ee3bbd6640601cd476180`.

## Observer-neutrality verdict

The on/off reports contain three rows each. For every round, all of these
fields are equal: step, action count, action geometry, complete boundary
records, full hashes, masked hashes, blocking/report/warning sets, and verdict.
Therefore token IDs, action masks, S_decode, S_prefill, and T_old are bitwise
unchanged by the host-only capture hook in this envelope.

Round action counts were 409, 565, and 897. Maximum logical KV lengths were
1467, 1311, and 1577. A-B and B-C were exact in all three rounds. This is an
expected under-depth local null and is not evidence that the production
carrier disappeared.

## Failures found by the rehearsal

- The first complete run used exception unwinding after the terminal marker;
  vLLM background threads kept the container alive. The local script now uses
  the same controlled exit 42 as the target.
- The next complete run recursively changed permissions under the state root
  and touched the read-only canonical overlay. Cleanup is now restricted to
  report, round, capture, and capsule outputs. The admitted on/off runs both
  exercised the corrected path.
- The full pinned-image gate found two stale row-cap contracts: the shared P33
  renderer validator and `00_env.sh` still expected 16 while P38.2l requires
  256. Both are synchronized; ordinary non-P38 capsules remain capped at two.

## Gates

- focused classifier: 34 tests PASS;
- GCS persistence, outer postflight, and evidence-seal suites: PASS;
- complete pinned-image P33 CPU/adjacent gate: PASS, terminal marker
  `[P33.WORKLOAD] CPU_GATE PASS`;
- exact-image Qwen3-1.7B and Qwen3-8B overlays: 23 tests each, all 29 manifest
  entries match, terminal marker `P33_EXACT_IMAGE_PASS`;
- shell syntax, Python compilation via isolated pycache, and
  `git diff --check`: PASS.

## Claim ceiling

This evidence admits the instrumentation surface for a clean source-pinned
target run. It does not prove a carrier cause, a numerical repair, or
full-training zero-TIM.
