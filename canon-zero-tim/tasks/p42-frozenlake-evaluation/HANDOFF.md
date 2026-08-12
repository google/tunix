# P42 FrozenLake evaluation handoff

Read `plan.md`, `state.md`, `log.md`, and
`../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md` before operating the target.

The implementation adds a separate evaluation-enabled FrozenLake full-run
manifest. It does not reinterpret the existing evaluation-disabled manifest,
change optimizer placement, or claim that the remaining decode-versus-prefill
carrier is fixed.

Target attempt `p42e2` is now archived. It completed the 100-prompt x
8-generation step-0 evaluation and proved the corrected DP16 geometry, then
stopped before the first fixed reduction because the reducer required all 16
compact rank-gradient signatures to be distinct. This was not an evaluation,
global-M, Pathways, OOM, or optimizer failure.

Production rewards are allowed to produce duplicate signatures. FrozenLake's
binary reward plus RLOO makes this expected when every generation for one
prompt has the same outcome and therefore receives an exact zero advantage.
The local fix disables only this value-diversity assumption in the production
adapter. Synthetic admission probes remain strict. Rank order 0 through 15,
exactly 16 contributions, the eight-round fixed tree, finite gradient health,
and post-reduction replica equality remain hard errors. Each production group
now prints `unique_rank_fingerprints=K/16` for durable observability.

The next operator action is to rerun the local gate at the final published SHA,
render `jobset-p33-frozenlake-full-eval.yaml`, perform server-side dry-run, and
launch only after explicit resource approval. The target retry must show all
16 `reverse_group_done` rows, `replicas_exact=1`, one optimizer commit, and
continuation to the next training step. The current workspace passed the
pinned-image gates, but the fix has not been published or target-tested; never
launch from an inferred or stale SHA.

Required target evidence is the enablement marker, 45 exact reward-inventory
rows at steps 0 through 440, monotonic W&B evaluation curves, complete P33
reports, classification, raw log, rendered manifest, and SHA-256 digests.

Rollback is selection of `jobset-p33-frozenlake-full.yaml`, whose evaluation
triple is `0/1/0`.
