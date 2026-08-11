# P42 FrozenLake evaluation handoff

Read `plan.md`, `state.md`, `log.md`, and
`../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md` before operating the target.

The implementation adds a separate evaluation-enabled FrozenLake full-run
manifest. It does not reinterpret the existing evaluation-disabled manifest,
change optimizer placement, or claim that the remaining decode-versus-prefill
carrier is fixed.

The next operator action is to rerun the local gate at the final published SHA,
render `jobset-p33-frozenlake-full-eval.yaml`, perform server-side dry-run, and
launch only after explicit resource approval. Target status remains NOT RUN.
The current workspace passed the pinned-image P33 gate, but the implementation
has not been published; never launch from an inferred or stale SHA.

Required target evidence is the enablement marker, 45 exact reward-inventory
rows at steps 0 through 440, monotonic W&B evaluation curves, complete P33
reports, classification, raw log, rendered manifest, and SHA-256 digests.

Rollback is selection of `jobset-p33-frozenlake-full.yaml`, whose evaluation
triple is `0/1/0`.
