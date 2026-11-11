# Dr. MAS source provenance

Reference: [langfengQ/DrMAS](https://github.com/langfengQ/DrMAS/tree/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9),
commit `c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9`.
The Apache 2.0 license is included in `LICENSE.DrMAS`.

- `drmas_math.py` is an unchanged copy of
  `agent_system/environments/env_package/math/utils.py`.
- `drmas_reward.py` copies that directory's `math_reward.py`, changing only
  the utility import to the local vendored module, adding attribution, and
  removing trailing whitespace on a blank line.
- `prompts.py` contains unchanged `SOLVER_PROMPT` and `VERIFIER_PROMPT`
  text values from `agent_system/agent/agents/math/`; one trailing prompt space
  is encoded as `\x20` in the Python source without changing the string value.
- `workflow.py` adapts the orchestration and projection to async Tunix calls,
  retaining the original semantics and copyright notices.
- `tests/dist_rl/swarm/reference/drmas.json` contains exact extracted source
  functions, source-file hashes, and original copyright headers. Tests execute
  these functions with small transport stubs rather than importing verl. This
  makes the numerical reference independent of the Tunix implementation and
  reproducible offline. The reference loss is PyTorch; the training loss is JAX.

The selected recipe explicitly enables shared weights. The upstream
`examples/drmas_trainer/run_math.sh` defaults to `model_sharing=False`;
the repository supports both choices. This recipe tests the shared variant.
