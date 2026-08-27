# p58z06 NNX loader-metadata failure

Verdict: `INCONCLUSIVE`.

The 128-device Qwen3-4B Zero-HP job loaded the clean 1,012-task dataset and
finished vLLM model warmup, but failed during disaggregated canonical-adapter
initialization. The first hard error was the raw NNX State-treedef comparison
between the Pathways-loaded live runner and the weight-free trainer-mesh
reconstruction. The live loader adds `_is_loaded=True` to all 398 parameter
Variables; that provenance participates in Flax's State treedef even though it
does not alter parameter paths or values.

The attempt produced no rollout, trajectory file, trainer logprob, alignment,
backward, optimizer commit, or checkpoint. It cannot be resumed. The
post-exception vLLM finalizer `AttributeError` is secondary shutdown noise.

The authoritative raw log is tracked at:

`../../../../debug_logs/p58_p58z06_deepswe_nnx_state_tree_mismatch.raw.log`

The raw log does not contain an exact git source SHA. Do not infer or silently
substitute one from the evidence commit.
