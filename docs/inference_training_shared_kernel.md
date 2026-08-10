# Inference / Training Shared Kernel Summary

## Shared Forward Path For Actor Logps

The key idea in `yuxzhang/p22-align-integration` is to make the
training-side actor-logprob scoring path reuse the same vLLM TPU engine-module
forward path used by inference and rollout.

Before this change, the two sides roughly followed separate paths:

```text
Inference / rollout:
  vLLM TPU engine forward
  -> vLLM compute_logits
  -> vLLM processed logprobs

Training:
  Tunix NNX model forward
  -> JAX log_softmax
  -> gather token logps
```

Even with identical weights, these two paths can produce mismatched logps
because they may differ in forward implementation, log-softmax implementation,
sampling processors, masking, or token alignment.

The branch introduces a canonical actor-logprob scoring path:

```text
trainer weights
  -> map to vLLM engine state format
  -> call vLLM engine model_fn
  -> call vLLM engine compute_logits_fn
  -> call shared canonical log_softmax / gather
  -> trainer logps
```

So the training side no longer computes actor logps through an independent NNX
forward path. Instead, `Qwen3EngineForwardAdapter` calls the rollout/inference
engine's real `model_fn` and `compute_logits_fn`, making trainer-side logps and
inference-side logps share the same engine-level forward/logprob path.

For actor-logprob scoring, this is the full forward/scoring path:

```text
tokens + trainer weights
  -> vLLM engine-module model forward
  -> vLLM compute_logits
  -> canonical log_softmax / token logprob gather
  -> per-token actor logps
```

So "shared forward path" here is intentionally stronger than "one small logprob
kernel": the trainer-side actor-logps forward/scoring computation is
canonicalized end to end. But it still does not mean the entire training step is
replaced. Training keeps its own GRPO loss, backward/VJP path, gradient
accumulation, optimizer update, trainer state, checkpointing, and scheduling.

The precise summary is:

> This branch does not make training and inference the exact same program.
> Instead, it canonicalizes the full trainer-side actor-logprob forward/scoring
> path: trainer-side actor logps are computed through the vLLM TPU engine-module
> forward, vLLM logits path, and canonical logprob gather used by inference.
> This aligns actor-logps computation, while leaving the overall training step
> trainer-owned.

## Can We Directly Use The Inference Kernel For Training?

Yes, but only with important constraints.

Using the inference forward/logprob path inside training is feasible when the
scope is trainer-side actor-logps scoring. In that scoped sense, the complete
logprob forward path can be shared. However, replacing the entire training step
with an inference path usually does not work unless the inference path supports
differentiable execution and has a valid VJP/backward path.

Inference kernels are normally optimized for serving:

```text
input tokens -> logits / logps
```

Training needs:

```text
input tokens -> logits / logps -> loss -> gradients -> optimizer update
```

So the inference kernel must satisfy several requirements before it can be used
in training:

```text
1. It must support value_and_grad so gradients can flow back to trainer params.
2. Parameter mapping must be pure and differentiable, not just mutate serving state.
3. Forward numerical semantics must match inference exactly.
4. log_softmax / gather must have a correct VJP.
5. cache, position, mask, and sampling processor semantics must be controlled.
```

`p22-align-integration` takes this scoped path. It does not blindly replace all
training compute with the inference implementation. Instead, it wraps the vLLM
engine forward in a canonical adapter and uses it specifically as the
trainer-side actor-logps forward/scoring path. It also adds a custom VJP for the
logprob/gather path so the trainer loss can still backpropagate.

The concise takeaway is:

> In principle, we can use the inference engine's full forward/logprob scoring
> path for trainer-side actor logps, but only after wrapping it as a
> differentiable engine module and providing stable VJP support for
> serving-oriented operations such as logprob gather. P22 does exactly this for
> actor-logps scoring; it does not replace the full training step.
