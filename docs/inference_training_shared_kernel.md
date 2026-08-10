# Inference / Training Shared Kernel Summary

## Shared Kernel For Logps Computation

The key idea in `yuxzhang/p22-align-integration` is to make the
training-side logps forward path reuse the same vLLM TPU engine kernel path
used by inference and rollout.

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

The branch introduces a canonical path:

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

The shared part is:

```text
model forward kernel
logits computation kernel
log_softmax / token logprob gather kernel
```

This does not mean the entire training program is shared. Training still has
its own GRPO loss, backward pass, gradient accumulation, and optimizer update.

The precise summary is:

> This branch does not make training and inference the exact same program.
> Instead, it canonicalizes the actor logps computation path: training-side
> actor logps are computed through the vLLM TPU engine-module forward and the
> same logprob pipeline used by inference, so inference and training are aligned
> at the logps kernel level.

## Can We Directly Use The Inference Kernel For Training?

Yes, but only with important constraints.

Using the inference forward/logps kernel inside training is feasible if we only
use it to compute training-side logps. However, replacing the entire training
forward/backward path with an inference kernel usually does not work unless the
inference kernel supports differentiable forward execution and has a valid
VJP/backward path.

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

`p22-align-integration` takes this safer middle path. It does not blindly
replace all training compute with the inference kernel. Instead, it wraps the
vLLM engine forward in a canonical adapter and uses it specifically for actor
logps computation. It also adds a custom VJP for the logprob/gather path so the
trainer loss can still backpropagate.

The concise takeaway is:

> In principle, we can use the inference kernel for training-side forward/logps
> computation, but only after wrapping it as a differentiable engine module and
> providing stable VJP support for serving-oriented operations such as logprob
> gather. P22 does exactly this: it safely connects the inference engine forward
> into trainer-side logps computation, rather than doing a raw kernel
> replacement.
