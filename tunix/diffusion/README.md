# Diffusion Training Contracts

This package defines the model-agnostic JAX boundary used by Tunix diffusion
objectives. Model integrations prepare target-aligned tensors before entering
Tunix; Tunix owns validation, scoring interfaces, and optimization.

## Canonical Batch

`DiffusionTokenBatch` contains:

- a model-input PyTree whose array leaves are batch-major and consumed by a
  model-specific scorer;
- target token IDs with shape `[batch, length]`; and
- explicit nonnegative loss weights with the same target shape.

Targets refer to physical prediction positions. A model integration must
resolve same-position versus shifted-logit alignment before constructing the
batch. Corruption policy, mask IDs, block sizes, tokenizer behavior, and prompt
selection remain outside this contract.

## Scoring

`DiffusionLogitsFn` receives an NNX model and the prepared model inputs, then
returns floating-point logits shaped `[batch, length, vocabulary]`.
`compute_diffusion_logits` validates the batch and verifies that returned logits
are target-aligned. Validation relies on static shape and dtype information
while tracing; concrete finite and range checks run when values are eagerly
addressable.

This package does not generate a rollout, determine whether a batch is fresh,
or define an SFT, distillation, or reinforcement-learning loss. Those features
compose this contract in later, independently reviewable layers.

## Comparison with Autoregressive Models

Autoregressive (AR) generation predicts one new token after another. Diffusion
generation instead starts from corrupted or masked positions and can refine
multiple positions during each denoising step. This changes the tradeoff rather
than guaranteeing that either approach is always faster or more accurate:

| Dimension | Autoregressive model | Diffusion model |
| --- | --- | --- |
| Sequential dependency | One decoding step per generated token | One step per denoising round; a round can update multiple positions |
| Compute profile | Incremental decoding can reuse a KV cache | Fewer sequential rounds are possible, but each round may score a full sequence or block |
| Quality controls | Sampling strategy and token budget | Training objective, corruption schedule, denoising schedule, and sampling strategy |
| Tunix behavior in this PR | Existing paths are unchanged | Contracts only; no training or inference implementation is added |

Consequently, this PR has no direct performance or quality delta to report.
Implementation PRs should compare matched model sizes, tokenizers, data, and
training-compute budgets. Runtime results should identify the hardware, batch
and sequence sizes, denoising schedule, and whether compilation is included,
then report latency and throughput. Quality should be measured on the same
fixed evaluation tasks and seeds at matched compute or latency budgets; an AR
perplexity and a diffusion training loss are not directly interchangeable.
