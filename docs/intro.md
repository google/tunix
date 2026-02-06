<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Tunix: A JAX-native LLM Post-Training Library

**Tunix (Tune-in-JAX)** is a JAX based library designed to streamline the
post-training of Large Language Models. It provides efficient and scalable
supports for:

- **SOTA Training performance on TPUs**
- **Supervised Fine-Tuning**
- **Reinforcement Learning (RL)**
- **Agentic RL**

Tunix leverages the power of JAX for accelerated computation and seamless
integration with JAX-based modeling framework like
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), and
integrates with high-performance inference engines like vLLM and SGLang-JAX for
rollout.

**Current Status: V2 Release**

Tunix is under active development. Our team is actively working on expanding its
capabilities, usability and performance. Stay tuned for upcoming updates and new
features! See [Talks and Announcements](talks.md) for latest updates, talks, and blog posts.


## High Level Architecture

Tunix serves as a state-of-the-art post-training library within the JAX training
stack, positioned to leverage foundational tools like Flax, Optax, Orbax, etc.
for efficient model refinement. It sits as an intermediate layer between these
core utilities and optimized models like MaxText and MaxDiffusion, streamlining
tuning workflows on top of the XLA and JAX infrastructure.

![Tunix in JAX ecosystem](images/tunix_in_jax_ecosystem.png)

See [Design Overview](design.md) for more details on the architecture.

## Key Features

-   **[Supervised Fine-Tuning (SFT)](algorithms.md)**:
    -   Full Weights Fine-Tuning
    -   [PEFT](performance.md#peft-with-lora) (Parameter-Efficient
        Fine-Tuning)
    -   [DPO](https://arxiv.org/abs/2305.18290) (Direct Preference Optimization)
      -   [ORPO](https://arxiv.org/abs/2403.07691) (Odds ratio Preference
          Optimization)
-   **[Reinforcement Learning (RL)](algorithms.md)**:
    -   [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization)
    -   [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy
        Optimization)
      -   [GSPO-Token](https://arxiv.org/abs/2507.18071) (Token-level Group
          Sequence Policy Optimization)
      -   [DAPO](https://arxiv.org/abs/2503.14476) (Direct Alignment via Preference
          Optimization)
      -   [Dr.GRPO](https://arxiv.org/abs/2503.14476) (Distributionally Robust
          GRPO)
-   **[Agentic RL](agentic_rl.md)**:
    -   Multi-turn tool use
    -   Asynchronous rollout for high-throughput trajectory collection
    -   Trajectory batching and grouping

## Framework & Infra Highlights

-   **Modularity**:
    -   Components are designed to be reusable and composable
    -   Easy to customize and extend
-   **Performance & Efficiency**:
    -   Native [vLLM](rollout.md#vllm) and
        [SGLang-JAX](rollout.md#sglang) on TPU integration for performant
        rollout
    -   Native [Maxtext](https://github.com/AI-Hypercomputer/maxtext) model
        integration for high performance kernels and model execution
    -   [Micro-batching](performance.md#batching-config) support for component
        level efficient execution
-   **Stability**
    -   Seamless multi-host distributed training with Pathways which can scale up
        to thousands of devices
    -   [Checkpointing and Fault Tolerance](reliability.md)

## Get Started

Jump to [Quick Start](quickstart.md) to install Tunix and run your first training
job.

## Supported Models

Tunix supports a growing list of models including Gemma, Llama, and Qwen families.
See [Models](models.md) for a full list and details on how to add new ones.

## Citing Tunix

```bibtex
@misc{tunix2025,
  title={Tunix (Tune-in-JAX)},
  author={Bao, Tianshu and Carpenter, Jeff and Chai, Lin and Gao, Haoyu and Jiang, Yangmu and Noghabi, Shadi and Sharma, Abheesht and Tan, Sizhi and Wang, Lance and Yan, Ann and Yu, Weiren and et al},
  year={2025},
  howpublished={\url{https://github.com/google/tunix}},
}
```

## Acknowledgements

Thank you to all our wonderful contributors!

[![Contributors](https://contrib.rocks/image?repo=google/tunix)](https://github.com/google/tunix/graphs/contributors)
