# Tunix: A lightweight LLM Post-Training Library

<div align="left">

<a href="https://tunix.readthedocs.io/en/latest/index.html"><img src="https://img.shields.io/badge/documentation-blue"></a>

</div>

**Tunix(Tune-in-JAX)** is a JAX based library designed to streamline the
post-training of Large Language Models. It provides efficient and scalable
supports for:

- **SOTA Training performance on TPUs**
- **Supervised Fine-Tuning**
- **Reinforcement Learning (RL)**
- **Knowledge Distillation**

Tunix leverages the power of JAX for accelerated computation and seamless
integration with JAX-based modeling framework
[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

**Current Status: Alpha Release**

Tunix is under active development. Our team is actively working on expanding its
capabilities, usability and performance. Stay tuned for upcoming updates and new
features!

## Key Features

- **Supervised Fine-Tuning:**
  - Full Weights Fine-Tuning
  - Parameter-Efficient Fine-Tuning (PEFT) with LoRA/Q-LoRA Layers
- **Reinforcement Learning (RL):**
  - Proximal Policy Optimization (PPO)
  - Group Relative Policy Optimization (GRPO)
  - Token-level Group Sequence Policy Optimization (GSPO-token)
- **Preference Fine-Tuning:**
  - Preference alignments with Direct Preference Optimization (DPO)
- **Knowledge Distillation:**
  - Logit Strategy: A classic approach where the student learns to match the
    teacher's output probability distribution.
  - Attention Transfer & Projection Strategies: Methods to align the attention
    mechanisms between the student and teacher models.
  - Feature Pooling & Projection Strategies: General techniques for matching
    intermediate feature representations, even between models of different
    architectures.

## Framework & Infra Highlights

- **Modularity:**
  - Components are designed to be reusable and composable
  - Easy to customize and extend
- **Performance & Efficiency:**
  - Native [vLLM](https://github.com/vllm-project/tpu-inference) and [SGLang](https://github.com/sgl-project/sglang-jax) on TPU integration for performant rollout
  - Native [Maxtext](https://github.com/AI-Hypercomputer/maxtext) model integration for high performance kernels and model execution
  - Micro-batching support for component level efficient execution
- **Stability**
  - Seamless multi-host distributed training with Pathways which can scale up to thousands of devices

## Upcoming/Experimental Features

- **Agentic RL Training:**
  - Async Rollout
  - Multi-turn & multi-step support
  - Tool usage
- **Advanced Algorithms:**
  - Additional state-of-the-art RL and distillation algorithms
- **User Guides:**
  - More advanced RL recipe

## Installation

- You can install Tunix in several ways:

1. From PyPI (recommended):

```sh
pip install "google-tunix[prod]"
```

2. Directly from GitHub (latest main branch)

```sh
pip install git+https://github.com/google/tunix
```

3. From source (editable install) If you plan to modify the codebase and run it
   in development mode. If you'd like to install vllm, the tpu-inference
   supported version is not released yet, please follow the instructions to
   install manually
   (https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/)
   or download the docker image (vllm/vllm-tpu:v0.11.1) then
   `pip install tpu-inference` for TPU backend:

```sh
git clone https://github.com/google/tunix.git
cd tunix
pip install -e ".[dev]"

# Then install vLLM and tpu-inference
```

- Using tunix with SGLang-Jax rollout

1. Install tunix using above ways
1. Then install SGLang-Jax

```
git clone git@github.com:sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
```

## Getting Started

To get started, we have a bunch of detailed examples and tutorials.

- [PEFT Gemma with QLoRA](https://github.com/google/tunix/blob/main/examples/qlora_gemma.ipynb)
- [Training Gemma on grade school Math problems using GRPO](https://github.com/google/tunix/blob/main/examples/grpo_gemma.ipynb)
- [Logit Distillation using Gemma models](https://github.com/google/tunix/blob/main/examples/logit_distillation.ipynb)
- [Training Llama3 or Qwen2 using GRPO and SGLang-Jax rollout](https://github.com/google/tunix/blob/main/scripts/grpo_demo_sglang_jax_rollout.py)

To setup Jupyter notebook on single host GCP TPU VM, please refer to the
[setup script](https://github.com/google/tunix/blob/main/scripts/setup_notebook_tpu_single_host.sh).

For the AI-powered interactive documentation, please check out [Tunix on Google 
CodeWiki](https://codewiki.google/github.com/google/tunix).

## Contributing and Feedbacks

We welcome contributions! As Tunix is in early development, the contribution
process is still being formalized. A rough draft of the contribution process is
present [here](https://github.com/google/tunix/blob/main/CONTRIBUTING.md). In
the meantime, you can make feature requests, report issues and ask questions in
our
[Tunix GitHub discussion forum](https://github.com/google/tunix/discussions).

## Collaborations and Partnership

[GRL](https://github.com/lmgame-org/GRL/blob/tunix_integration_dev/README.md)
(Game Reinforcement Learning), developed by
[Hao AI Lab](https://hao-ai-lab.github.io/) from UCSD, is an open-source
framework for post-training large language models through multi-turn RL on
challenging games. In collaboration with Tunix, GRL integrates seamless TPU
support—letting users quickly run scalable, reproducible RL experiments (like
PPO rollouts on Qwen2.5-0.5B-Instruct) on TPU v4 meshes with
[minimal setup](https://github.com/lmgame-org/GRL/blob/tunix_integration_dev/README.md#5-launch-the-quick-test-defaults-to-qwen2505b-supports-4-tpu-v4-with-mesh-22).
This partnership empowers the community to push LLM capabilities further,
combining Tunix’s optimized TPU runtime with GRL’s flexible game RL pipeline for
cutting-edge research and easy reproducibility.

## Stay Tuned!

Thank you for your interest in Tunix. We're working hard to bring you a powerful
and efficient library for LLM post-training. Please follow our progress and
check back for updates!

## Citing Tunix

```bibtex
@misc{tunix2025,
  title={Tunix},
  author={Bao, Tianshu and Wang, Lance and Sharma, Abheesht and Shin, Jiwon and
  Yan, Ann and Tan, Sizhi and Gao, Haoyu and Ha, Jen and Chai, Lin and
  Liu, Dangyi and Iyer, Rakesh and Sahu, Mridul and et al.},
  year={2025},
  howpublished={\url{https://github.com/google/tunix}},
}
```

## Acknowledgements

Thank you to all our wonderful contributors!

[![Contributors](https://contrib.rocks/image?repo=google/tunix)](https://github.com/google/tunix/graphs/contributors)
