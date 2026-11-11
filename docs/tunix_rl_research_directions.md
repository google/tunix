# Tunix RL Development Directions

Sources checked through **November 9, 2026**. This roadmap starts with broad areas, then lists **feature milestones, observable acceptance outcomes, and references**. Swarm architecture and recipe selection were updated on November 10, 2026: Dr. MAS shared-policy math first, frozen-worker orchestration next, then Avalon-inspired dynamic parallelism. Paper and code availability are recorded in the [research reference index](#research-reference-index).

| Area | Opportunities | Representative references |
| --- | --- | --- |
| **1. OPD and Distillation** | Extend OPD with feedback, hindsight, roles, and multiple teachers. | [SDPO](https://arxiv.org/abs/2601.20802), [SEED](https://arxiv.org/abs/2607.14777), [SRD](https://arxiv.org/abs/2610.08077), [MAS-OPD](https://arxiv.org/abs/2609.34234) |
| **2. Swarms: Multi-Agent Collaborative RL** | Reusable execution/binding/learning contracts; shared-policy math first, then frozen-worker orchestration and dynamic parallel teams. | [Dr. MAS](https://github.com/langfengQ/DrMAS), [RLCR](https://www.gr.inc/avalonm1.pdf), [PARL](https://www.kimi.com/en/blog/kimi-k2-5), [AT-GRPO](https://arxiv.org/abs/2510.11062), [C3](https://arxiv.org/abs/2603.06859) |
| **3. RSI: Recursive Self-Improvement** | Automatic curricula, skill evolution, team experience, workflow/code improvement. | [R²Orch](https://arxiv.org/abs/2610.22740), [Skill-SP](https://arxiv.org/abs/2607.22529), [MASS](https://arxiv.org/abs/2610.12176), [SwarmRSI](https://arxiv.org/abs/2610.20431) |
| **4. Multimodal: Understanding and Interactive RL** | Image/audio/video inputs, visual tools, multimodal OPD. | [VAGEN](https://arxiv.org/abs/2510.16907), [Miles GEO3K VLM](https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm) |
| **5. Diffusion: Image/Video Generation RL** | Continuous diffusion/flow training, generation rewards, sampling objectives. | [Flow-GRPO](https://arxiv.org/abs/2505.05470), [DiffusionNFT](https://arxiv.org/abs/2509.16117), [Miles Diffusion](https://github.com/radixark/miles_diffusion) |
| **6. Long-Horizon Agents** | Long tool interactions, memory, segment/milestone credit. | [Mem-α](https://arxiv.org/abs/2509.25911), [BEACON](https://arxiv.org/abs/2605.06078), [Agent Lightning](https://arxiv.org/abs/2508.03680) |
| **7. Environments and World Models** | Executable environment generation, co-evolution, transition prediction. | [AWM](https://arxiv.org/abs/2602.10090), [GenEnv](https://arxiv.org/abs/2512.19682), [EvoEnv](https://arxiv.org/abs/2605.14392), [RWML](https://arxiv.org/abs/2602.05842) |
| **8. Rewards and Verification** | Outcome/process rewards, generator/verifier learning, tests and formal proofs. | [MAPPA](https://arxiv.org/abs/2601.23228), [V₁](https://arxiv.org/abs/2603.04304), [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801), [SWE-RL](https://arxiv.org/abs/2502.18449) |
| **9. Test-Time Learning and Adaptation** | Task-local updates, feedback adaptation, search/training budgets. | [TTT-Discover](https://arxiv.org/abs/2601.16175), [SEAL](https://arxiv.org/abs/2506.10943) |
| **10. RL Algorithms and Sample Efficiency** | Objectives/advantages, replay and off-policy correction, learned update rules. | [DistIL](https://arxiv.org/abs/2606.05152), [Experience Replay](https://arxiv.org/abs/2604.08706), [DiscoRL](https://google-deepmind.github.io/disco_rl/) |
| **11. RL Systems and Scalable Training** | Asynchrony, resource scheduling, weight synchronization, multi-policy/LoRA services, recovery. | [RLinf](https://arxiv.org/abs/2509.15965), [FlexMARL](https://arxiv.org/abs/2602.09578), [Belayer](https://arxiv.org/abs/2608.14635), [Miles](https://github.com/radixark/miles) |

**Near-term priority: OPD → Swarms → RSI.** Other areas follow as optional extensions.

**How to read this roadmap:** each row is a user-visible milestone demonstrated by a recipe, example, or evaluation report. Stages progress from foundations to extensions; audio, video, and OPD branches can proceed in parallel. DTOs, masks, and scorers are implementation tasks rather than standalone features. Existing capabilities are labeled where recipe maintenance or hardware acceptance remains. These are proposed outcomes, not claims of completed support. Miles references use inspected fixed revisions; see the [reference index](#research-reference-index) for availability boundaries.

## 1. OPD and Distillation

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| KD1 | **Fixed-teacher text OPD acceptance** | The Qwen3-8B←32B recipe already exists; add actual teacher/TPU execution, recovery, and before/after quality reports. | [Existing Tunix recipe](../tunix/dist_rl/examples/on_policy_distillation/QWEN3_8B_OPD.md), [Miles OPD][m-opd-example] |
| KD2 | **Candidate-distribution distillation mode** | Select sampled-token or top-k distillation and compare quality and teacher cost with the same model pair/task. | [Miles teacher scoring][m-teacher], [OPD loss][m-opd-loss] |
| KD3 | **Task-based multi-teacher routing** | Select specialist teachers by sample type in one run; record each teacher's usage, cost, and outcomes. | [Miles multi-teacher OPD examples][m-opd-example] |
| KD4 | **Experience/feedback self-distillation recipe** | Use error feedback, successful demonstrations, or hindsight experience to improve the policy, with ordinary RL as a control. | [SDPO](https://github.com/lasgroup/SDPO), [SEED](https://github.com/jinyangwu/SEED) |
| KD5 | **Multi-turn tool/role distillation recipe** | Distill on tool interactions or multi-agent role tasks; compare role quality, team performance, and scoring cost. | [MAS-OPD method](https://arxiv.org/abs/2609.34234), [Miles agentic session][m-session] |

<a id="swarms-rlcr"></a>

## 2. Swarms: Multi-Agent Collaborative RL

Implementation checkpoint (November 11, 2026): the shared Dr. MAS math slice now has local numerical and training validation. Distributed transport and pretrained 4B quality remain the next gate. [Recipe](../examples/swarm_math/README.md) · [Validation](swarm_training_validation.md).

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| SW1 | **Conditional shared-policy collaboration** | Shared Qwen3-4B Solver → Verifier → optional Solver revision; export exact per-call traces and rule-based final scores. | [Dr. MAS workflow](https://github.com/langfengQ/DrMAS) |
| SW2 | **First verifiable multi-agent RL recipe** | Reference-aligned Dr. MAS/global normalization; train/sync/resume; locked-test quality and cost against untrained teams and matched-budget single-agent RL. | [Dr. MAS launcher](https://github.com/langfengQ/DrMAS/blob/master/examples/drmas_trainer/run_math.sh) |
| SW3 | **Trainable orchestrator with frozen workers** | Change workflow and policy binding without rewriting the trainer; verify frozen parameters do not update. | [PARL](https://www.kimi.com/en/blog/kimi-k2-5), [MARTI](https://github.com/TsinghuaC3I/MARTI) |
| SW4 | **Dynamic parallel, time-budgeted teams** | Dynamic members, messages/workspace, anytime scoring, physical deadlines and prefix-preserving abort; compare quality/latency at matched budgets. | [RLCR][rlcr-blog], [Swarm-Repair design](swarm_rlcr_design.md) |
| SW5 | **Optional: independent-role policy co-training** | Per-policy routing, versions, optimizers, synchronization and checkpoint bundles; restore all policies consistently. | [Dr. MAS](https://github.com/langfengQ/DrMAS), [Miles multi-policy][m-multipolicy] |
| SW6 | **Optional: teacher, preference and counterfactual learning** | Role OPD with actor/teacher context separation; offline preference generation; environment forks and recipe-specific credit. | [MAS-OPD](https://arxiv.org/abs/2609.34234), [MALT](https://arxiv.org/abs/2412.01928), [MPT](https://github.com/Embodied-Minds-Lab/MPT) |

Common contracts separate interaction, parameter binding, and learning. Static workflows, conditional loops, and dynamic swarms use the same episode/call trace and trainer integration. Shared files and deadlines are recipe capabilities. See the [formal system design](swarm_training_design.md).

### 2.1 Focus reference: How to Train a Swarm / Avalon-M1

Primary sources: [How to Train a Swarm (blog)](https://www.gr.inc/blog/how-to-train-a-swarm); [Avalon-M1: Reinforcement Learning from Collective Rewards (report)](https://www.gr.inc/avalonm1.pdf) ([original PDF URL](https://gr.inc/avalonm1.pdf)).

**The focus is training the whole team to organize computation and collaborate.** The CEO and specialists share policy parameters, and team returns produce a common collective advantage. [Method source](https://www.gr.inc/avalonm1.pdf)

| Aspect | Mechanism to study |
| --- | --- |
| Shared weights | Role instances share a policy and jointly contribute training behavior. |
| Collective advantage | Compare team rollouts for the same task and apply one team advantage to each agent's valid generated tokens. |
| Communication | The CEO can hire/fire; members coordinate through direct messages, channels, and a shared workspace. |
| Anytime return | Score intermediate task progress when execution stops. |
| Time pressure | Impose wall-clock limits and progressively longer horizon curricula to make useful parallel work valuable. |
| Objective | Most environments use CISPO with group-relative advantages; some long-term-return environments use GKPO. |

Method details follow [report §2](https://www.gr.inc/avalonm1.pdf); time pressure, anytime returns, and communication follow the [blog](https://www.gr.inc/blog/how-to-train-a-swarm). Equal-return rollouts completed before the deadline do not receive different rewards solely because one finishes earlier: a deadline is not a per-second latency bonus.

### 2.2 Initial Tunix Recipe and Delivery Order

Read the [one-pager](swarm_training_onepager.md), [English overview](swarm_training_overview.md), [Chinese overview](swarm_training_overview_zh.md), and [formal system design](swarm_training_design.md). The first recipe uses shared Qwen3-4B math with K=8 complete team attempts, a conditional Solver–Verifier workflow, and external rule-based scoring. Reference starting learning rate is `1e-6`; deployment and sampling details require a pinned source and actual validation.

Port the active-call role statistics and the global baseline's episode-level aggregation separately. Keep credit scope, statistical unit, and loss weighting explicit; Tunix GRPO defaults are not a faithful port by themselves. Reserve development data from training and lock the test set. Publish baseline/final quality, cost, seed uncertainty, numerical parity, and real accelerator sync/restore evidence.

Milestones: **SW1 conditional execution → SW2 first math result → SW3 frozen-worker orchestration → SW4 dynamic parallel Swarm-Repair**. SW5/SW6 add independent policy and learning capabilities; RSI task/skill generation stays in an outer versioned curriculum controller.

The [Swarm-Repair specification](swarm_rlcr_design.md) retains Team-GRPO, communication/workspace, bounded dynamic membership, physical deadlines and distributed train/eval/resume as a later specialization. Its fixed team-capacity denominator does not apply to every recipe. Native Team-GRPO is an adaptation of RLCR, whose original main objective is CISPO.

### 2.3 Publication Evidence

The first report should document a reproducible multi-agent math recipe: pinned model/data/prompts, the conditional workflow, normalization semantics, numerical parity, locked-test controls, costs and limitations. Quality or stability improvement must be established experimentally. This result does not establish dynamic parallelism or a latency speedup.

The later Avalon-inspired report can study shared weights, communication, anytime progress and physical deadlines, using matched-budget controls and actual execution traces. Keep the source conditions when citing **3.9×**: it is reported average **derived latency** at matched output-token budgets and comparable scores, not training throughput or a guaranteed wall-clock gain. [Avalon report](https://www.gr.inc/avalonm1.pdf)

## 3. RSI: Recursive Self-Improvement

Task generation, skill evolution and self-play selection run outside the swarm episode collector. Each cycle publishes versioned task/skill manifests and an explicit learning plan; trainable Proposer/Director policies require the corresponding policy lifecycle support. See the [common system design](swarm_training_design.md#7-extending-to-different-swarms).

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| RS1 | **Verification-driven self-training loop** | Generate attempts, filter them through verification, update, and repeat; retain per-round data, models, and held-out evaluations. | [Miles two-phase self-distillation][m-selfdistill], [MASS](https://github.com/SakanaAI/mass) |
| RS2 | **Automatic task generation and adaptive curricula** | Generate/select tasks according to model capability; report difficulty distributions and progress on a fixed test set. | [R-Zero](https://github.com/Chengsong-Huang/R-Zero), [GenEnv](https://github.com/Gen-Verse/GenEnv) |
| RS3 | **Skill-augmented self-improving agent** | Extract, reuse, and update skills from experience; compare no-skill, static-skill, and evolving-skill configurations. | [SkillRL](https://github.com/aiming-lab/SkillRL), [Skill-SP](https://github.com/Qwen-Applications/skill-self-play) |
| RS4 | **Multi-cycle self-improvement experiment recipe** | Run several improvement cycles, select models/workflows using fixed evaluation, and report transfer, forgetting, and cumulative cost. | [MASS](https://github.com/SakanaAI/mass), [R²Orch](https://arxiv.org/abs/2610.22740) |

## 4. Multimodal: Understanding and Interactive RL

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| MM1 | **Single-image understanding RL recipe** | Use one fixed VLM and visual-QA task with training, save/restore, and baseline/final evaluation; initially freeze the vision encoder. | [Miles GEO3K recipe][m-vlm-run] |
| MM2 | **Multi-image understanding RL** | Accept several images per question and provide comparison/joint-reasoning training and evaluation examples. | [Miles VLM][m-vlm], [Media processing][m-media] — Specific multi-image recipes require separate validation. |
| MM3 | **Multi-turn visual tool-agent recipe** | Let the model inspect images, invoke crop/screenshot tools, read new images, and continue acting; train and evaluate this interaction loop. | [Miles GEO3K multi-turn][m-vlm-turn] |
| MM4 | **Audio understanding RL recipe** | Train on transcription or audio QA with audio input, text output, and a fixed test-set report. | [Tunix Gemma4 inference example](../tunix/models/gemma4/sampling_example.ipynb) — Inference foundation exists; RL integration is proposed. |
| MM5 | **Video understanding RL recipe** | Train on fixed-frame/fixed-duration video QA and provide evaluation examples sensitive to temporal order. | [Miles media components][m-media] — Component reference only, not an existing complete video RL recipe. |
| MM6 | **Multimodal OPD recipe** | Sample student responses on image tasks and obtain teacher supervision for the same media and responses; compare task RL with distillation. | [Miles VLM][m-vlm], [Miles OPD][m-opd-example] — The combined feature needs independent validation. |

## 5. Diffusion: Image/Video Generation RL

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| DF1 | **Image-generation SFT recipe** | Fine-tune a fixed continuous-generation model on image/text data, save it, and produce comparison samples. | [Miles Diffusion model backend][md-model], [SFT entry point](https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/train_sft.py) |
| DF2 | **Text-to-image RL recipe** | Train with OCR or image/text preference rewards; report before/after images, rewards, and costs. | [Miles Diffusion rollout][md-rollout], [reward hub][md-reward] |
| DF3 | **Selectable image RL modes** | Choose Flow-GRPO or NFT for the same model/task and compare quality and sampling cost; extend to LoRA afterward. | [Miles Flow-GRPO][md-flow], [NFT][md-nft] |
| DF4 | **Text-to-video RL recipe** | Train and evaluate on fixed-duration video tasks; report temporal/task rewards and generation cost. | [Miles Diffusion](https://github.com/radixark/miles_diffusion/tree/221f965fa68e4d865665a48e8478932a231dc4ad) |

## 6. Long-Horizon Agents

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| LH1 | **Agent with trainable memory management** | Learn what to retain, retrieve, and update during long interactions; report results against a no-memory baseline. | [Mem-α](https://github.com/wangyu-ustc/Mem-alpha) |
| LH2 | **Resumable long-horizon tool tasks** | Resume interrupted tasks at consistent boundaries while preserving files, memory, tool state, and model version. | [Miles session][m-session], [Tunix DeepSWE](../tunix/dist_rl/examples/deepswe_dist) |
| LH3 | **Process-reward long-horizon RL recipe** | Provide milestone or correct-segment training signals and compare with final-outcome-only rewards. | [BEACON](https://github.com/ZJU-REAL/BEACON) |
| LH4 | **Long-horizon evaluation suite** | Report progress, success rate, tool/token cost, and post-recovery outcomes across time budgets. | [Mem-α](https://github.com/wangyu-ustc/Mem-alpha), [Tunix DeepSWE evaluation](../tunix/dist_rl/examples/deepswe_dist) |

## 7. Environments and World Models

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| ENV1 | **External tool-environment training recipe** | Train/evaluate in an external stateful environment without rewriting the core training loop for each environment. | [Miles OpenEnv][m-openenv], [AWM](https://github.com/Snowflake-Labs/agent-world-model) |
| ENV2 | **Executable synthetic environment package** | Provide replayable tool/database tasks, graders, and fixed train/test splits for agent RL. | [AWM](https://github.com/Snowflake-Labs/agent-world-model) |
| ENV3 | **Automatic task/environment evolution** | Generate and validate environments, select tasks for current capability, and record environment quality and curriculum benefits. | [GenEnv](https://github.com/Gen-Verse/GenEnv), [EvoEnv](https://arxiv.org/abs/2605.14392) |
| ENV4 | **World-model-assisted training experiment** | Explore or train with a learned transition model and evaluate simulation-to-real transfer in actual environments. | [RWML](https://arxiv.org/abs/2602.05842) — Public training code was not verified. |

## 8. Rewards and Verification

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| RW1 | **Composable reward training configuration** | Combine task correctness, format, cost, and other rewards; record component scores and training outcomes. | [Miles reward hub][m-reward] |
| RW2 | **Verifiable-task RL recipe** | Provide a complete code-test or formal-proof example spanning generation, verification, training, and evaluation. | [Miles Harbor][m-harbor], [DeepSeek-Prover-V2](https://github.com/deepseek-ai/DeepSeek-Prover-V2) — Proof/verification/inference reference. |
| RW3 | **Process/team reward training modes** | Switch between outcome, process, and role rewards on the same task; compare quality and scoring overhead. | [MAPPA](https://github.com/ltjed/multiagent-coaching), [BEACON](https://github.com/ZJU-REAL/BEACON) |
| RW4 | **Generator/verifier co-learning** | Train and use a verifier to assess candidates; report verification accuracy, calibration, and final task performance. | [V₁ paper](https://arxiv.org/abs/2603.04304), [Miles solver/verifier][m-multipolicy] — The inspected V₁ repository exposes inference; training code was not verified. |

## 9. Test-Time Learning and Adaptation

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| TT1 | **Single-task test-time RL example** | Perform budgeted policy updates for the current task; return the best solution, adaptation checkpoint, and total cost. | [TTT-Discover](https://github.com/test-time-training/discover) |
| TT2 | **Feedback-driven rapid adaptation** | Use test/environment feedback to construct update data for new tasks and compare pre/post-adaptation performance. | [SDPO](https://github.com/lasgroup/SDPO), [SEAL](https://arxiv.org/abs/2506.10943) |
| TT3 | **Joint training/search budget modes** | Allocate budgets among sampling, feedback, and updates; compare solution-discovery probability at fixed cost. | [TTT-Discover](https://github.com/test-time-training/discover) |
| TT4 | **Continual task-stream adaptation experiment** | Adapt over successive tasks, with reset or retained updates between tasks; report transfer and forgetting. | [SEAL method](https://arxiv.org/abs/2506.10943) — Continual task-stream support is a proposed extension. |

## 10. RL Algorithms and Sample Efficiency

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| ALG1 | **Configurable objective-comparison recipe** | Switch integrated objectives and advantage estimators on the same model/task; produce comparable quality/cost reports. | [Miles loss hub][m-loss], [DistIL](https://github.com/rishabh-1086/distIL) |
| ALG2 | **Filtered-sampling RL mode** | Use validated top-k/top-p configurations in recipes with consistent sampling/training distribution semantics. | [Miles sampling-support replay][m-support] |
| ALG3 | **Sample-reuse training mode** | Configure fresh/replayed trajectory mixtures and reuse budgets; compare learning at equal generation cost, initially for ordinary task RL. | [Miles async][m-async], [Experience Replay](https://arxiv.org/abs/2604.08706) |
| ALG4 | **Training/inference consistency acceptance recipe** | For a supported fixed model combination, report numerical differences among sampling, rescoring, and training, including correction effects. | [Miles true-on-policy][m-top], [mismatch helper][m-mismatch] |

## 11. RL Systems and Scalable Training

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| SYS1 | **Standardized launch and reproduction recipes** | Run training, evaluation, save/restore, and experiment recording from fixed configuration; extend the existing OPD approach. | [Tunix OPD recipe](../tunix/dist_rl/examples/on_policy_distillation/recipe.py), [Miles examples][m-examples] |
| SYS2 | **Configurable asynchronous training mode** | Run rollout/training concurrently for an allocation and report useful throughput, utilization, and policy staleness. | [Miles fully async][m-async], [RLinf](https://github.com/RLinf/RLinf) |
| SYS3 | **Automatic recovery after worker failure** | Rebuild failed workers and restore service; use controlled failures to report data/version consistency and lost time. | [Miles fault tolerance][m-ft], [P2P transfer][m-p2p] |
| SYS4 | **Independent multi-policy training service** | Run and co-train several policies with separate optimizers, model versions, and recovery points. | [Miles multi-policy][m-multipolicy] |
| SYS5 | **Multi-tenant LoRA training service** | Let clients share a base model while managing separate adapters, training state, and versioned sampling. | [Miles multi-LoRA / Tinker][m-multilora] |
| SYS6 | **Independent asynchronous evaluation service** | Submit evaluations for fixed policy versions during training; generate results on separate workers and aggregate historical metrics. | [Miles checkpoint evaluation][m-eval] |

These areas can combine—for example multimodal swarms, swarm OPD, or diffusion self-distillation—after their foundations are validated separately. RSI includes weight training and nonparametric evolution: MASS uses search plus SFT, while SwarmRSI primarily evolves programs and experimental state.

Tunix already has OPD, asynchronous scheduling, and multimodal model components. See the [capability checklist](tunix_capability_checklist.md) for current coverage and the [OPD guide](../tunix/dist_rl/examples/on_policy_distillation/QWEN3_8B_OPD.md) for the existing recipe.

<a id="research-reference-index"></a>

## 12. Research Reference Index

Sources and code availability were checked through November 9, 2026. Training was not reproduced. The entries below preserve the distinction between training implementations, inference-only releases, partial code, and unverified availability.

| ID | Paper / project | Code availability and scope |
| --- | --- | --- |
| R01 | **SDPO: Reinforcement Learning via Self-Distillation**; [Paper](https://arxiv.org/abs/2601.20802); 2026-01 | [Author repository](https://github.com/lasgroup/SDPO). Training implementation. Same-model self-distillation requires extensions to the fixed-teacher contract. |
| R02 | **DistIL: Reinforcement Learning from Rich Feedback with Distributional DAgger**; [Paper](https://arxiv.org/abs/2606.05152); 2026-06 | [Author repository](https://github.com/rishabh-1086/distIL). Author implementation. Improvement guarantees depend on assumptions and do not automatically cover arbitrary teachers or approximations. |
| R03 | **SEED: Self-Evolving On-Policy Distillation for Agentic RL**; [Paper](https://arxiv.org/abs/2607.14777); 2026-07 | [Author repository](https://github.com/jinyangwu/SEED). Two-stage training scripts; the analyzer can be removed at deployment. The objective differs from existing Tunix OPD. |
| R04 | **SRD: Self-Retrospection Distillation**; [Paper](https://arxiv.org/abs/2610.08077); 2026-10 | [Author repository](https://github.com/SalesforceAIResearch/SRD). Salesforce training/evaluation configurations. Recent preprint; explicit foresight generation is not required during inference. |
| R05 | **Co-rewarding: Stable Self-supervised RL**; [Paper](https://arxiv.org/abs/2508.00410); 2025-08 | [Author repository](https://github.com/tmlr-group/Co-rewarding). Code and data. Label-free self-rewards are not necessarily objectively verifiable. |
| R06 | **R-Zero: Self-Evolving Reasoning LLM from Zero Data**; [Paper](https://arxiv.org/abs/2508.05004); 2025-08 | [Author repository](https://github.com/Chengsong-Huang/R-Zero). Training scripts. Zero-data training still relies on pretrained capability; pseudo-labels can be wrong. |
| R07 | **Tool-R0: Self-Evolving Tool-Learning Agents**; [Paper](https://arxiv.org/abs/2602.21320); 2026-02 | [Author repository](https://github.com/emrecanacikgoz/Tool-R0). Actual training entry points exist despite a Coming Soon project-page label; documentation is insufficient for a complete reproduction recipe. |
| R08 | **SPELL: Self-Play RL for Evolving Long-Context Language Models**; [Paper](https://arxiv.org/abs/2509.23863); 2025-09 | [Author repository](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/SPELL). Training and evaluation code. Raw documents are still required; semantic verification is not executable ground truth. |
| R09 | **GenEnv: Difficulty-Aligned Agent/Environment Co-Evolution**; [Paper](https://arxiv.org/abs/2512.19682); 2025-12 | [Author repository](https://github.com/Gen-Verse/GenEnv). Training implementation. The two sides do not both use GRPO, and seed data is still used. |
| R10 | **EvoEnv: Learning to Build the Environment**; [Paper](https://arxiv.org/abs/2605.14392); 2026-05 | Official public code was not verified. The report starts with 10 seed environments and fixes grading logic after acceptance. |
| R11 | **AgentEvolver: Towards Efficient Self-Evolving Agent System**; [Paper](https://arxiv.org/abs/2511.10395); 2025-11 | [Author repository](https://github.com/modelscope/AgentEvolver). ModelScope framework, specifically arXiv 2511.10395; distinct from the same-name fixed-backbone project. |
| R12 | **SkillRL: Recursive Skill-Augmented RL**; [Paper](https://arxiv.org/abs/2602.08234); 2026-02 | [Author repository](https://github.com/aiming-lab/SkillRL). Training implementation. External skills and weights co-evolve; this does not establish skill-free deployment. |
| R13 | **Skill-SP: Skill Self-Play**; [Paper](https://arxiv.org/abs/2607.22529); 2026-07 | [Author repository](https://github.com/Qwen-Applications/skill-self-play). Qwen training scripts for tool use and logic tasks; the final solver does not require skill prompts. |
| R14 | **Mem-α: Learning Memory Construction via RL**; [Paper](https://arxiv.org/abs/2509.25911); 2025-09 | [Author repository](https://github.com/wangyu-ustc/Mem-alpha). Author implementation. Trains memory management, rather than storing the entire interaction history in model weights. |
| R15 | **TTT-Discover: Learning to Discover at Test Time**; [Paper](https://arxiv.org/abs/2601.16175); 2026-01 | [Author repository](https://github.com/test-time-training/discover). Author code; original training depends on Tinker. Separate single-problem adaptation from cross-task generalization. |
| R16 | **V₁: Unifying Generation and Self-Verification for Parallel Reasoners**; [Paper](https://arxiv.org/abs/2603.04304); 2026-03 | [Author repository](https://github.com/HarmanDotpy/pairwise-self-verification). The inspected repository provides V₁-Infer; PairRL training code was not verified. Paper training still uses ground-truth tests. |
| R17 | **BookNavigator: Incentivizing Self-Learning via RL**; [Paper](https://arxiv.org/abs/2610.39671); 2026-10 | Recent preprint; official code was not verified. Learning an in-context study strategy is distinct from online deployment-time weight updates. |
| S01 | **AT-GRPO / Stronger-MAS**; [Paper](https://arxiv.org/abs/2510.11062); 2025-10 | [Author repository](https://github.com/pettingllms-ai/PettingLLMs). PettingLLMs training code. Start with shared-policy mode; shared, role-LoRA, and independent-policy modes are distinct. |
| S02 | **C3: The Trace Is the State**; [Paper](https://arxiv.org/abs/2603.06859); 2026-03 | [Author repository](https://github.com/EIT-EAST-Lab/C3). Author code; not all datasets/checkpoints are included. Theoretical claims require sufficient traces. Title follows v4, October 7, 2026. |
| S03 | **CCPO / SEPO: Counterfactual Credit for Collaboration**; [Paper](https://arxiv.org/abs/2603.21563); 2026-03 | [Author repository](https://github.com/bhai114/ccpo). Two credit-construction schemes in author code; chiefly a reward-layer extension rather than a replacement policy optimizer. |
| S04 | **MAPPA: Scaling Multiagent Systems with Process Rewards**; [Paper](https://arxiv.org/abs/2601.23228); 2026-01 | [Author repository](https://github.com/ltjed/multiagent-coaching). Training implementation. External coach cost and bias matter; process scores are not necessarily causal contributions. |
| S05 | **Puppeteer: Multi-Agent Collaboration via Evolving Orchestration**; [Paper](https://arxiv.org/abs/2505.19591); 2025-05 | [Author repository](https://github.com/OpenBMB/ChatDev/tree/puppeteer). Author branch. Primarily trains the orchestrator, not all worker LLMs jointly. |
| S06 | **MetaAgent-X: End-to-End RL for Automatic Multi-Agent Systems**; [Paper](https://arxiv.org/abs/2605.14212); 2026-05 | [Author repository](https://github.com/pettingllms-ai/PettingLLMs). PettingLLMs training code, including shared-policy co-training; not evidence of unbounded self-improvement. |
| S07 | **MACA: Post-Training on Multi-Agent Debate**; [Paper](https://arxiv.org/abs/2509.15172); 2025-09 | [Author repository](https://github.com/facebookresearch/maca). Meta author code. Primarily preference-based post-training, not uniformly online multi-policy RL; consensus can be incorrect. |
| S08 | **SocialDeductionLLM: Multi-Agent RL for Social Deduction**; [Paper](https://arxiv.org/abs/2502.06060); 2025-02 | [Author repository](https://github.com/SocialDeductionLLM/SocialDeductionLLM). Author code. Evidence comes from partially observable social games, not a universal collaboration solution. |
| S09 | **TalkMesh: RL of Communication in Small-Model Meshes**; [Paper](https://arxiv.org/abs/2609.30578); 2026-09 | Official source code was not verified. Confidence/talk adapters have distinct roles; this is not dozens of independent policies trained jointly. |
| S10 | **ACTS: Role-Aware Communication Topology Scheduling**; [Paper](https://arxiv.org/abs/2610.25851); 2026-10 | [Project site](https://acts-mas.github.io/ACTS/). The project site was inaccessible during this review; source code was not verified. Trains a controller with frozen LLMs. |
| S11 | **SHARP: Shapley Credit-based Optimization**; [Paper](https://arxiv.org/abs/2602.08335); 2026-02 | Author-verified code was not found. Additional sampling cost matters; do not assume cheap, exact causal attribution. |
| S12 | **FlexMARL: Graph-Driven Rollout-Training Co-Design**; [Paper](https://arxiv.org/abs/2602.09578); 2026-02 | Official code was not verified. Runtime system rather than a new loss. Uses v2, October 24; its listed November conference had not occurred at the review cutoff. |
| S13 | **IAPO: Interaction-Aware Multi-Agent Orchestration**; [Paper](https://arxiv.org/abs/2611.07495); 2026-11-05 | Recent preprint; official code was not verified. Theoretical guarantee assumptions need separate examination. |
| G01 | **BEACON: Milestone-Guided Policy Learning**; [Paper](https://arxiv.org/abs/2605.06078); 2026-05 | [Author repository](https://github.com/ZJU-REAL/BEACON). Training code. Requires verifiable environment predicates; not a task-independent automatic process reward model. |
| G02 | **Agent Lightning: Agent Training/Execution Decoupling**; [Paper](https://arxiv.org/abs/2508.03680); 2025-08 | [Author repository](https://github.com/microsoft/agent-lightning). Microsoft framework. General integration does not automatically establish correct credit assignment. |
| G03 | **AWM: Agent World Model**; [Paper](https://arxiv.org/abs/2602.10090); 2026-02 | [Author repository](https://github.com/Snowflake-Labs/agent-world-model). Snowflake code and environment assets. Generates executable environments rather than a neural transition simulator. |
| G04 | **RWML: Reinforcement World Model Learning**; [Paper](https://arxiv.org/abs/2602.05842); 2026-02 | Official training code was not verified. Uses v3, November 1; text/embedding similarity does not guarantee equivalent dynamics. |
| G05 | **Efficient RL Training for LLMs with Experience Replay**; [Paper](https://arxiv.org/abs/2604.08706); 2026-04 | [Published paper](https://proceedings.mlr.press/v306/arnal26a.html). Published paper; official implementation was not verified. First validate on ordinary task RL rather than relaxing current OPD's zero-staleness restriction. |
| G06 | **DeepSeek-Prover-V2: RL for Subgoal Decomposition**; [Paper](https://arxiv.org/abs/2504.21801); 2025-04 | [Author repository](https://github.com/deepseek-ai/DeepSeek-Prover-V2). Official models/proofs/inference repository; not a complete public RL training stack. |
| G07 | **SWE-RL: RL on Open Software Evolution**; [Paper](https://arxiv.org/abs/2502.18449); 2025-02 | [Author repository](https://github.com/facebookresearch/swe-rl). Meta prompts, rewards, and inference tools; not a complete trainer. Patch similarity is not functional correctness. |
| G08 | **VAGEN: World-Model Reasoning for Multi-Turn VLM Agents**; [Paper](https://arxiv.org/abs/2510.16907); 2025-10 | [Author repository](https://github.com/mll-lab-nu/VAGEN). Author code. Distinguish paper/legacy versions from later refactoring; judge quality is task-dependent. |
| G09 | **Flow-GRPO: Training Flow Matching Models via Online RL**; [Paper](https://arxiv.org/abs/2505.05470); 2025-05 | [Author repository](https://github.com/yifan123/flow_grpo). GPU reference implementation; distinct from Tunix's existing discrete-token diffusion CE. |
| G10 | **DiffusionNFT: Online Diffusion Reinforcement with Forward Process**; [Paper](https://arxiv.org/abs/2509.16117); 2025-09 | [Author repository](https://github.com/NVlabs/DiffusionNFT). NVIDIA implementation. Needs new model/objective contracts; efficiency depends on model, solver, and budget. |
| G11 | **RLinf: Flexible and Efficient Large-Scale RL**; [Paper](https://arxiv.org/abs/2509.15965); 2025-09 | [Author repository](https://github.com/RLinf/RLinf). Author framework. Scheduling ideas transfer; CUDA/NCCL implementations do not directly transfer to TPU. |
| G12 | **Belayer: Fault Tolerance for Agentic RL**; [Paper](https://arxiv.org/abs/2608.14635); 2026-08 | [Candidate repository](https://github.com/Zhou-jiecheng/Belayer). Candidate author repository; official reciprocal linkage was not verified. Some natural failures do not yet trigger automatic reruns. |
| E01 | **MASS: Multi-Agent Self-Supervision**; [Paper](https://arxiv.org/abs/2610.12176), [Team introduction](https://sakana.ai/mass/); 2026-10 | [Author repository](https://github.com/SakanaAI/mass). SakanaAI search-plus-SFT pipeline. Public comparison procedures partly differ from the paper; not direct team policy-gradient training. |
| E02 | **SwarmRSI: Evolving Scientific Agent Swarms**; [Paper](https://arxiv.org/abs/2610.20431); 2026-10 | [Author repository](https://github.com/EvoScientist/SwarmRSI). Author workflow code. Uses v3, November 2; evolves programs and experimental state, not an established LLM-weight swarm-RL algorithm. |
| E03 | **MetaSkill-Evolve: Two-Timescale Meta-Skill Evolution**; [Paper](https://arxiv.org/abs/2607.05297); 2026-07 | Official code was not verified. The paper explicitly freezes the backbone: nonparametric skill evolution. |
| E04 | **Hyperagents**; [Paper](https://arxiv.org/abs/2603.19461); 2026-03 | [Author repository](https://github.com/facebookresearch/Hyperagents). Meta code. Uses v2, November 1; evolves task/meta-agent programs rather than LLM weights through RL. |
| M01 | **DiscoRL: Discovering State-of-the-Art RL Algorithms**; [Project and paper](https://google-deepmind.github.io/disco_rl/); 2025 | [Author repository](https://github.com/google-deepmind/disco_rl). DeepMind minimal JAX harness and Disco103 weights. Evidence is primarily from classical RL, not a drop-in LLM GRPO replacement. |

<!-- Engineering references pinned to inspected revisions -->
[md-model]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/backends/fsdp_utils/model_backend.py
[md-rollout]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/rollout/sglang_diffusion_rollout.py
[md-prob]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/utils/sde_log_prob.py
[md-reward]: https://github.com/radixark/miles_diffusion/tree/221f965fa68e4d865665a48e8478932a231dc4ad/miles/rollout/rm_hub
[md-flow]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/ray/data_conversion_hub/flow_grpo.py
[md-nft]: https://github.com/radixark/miles_diffusion/blob/221f965fa68e4d865665a48e8478932a231dc4ad/miles/ray/data_conversion_hub/nft.py
[m-media]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/data/multimodal.py
[m-vlm]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm
[m-vlm-run]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm/run_geo3k_vlm.sh
[m-vlm-turn]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm/multi_turn
[m-multipolicy]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/multi_policy
[m-data]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/data_source.py
[m-filter]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/filter_hub
[m-selfdistill]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/on_policy_distillation/qwen3_5_35b_selfdistill
[m-opd-example]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/on_policy_distillation
[m-teacher]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/on_policy_distillation.py
[m-opd-loss]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss/hub/opd.py
[m-session]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/user-guide/agentic-rollout.md
[m-openenv]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/experimental/openenv
[m-reward]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/rm_hub
[m-harbor]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/swe-agent-harbor-docker
[m-loss]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss/hub
[m-support]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/sampling-support-replay.md
[m-async]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/fully_async
[m-top]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/true_on_policy
[m-mismatch]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/train_infer_mismatch_helper
[m-ft]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/fault-tolerance.md
[m-p2p]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/p2p_weight_transfer
[m-multilora]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/multi_lora
[m-eval]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout/checkpoint_eval.py
[m-examples]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/README.md
[rlcr-pdf]: https://www.gr.inc/avalonm1.pdf
[rlcr-blog]: https://www.gr.inc/blog/how-to-train-a-swarm
