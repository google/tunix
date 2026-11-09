# Tunix RL Development Directions

Sources checked through **November 9, 2026**. This roadmap starts with broad areas, then lists **feature milestones, observable acceptance outcomes, and references**. Swarms emphasizes Avalon-M1 / RLCR and includes a proposed blog outline. Paper and code availability are recorded in the [research reference index](tunix_rl_research_references.md).

| Area | Opportunities | Representative references |
| --- | --- | --- |
| **1. OPD and Distillation** | Extend OPD with feedback, hindsight, roles, and multiple teachers. | [SDPO](https://arxiv.org/abs/2601.20802), [SEED](https://arxiv.org/abs/2607.14777), [SRD](https://arxiv.org/abs/2610.08077), [MAS-OPD](https://arxiv.org/abs/2609.34234) |
| **2. Swarms: Multi-Agent Collaborative RL** | Shared-policy teams, collective rewards and time pressure, orchestration, communication, and credit. | [RLCR](https://www.gr.inc/avalonm1.pdf), [PARL](https://www.kimi.com/en/blog/kimi-k2-5), [AT-GRPO](https://arxiv.org/abs/2510.11062), [C3](https://arxiv.org/abs/2603.06859) |
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

**How to read this roadmap:** each row is a user-visible milestone demonstrated by a recipe, example, or evaluation report. Stages progress from foundations to extensions; audio, video, and OPD branches can proceed in parallel. DTOs, masks, and scorers are implementation tasks rather than standalone features. Existing capabilities are labeled where recipe maintenance or hardware acceptance remains. These are proposed outcomes, not claims of completed support. Miles references use inspected fixed revisions; see the [reference index](tunix_rl_research_references.md) for availability boundaries.

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

| Stage | Feature milestone | Observable outcome / acceptance demonstration | References |
| --- | --- | --- | --- |
| SW1 | **Shared-model collaborative team example** | Run one model in 2–4 roles that divide work, communicate, and use a shared workspace; export team results and complete trajectories. | [RLCR method][rlcr-pdf], [PettingLLMs](https://github.com/pettingllms-ai/PettingLLMs) |
| SW2 | **Shared-policy team RL recipe** | Train the shared policy for all roles from team scores; produce restorable checkpoints and single-agent/team comparison reports. | [Avalon-M1 / RLCR][rlcr-pdf], [AT-GRPO implementation](https://github.com/pettingllms-ai/PettingLLMs) — Method reference versus implementation reference. |
| SW3 | **Time-budgeted team training** | Allow a team deadline and intermediate progress scoring; report quality, actual elapsed time, and cost at fixed budgets. | [How to Train a Swarm][rlcr-blog] |
| SW4 | **Dynamic team size and parallel collaboration** | Recruit/terminate members, execute work concurrently, and join results; compare team sizes and communication overhead. | [RLCR][rlcr-blog], [PARL](https://www.kimi.com/en/blog/kimi-k2-5) — These references use different training approaches. |
| SW5 | **Optional: independent-role policy co-training** | Allow solver/verifier or other roles to use separate models/configurations and collaborate, update, and restore within one task. | [Miles multi-policy][m-multipolicy] |
| SW6 | **Optional: role credit and team distillation** | Compare shared rewards, role/counterfactual credit, or role OPD; evaluate specialization and collaboration benefits. | [C3](https://github.com/EIT-EAST-Lab/C3), [MAS-OPD](https://arxiv.org/abs/2609.34234) |

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

### 2.2 Initial Tunix RLCR Prototype

Milestones: **SW1 demonstrable collaborative team → SW2 trainable shared-policy team → SW3 time-budgeted RLCR-style recipe**. SW4 adds dynamic parallelism; SW5/SW6 are independent-policy, credit-assignment, and distillation branches rather than initial prerequisites.

The first prototype can reuse Tunix GRPO for **RLCR-style** validation; the original primarily uses CISPO, so a blog should state that difference. Broadcasting a collective advantage does not replace token ownership, failure handling, or valid-group checks.

### 2.3 Proposed Blog Angle

Suggested title: **“From Single-Agent RL to Shared-Policy Teams: An RLCR Design for Tunix.”**

1. **Question:** when do more agents and tokens translate into faster task completion?
2. **Training:** shared weights, collective advantages, and ownership of generated-token gradients.
3. **Environment:** communication, shared workspaces, anytime rewards, and actual time budgets.
4. **Tunix milestones:** a collaborative-team example, shared-policy RL recipe, and time-budgeted training/evaluation reports.
5. **Evidence:** budget-matched controls, progress over time, and communication ablations; separate measurements from untested hypotheses.

Keep the conditions when citing results: the reported **3.9×** is an average **derived-latency** speedup at matched output-token budgets and comparable scores, not training throughput or a universal hardware wall-clock speedup. [Report §3.2.2](https://www.gr.inc/avalonm1.pdf) The inspected report is still marked under revision.

## 3. RSI: Recursive Self-Improvement

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
