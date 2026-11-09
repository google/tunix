# Tunix Capability and Research Checklist

Updated: November 9, 2026. This is a checklist, not a detailed design or a commitment to implement every item. Miles is one comparison point; swarm training and recursive self-improvement are independent, optional Tunix directions.

The broader research survey and feature milestones are listed in [RL Development Directions](tunix_rl_research_directions.md).

## Scope and Status

| Repository | Inspected snapshot |
| --- | --- |
| Tunix | `haoyu-next`, `66bc9ecb18e2ac8fa9a3bb19baec9c529508a342`, including the fixed Qwen3 sampled-token OPD recipe |
| Miles | `88cf4fc932fca650aebc409ccbb5bc9dd7f59afe` |
| Miles Diffusion | Separate repository, `221f965fa68e4d865665a48e8478932a231dc4ad` |

These are reproducible source snapshots, not a claim that either snapshot is the latest upstream main. Paper and project links were checked separately for this checklist. Source presence, an experimental example, a complete recipe, and a measured accelerator result are different levels of evidence. No new accelerator experiments were run for this document.

**Status vocabulary:** **Existing** = implementation found; **Partial** = relevant components exist but the named workflow is incomplete; **Recipe** = an end-to-end example/configuration exists, without implying reproduction here; **Not found** = no corresponding standard implementation or recipe was found in the inspected snapshot; **Optional** = a proposed extension.

The Tunix work column lists possible additions or validation work. An entry there does not mean that the underlying capability is entirely absent.

## 1. Platform, Models, and Training Stack

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| P01 | Core tensor/compiler stack | PyTorch; Megatron/FSDP2/torchtitan | Existing: JAX, Flax NNX, XLA | Preserve native JAX implementation; compare behavior at interfaces |
| P02 | Training backend choice | Megatron-LM, FSDP2, torchtitan | Existing: native Tunix and MaxText integration | Publish feature parity per backend |
| P03 | Accelerator coverage | NVIDIA/AMD paths; `torch_tpu` platform and image plumbing | TPU-focused; CPU tests and GPU examples, including QLoRA | Validate each model/backend/device combination separately |
| P04 | Large-model parallelism | Backend-dependent TP/PP/CP/EP/DP | JAX meshes/sharding; native/MaxText coverage differs | Document supported parallel layouts and combinations |
| P05 | Model catalog | Broad SGLang/backend adapters and launch profiles | Native Gemma/Gemma3/Gemma4, Llama3, Qwen2/Qwen3; additional MaxText models | Add models together with mapping, synchronization, and recipe tests |
| P06 | Full-parameter SFT | Existing | Existing | Maintain reproducible model/data recipes |
| P07 | LoRA training | Existing across selected backends | Existing native LoRA | Expand model-specific adapter paths and rollout parity |
| P08 | QLoRA and low-bit training | FP8/MXFP8/NVFP4 and INT4 QAT paths | Existing Qwix/QLoRA and mixed precision | Add an explicit low-bit RL support matrix and numerical checks |
| P09 | Rematerialization and memory tuning | Backend-specific activation recompute/offload | Existing rematerialization, accumulation, sharding | Publish memory/throughput profiles for representative recipes |
| P10 | Hugging Face model interoperability | Conversion/export and backend mappings | HF loading/mappings; selected LoRA-merged safetensors export helpers | Validate model-specific exports; generic Orbax-to-HF export is not established |

Evidence: [Miles backends][m-backends], [Miles platforms][m-platforms], [Tunix model registry](../tunix/models/registry.py), [Tunix training](../tunix/dist_rl/train), [Tunix model loading](../tunix/models/automodel.py), [Tunix examples](../examples).

## 2. Objectives, Advantages, and Distillation

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| A01 | GRPO | Existing | Existing | Compare grouping, masking, normalization, and reduction |
| A02 | PPO with a critic/GAE | Existing; Megatron recipe with colocated critic | Existing PPO/GAE | Maintain a distributed actor/critic acceptance recipe |
| A03 | GSPO | Sequence-level objective | Existing GSPO-Token | Treat sequence and token variants as separate objectives |
| A04 | DAPO and Dr.GRPO | Configurable losses/reducers; Dr.GRPO experimental example | Existing DAPO and Dr.GRPO | Compare precise loss and filtering semantics |
| A05 | Leave-one-out group baselines | GRPO variants | Existing RLOO/GRPO-LOO | Document estimator differences rather than count names |
| A06 | REINFORCE++ | Dedicated estimator and baseline variants | Not found as an equivalent named implementation | Add estimator/reward-normalization oracles if selected |
| A07 | DPO and ORPO | RL/SFT stack; no equivalent dedicated native preference trainer found | Existing DPO/ORPO trainers | Retain preference training as a Tunix capability |
| A08 | Offline logit/feature/attention distillation | No equivalent general strategy stack found | Existing distillation trainer and strategies | Maintain the offline strategy APIs and examples |
| A09 | Sampled-token online OPD | Existing teacher-on-student-trajectory path | Recipe: fixed Qwen3-8B student / Qwen3-32B teacher | Complete real teacher/TPU and learning-quality acceptance |
| A10 | Online top-k distribution OPD | Candidate-set selection/weighting paths | Not found online; offline teacher-top-k KL already exists | Add candidate IDs/probabilities and an independently checked objective |
| A11 | Teacher deployment | External SGLang and local Megatron teacher | Existing external SGLang teacher | Optional local teacher with immutable identity and resource isolation |
| A12 | Multi-teacher routing | Per-sample specialist selection and fallback | Not found in current OPD recipe | Add teacher routing, per-sample identity, and compatible tokenizers |
| A13 | Two-phase self-distillation | RLVR teacher followed by frozen-teacher OPD example | OPD building blocks; no corresponding two-phase recipe | Add teacher promotion/freeze and phase-specific records |
| A14 | Single-response pure OPD | Separate from task-group requirements | Partial: current OPD requires at least two valid responses/group | Separate pure-distillation validity from GRPO group baselines |
| A15 | Mixed task reward and OPD | Existing composition | Existing independent task/OPD coefficients and reference beta | Add controlled pure/mixed/task-only quality comparisons |
| A16 | Importance/rejection sampling | TIS/MIS and correction helpers | Existing sampler IS/RS controls | Compare exact weighting, thresholds, and diagnostics |
| A17 | Sampling-support replay | Explicit filtered-support replay | Not found as a standard data contract | Record actual support; distinguish behavior from full-policy scores |
| A18 | Score centering | Candidate-probability-based objective/example | Not found as an equivalent objective | Add candidate data, loss, and independent gradient checks |
| A19 | Joint multi-token prediction auxiliary training | Megatron-specific MTP training configuration | No matching RL MTP-training workflow established | Add auxiliary heads/loss and train/rollout compatibility if selected |
| A20 | Clipping, KL, entropy, and normalization | Configurable clipping, KL estimators/placement, entropy, whitening | Existing policy-loss/KL/entropy/normalization primitives | Audit exact modes and defaults; names do not establish equivalence |
| A21 | Custom correction hooks / OPSM | Custom TIS/correction and OPSM implementation | IS/RS overlap; exact OPSM parity not established | Add only the selected correction after a mathematical comparison |

Evidence: [Miles loss implementation][m-loss], [Miles OPD][m-opd], [Miles score centering][m-centering], [Miles support replay][m-support], [Tunix algorithms](../tunix/rl/algorithm_config.py), [Tunix loss](../tunix/rl/algo_core.py), [Tunix offline distillation](../tunix/distillation), [Tunix OPD guide](../tunix/dist_rl/examples/on_policy_distillation/QWEN3_8B_OPD.md).

## 3. Multimodal and Generative Models

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| M01 | Image input models/processors | VLM adapters and multimodal training inputs | Existing Gemma3/Gemma4 vision and image processing | Expose a consistent training-facing media contract |
| M02 | Image SFT | VLM example | Partial: Gemma3 reference script needs maintenance | Repair API/path assumptions and validate on an accelerator |
| M03 | Image DPO | No matching dedicated recipe found | Recipe: Gemma3 image-preference notebook | Maintain and validate the existing recipe |
| M04 | Single-turn image RL | Recipe: Qwen3-VL/GEO3K | Partial: media-aware generation/model components; no standard VLM GRPO recipe | Connect media DTO, scoring, attention, assembler, and loss |
| M05 | Multi-turn image/tool RL | GEO3K image-cropping example | Agentic and media components; no complete corresponding recipe | Align images/screenshots with turns and tool masks |
| M06 | Audio understanding RL | Inkling components and legacy audio path; default-path limitations | Gemma4 audio/processor/sampling; no standard audio RL recipe | Add end-to-end audio scoring, batching, and task evaluation |
| M07 | Video understanding RL | Processor/grid/MRoPE components; no general default-loop guarantee | No standard video RL recipe found | Add frame/time metadata, token budgets, and a fixed-frame recipe |
| M08 | Encoder freeze/update | Model-specific tower support; some update/sync limitations | Encoder parameters exist; no standard VLM RL freeze/update recipe | Define trainable filters and synchronize the same graph |
| M09 | Media packing and attention | Model/backend-specific media expansion | Text packing exists; generic RL media plumbing incomplete | Preserve media positions, segment attention, and microbatch mapping |
| M10 | Multimodal OPD | Components must be checked by model/backend/path | Current standard OPD recipe is text-only | Combine only after media scoring and text OPD pass separately |
| M11 | Discrete-token diffusion SFT | Distinct from the separate continuous-diffusion project | Existing contracts and weighted target-aligned CE integration | Add a concrete model, denoising sampler, and runnable recipe |
| M12 | Continuous image/video diffusion RL | Separate Miles Diffusion: Flow-GRPO/DiffusionNFT/SFT | No corresponding continuous-latent RL stack found | Separate model/sampler/reward/likelihood and distributed training project |

Evidence: [Miles image recipe][m-vlm], [Miles media data][m-media], [Miles Diffusion][m-diffusion], [Tunix Gemma3](../tunix/models/gemma3), [Tunix Gemma4](../tunix/models/gemma4), [Tunix image SFT](../examples/sft/vlm_training.py), [Tunix image DPO](../examples/vl_dpo_gemma3.ipynb), [Tunix discrete diffusion](../tunix/sft/diffusion.py).

An image-capable model is not evidence of a complete image RL recipe. Audio/video components do not establish the same support level as Miles' image recipe. Tunix's discrete-token diffusion SFT is implemented, but is not a continuous image/video diffusion RL system.

## 4. Rollout, Scheduling, and Numerical Consistency

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| R01 | Inference engines | SGLang-centered | Existing native, vLLM-JAX, and SGLang-JAX paths | Publish per-engine feature compatibility |
| R02 | Decoupled rollout/training | Existing | Existing distributed workers and orchestrator | Benchmark both colocated and disaggregated layouts |
| R03 | Asynchronous scheduling | Continuous generation/buffers and async examples | Existing queues, staleness, dispatch windows, backpressure | Extend coverage and measure useful-sample throughput |
| R04 | Policy versions/staleness | Existing version-aware scheduling | Existing version tracking and staleness limits | Validate version spans for multi-turn and resumed work |
| R05 | Zero-lag OPD | Sampling/scoring contract depends on selected mode | Explicit current OPD restriction: zero lag, no in-flight updates | Keep synchronous acceptance; finite-lag OPD is a separate extension |
| R06 | Token-in-token-out | Session/proxy paths for external harnesses | Existing internal exact-token continuity | Add an external-harness adapter without retokenization |
| R07 | Multi-turn/tool trajectories | Existing | Existing agent/environment/collector paths | Expand task coverage and cancellation/timeout tests |
| R08 | Long-output continuation/partial work | Rollout scheduling/session mechanisms | Existing bounded partial-rollout/in-flight barriers and turn versions | Expand interruption, partial-response, and resumed-token coverage |
| R09 | MoE routing replay | R3 across selected models/backends | Existing collection, compact encoding, packing, MaxText replay | Expand model/engine coverage and measure transport overhead |
| R10 | Numerically matched train/infer | True-on-policy profiles and contract checks | Existing Zero-TIM kernels, canonical Qwen3 work, agreement metrics | Turn a supported kernel/model combination into a standard recipe |
| R11 | Text packing | Backend-dependent packing/context parallelism | Existing padded/packed assemblers | Add actual-model packed/unpacked equivalence and accelerator coverage |
| R12 | Prefill/decode disaggregation | Dedicated integration/docs | No equivalent standard recipe found | Add engine integration only for a measured deployment need |
| R13 | Speculative decoding | SGLang integration/docs | No equivalent standard RL recipe found | Verify sampled-token probabilities and routing traces before adoption |
| R14 | KV cache/rollout transfer | Mooncake-oriented integration | Engine-managed caches; no matching standard transfer layer found | Optional transport integration with token/version checks |

Evidence: [Miles async][m-async], [Miles agentic rollout][m-agentic], [Miles routing replay][m-r3], [Miles true-on-policy][m-top], [Miles advanced integrations][m-advanced], [Tunix orchestrator](../tunix/dist_rl/orchestrator), [Tunix rollout](../tunix/dist_rl/rollout), [Tunix agentic stack](../tunix/rl/agentic), [Tunix Zero-TIM](../tunix/zero_tim_kernel/README.md).

## 5. Distribution, Recovery, and Training Services

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| D01 | Weight synchronization | Multiple transports plus engine handshakes | Existing Raiden, resharding/mapping, version coordination | Compare synchronization time and supported parameter layouts |
| D02 | Atomic policy publication | Pause/update/announce/resume mechanisms | Existing synchronization coordinator | Validate receiver failures and half-published versions |
| D03 | Worker health/retry/eviction | Existing | Existing; fault tolerance is opt-in | Add failure-injection coverage |
| D04 | Automatic worker rebuild/rejoin | Recovery/controller reconciliation paths | Partial: dynamic rejoin remains unimplemented | Rebuild, initialize, synchronize, then safely reactivate workers |
| D05 | Checkpoint/restore | Backend-specific training state | Existing Orbax model/optimizer and program metadata | Test data cursor, objective identity, and committed boundaries |
| D06 | Program/queue recovery | Runtime reconciliation and lifecycle controls | Existing committed-boundary cursor/version recovery; no durable pending-queue replay | Define durable replay/deduplication for pending and partially consumed groups |
| D07 | Live orchestration hot restart | Dedicated example | No equivalent standard live hot-restart recipe found | Persist/rebind orchestration state independently of model workers |
| D08 | Cluster scheduling | Ray or Kubernetes/Helm | Existing RPC/discovery, local/Kubernetes executors, Pathways; Ray bootstrap for multi-host vLLM | Compare packaging/lifecycle; Tunix already has Kubernetes deployment |
| D09 | Split deployment | Separate trainer/engine/orchestrator Helm releases | Separate roles exist; no matching Helm product found | Add independently managed services if required |
| D10 | Multi-policy training | Solver/verifier recipe with distinct trainers/engines | Multiple roles/components; no general co-training recipe found | Policy registry, optimizer/checkpoint isolation, reward routing |
| D11 | Multi-LoRA training service | Tinker gateway, tenants/adapter slots | Existing single-task LoRA/QLoRA | Adapter lifecycle, per-tenant queues/optimizers, versioned sampling |
| D12 | Resource elasticity | Engine fleet/lifecycle management | Worker registry and fixed recipe layouts | Support safe capacity changes and resource accounting |

Evidence: [Miles recovery][m-ft], [Miles cluster backend][m-cluster], [Miles infrastructure examples][m-infra], [Miles multi-policy][m-multipolicy], [Miles multi-LoRA][m-multilora], [Tunix weight synchronization](../tunix/dist_rl/weight_sync), [Tunix worker registry](../tunix/dist_rl/orchestrator/worker_registry.py), [Tunix runtime contracts](../tunix/dist_rl/common/datatypes.py), [Tunix trainer checkpointing](../tunix/dist_rl/train/peft_trainer_v2.py).

## 6. Data, Rewards, Evaluation, and Developer Experience

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| E01 | Custom reward/environment APIs | Hooks, reward implementations, external environments | Existing reward managers/functions and agent environments | Standardize task adapters and reward provenance |
| E02 | Dataset filtering/sampling | Dynamic filters, sample hooks, nonzero-variance oversampling | Existing masks/group filters/dispatch; legacy DAPO dynamic sampling remains TODO | Complete dynamic resampling and reusable curriculum/data-state interfaces |
| E03 | Multi-task/OOD evaluation | Configurable tasks and experimental evaluation examples | Existing task-specific GSM8K/DeepSWE evaluation; no unified multi-task/OOD registry found | Generalize task registry, metrics, and held-out datasets |
| E04 | Evaluation schedule | Sync/async and snapshot-oriented options | OPD baseline/final/checkpoint evaluation; other task examples | Add periodic and snapshot-isolated async generation evaluation |
| E05 | Reproducible experiment records | Launch profiles and deterministic examples | OPD fixed revisions, manifest digests, seeds, per-example reports | Extend the same record format across recipes |
| E06 | Metrics/profiling | Dashboard, trainer/rollout metrics, mismatch diagnostics | Existing TensorBoard/W&B/OTel integrations, Perfetto, system metrics | Unified cross-process dashboards and performance gates |
| E07 | Trajectory persistence/inspection | Rollout/debug/dashboard facilities | Stores, rollout writer, explorer; active orchestrator-side persistence remains partial | Complete consumption/recovery wiring; add team/role/media/credit views |
| E08 | CI and performance evidence | Tests, model/hardware recipes, experimental labels | Existing tests/nightly infrastructure; OPD CPU oracles | Publish a recipe/backend/hardware support matrix |
| E09 | Numerical regression tests | Selected train/infer correctness profiles | Existing independent OPD gradient/update oracles and replay tests | Extend to each new objective, media contract, and execution mode |
| E10 | Extension boundaries | Backend/custom rollout/reward hooks | Algorithm registry, engines, trainers, environments, distillation strategies | Keep stable interfaces across legacy and distributed stacks |
| E11 | Unified dataset ingestion | Common JSONL/Parquet/glob/slice, chat/tool/media metadata path | Existing Grain/TFDS/Hugging Face and custom/task-specific loaders | Unify data schema, validation, checkpoint state, and media mapping |
| E12 | Reward catalog and service hooks | Named task graders plus async/batched/group/custom/HTTP rewards | Existing callable/environment/task rewards | Extend named graders and common remote/group interfaces as needed |
| E13 | Token/FLOPs batch balancing | Dynamic token batches and work-balancing controls | Packing, microbatches, accumulation and sharding foundations | Add explicit workload balancing and useful-token efficiency measurements |

Evidence: [Miles examples index][m-examples], [Miles rollout code][m-rollout], [Miles dashboard][m-dashboard], [Tunix metrics](metrics.md), [Tunix performance](performance.md), [Tunix trajectory explorer](../tunix/dist_rl/trajectory/explorer/README.md), [Tunix OPD validation](opd_validation.md).

## 7. Recipes and Agent Ecosystem

| ID | Capability | Miles | Tunix | Tunix work to consider |
| --- | --- | --- | --- | --- |
| X01 | Text math RL | PPO/OPD and model launch recipes | Existing GSM8K, math, and smoke examples | Maintain model-specific fixed train/eval configurations |
| X02 | Coding/terminal agents | Harbor/Docker and Daytona examples | Existing DeepSWE distributed example | Add benchmark adapters and reproducible sandbox images |
| X03 | Tool/code execution | Retool and experimental Search-R1 | Existing tools, calculator, agentic environments | Extend tool ecosystem and test long interactions |
| X04 | Small interactive environments | OpenEnv/other integrations | Existing FrozenLake | Keep small environments for end-to-end correctness checks |
| X05 | Computer-use/image tools | Experimental HUD and image-cropping examples | Agentic and image foundations; no matching standard recipe found | Screenshot/action task, media propagation, action verifier |
| X06 | External environment ecosystems | Harbor, NeMo Gym, OpenEnv, Verifiers | Native environments and selected task examples | Add one connector per actual task requirement |
| X07 | Sandbox providers | AgentENV/Daytona/E2B/Modal integrations or examples | Existing DeepSWE SandboxFleet integration and task-specific environments | Common sandbox lifecycle, file/artifact, timeout, cleanup API |
| X08 | Additional research workflows | Experimental formal math, Tau-bench, Strands, multi-agent | Custom extensible environments | Add Lean/formal-verifier, web/search, customer-service, or framework adapters selectively |
| X09 | Multi-agent benchmark recipe | Experimental multi-agent plus distinct multi-policy example | Agentic foundations; no general swarm-training recipe found | Choose shared-policy, role-policy, or orchestrator-only training explicitly |

Evidence: [Miles examples and experimental labels][m-examples], [Miles environments][m-env], [Tunix distributed examples](../tunix/dist_rl/examples), [Tunix agentic documentation](agentic_rl.md).

## 8. Optional Swarm and Multi-Agent Training Directions

All ten unique references supplied for this task are retained below; the duplicated RLCR entry is included once. These are optional research tracks, not claims of existing Tunix recipes. The component column is a proposed Tunix work list, not a description of what the cited paper implements in Tunix.

| ID | Reference | Verified mechanism / update pattern | Tunix components to add |
| --- | --- | --- | --- |
| S01 | [Avalon-M1 / RLCR: How to Train a Swarm](https://www.gr.inc/blog/how-to-train-a-swarm), [report](https://www.gr.inc/avalonm1.pdf) | Shared policy across the organization; broadcast collective group-relative advantage to self-generated active tokens; asynchronous messages/workspace and wall-clock deadlines. Most environments use CISPO, so this is not simply a new GRPO estimator. | Joint team episodes; team-group advantage; agent/token ownership masks; shared workspace/messages; anytime termination and deadline-aware cancellation |
| S02 | [MPT: Learning Self-Orchestration in Decentralized Agent Swarms](https://arxiv.org/abs/2610.22322), [code](https://github.com/Embodied-Minds-Lab/MPT) | Independent agent policies; each policy generates both actions and bids using the same parameters with alternating updates. Auctions, bucket-brigade credit, and counterfactual bid preferences select one winning agent per action round. | Policy routing; action/bid samples and objectives; credit-transfer ledger; environment snapshot/restore and counterfactual branches |
| S03 | [R²Orch](https://arxiv.org/abs/2610.22740), [code](https://github.com/beita6969/R2Orch) | Proposer/Director self-play with frozen execution workers; graph editing and recursive-learnability task selection inside a skill-internalization loop. Skill-visible/hidden comparisons drive skill removal and internalization into orchestration behavior. | Proposer and Director policies; graph-edit environment; versioned skills; paired skill ablations; outer-loop task/skill controller |
| S04 | [Kimi K2.5 / PARL](https://www.kimi.com/en/blog/kimi-k2-5) | Train the central orchestrator while subagents stay frozen; learn decomposition and useful parallel execution with scheduled rewards and critical-path constraints. | Spawn/wait/join/cancel primitives; frozen worker services; orchestrator-only loss mask; critical-path metrics and reward schedules |
| S05 | [MAGRPO: LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652) | Cooperative multi-agent, multi-turn group-relative policy optimization. | Team-level rollout grouping; per-agent/turn logprobs and masks; joint-policy objective; team reward evaluation |
| S06 | [Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems](https://arxiv.org/abs/2602.08847) | Normalize advantages using each agent's reward statistics to stabilize heterogeneous role gradients. | Agent-wise statistics; configurable normalization groups; role-wise gradient/reward diagnostics; per-agent optimization settings |
| S07 | [MAPoRL](https://arxiv.org/abs/2502.18439) | Independently draft, discuss over multiple turns, then co-train using correctness plus corrective/persuasive discussion rewards. | Discussion environment; immutable speaker ownership; answer/discussion verifier; multi-policy updates and collaboration metrics |
| S08 | [MALT](https://arxiv.org/abs/2412.01928) | Generate/verify/refine search trees; value propagation produces role-specific offline SFT/preference data. This is not an online multi-agent GRPO recipe. | Tree collector; return/value propagation; role-specific dataset export; existing SFT/DPO trainer integration |
| S09 | [MAS-OPD: On-Policy Distillation for Multi-Agent Systems](https://arxiv.org/abs/2609.34234) | Independent role policies plus frozen-teacher token scores; target-role and counterfactual-role scoring, with completed-history error attribution. | Extend current OPD with role-conditioned teacher calls, contrastive scores, history-only attribution, per-role policy/version routing |
| S10 | [MARTI](https://github.com/TsinghuaC3I/MARTI) | Multi-agent RL framework with centralized interaction/reward allocation and distributed policy training; graph workflows and tree-search extensions. | Workflow adapter; centralized credit/reward API; independent policy trainer management; optional tree-search trajectory adapter |

**Result/claim corrections:** Avalon reports a mean **3.9× derived-latency** reduction across 12 benchmarks at matched output-token budgets and comparable scores; the metric replays execution logs at fixed inference rates and is not a universal measured wall-clock or training-throughput gain. The report is marked under revision. PARL's official blog separately reports **up to 4.5×**. MPT's decentralized bidding should not be described as all agents acting concurrently. MALT uses offline post-training. R²Orch internalizes orchestration skills into Director weights; its Proposer learns task selection and execution Workers stay frozen. These distinctions follow the linked primary sources.

For MAS-OPD, teacher-side attribution must use completed past turns; a verdict produced from the currently scored response must not enter that response's teacher context. A frozen teacher can still receive different role/history conditioning. No verified official MAS-OPD implementation link was found in the inspected paper.

## 9. Optional Recursive Self-Improvement Directions

Here, RSI is an umbrella for iterative improvement loops. Weight updates, skill internalization, prompt/workflow evolution, memory changes, and agent-code evolution are separate options. The following Tunix integrations are proposals; the source links identify related demonstrated mechanisms rather than existing Tunix support.

| ID | Optional direction | Minimal Tunix additions | Related reference |
| --- | --- | --- | --- |
| I01 | Iterative verified self-training | Generate/filter/export/train controller; executable or answer verifier; versioned synthetic datasets | [STaR](https://arxiv.org/abs/2203.14465) |
| I02 | Self-play preference improvement | Previous-policy sampling; preserved demonstration seed data; iterative preference objective and checkpoint lineage | [SPIN](https://arxiv.org/abs/2401.01335) |
| I03 | Self-rewarding / co-evolving judges | Judge scoring pipeline; preference construction; iterative DPO; independent judge calibration | [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) |
| I04 | Task generation and automatic curriculum | Proposer/solver loop; solvability/novelty/learning-progress rewards; validated task buffer | [Absolute Zero](https://arxiv.org/abs/2505.03335) |
| I05 | Learnability-based task selection | Per-task outcome distributions; nondegenerate group selection; proposer updates; held-out progress measurement | [R²Orch](https://arxiv.org/abs/2610.22740) |
| I06 | Skill internalization into Director weights | Director RL updates under randomized skill-visible/hidden contexts; skill ledger/removal criteria; zero-skill deployment evaluation; execution Workers stay frozen | [R²Orch](https://arxiv.org/abs/2610.22740) |
| I07 | Self-generated adaptation updates | Isolated inner-loop SFT; self-edit/update directives; downstream-reward outer loop; restore and budget accounting | [SEAL: Self-Adapting Language Models](https://arxiv.org/abs/2506.10943) |
| I08 | Phase-based self-distillation | Promote/freeze an improved checkpoint as teacher; OPD phase controller; transfer and forgetting evaluations | [Miles self-distillation example][m-selfdistill]; existing Tunix OPD |
| I09 | Self-play leagues and role curricula | Opponent/partner checkpoint pools; role assignment; evaluation against fixed and historical partners | Proposed extension using S02/S05/S07 building blocks |
| I10 | Search-to-training feedback | Search/tree rollouts; verified value propagation; SFT/preference/OPD export; held-out search-budget controls | [MALT](https://arxiv.org/abs/2412.01928); proposed OPD integration |
| I11 | Workflow, prompt, and memory evolution | Versioned nonparametric artifacts; candidate execution; paired evaluation; selection/promotion; optional later distillation | Proposed extension; keep distinct from model-weight training |
| I12 | Agent-code evolution | Sandboxed candidate agent implementations; archive/branch selection; benchmark runner; code/artifact lineage | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) |
| I13 | Continual adaptation without excessive forgetting | Replay mixtures; retained task suites; drift/forgetting metrics; stage-specific checkpoints and promotion criteria | Proposed extension to existing training/checkpoint infrastructure |
| I14 | Automated training experiments | Bounded experiment controller; configurable data/objective/sampling choices; cost accounting; independent promotion evaluation | Proposed extension; no autonomous training-research loop is claimed today |

## 10. Shared Modules for the Optional Tracks

These modules can support multiple swarm/RSI tracks. Choosing one track does not require implementing every module.

- [ ] **Team trajectory schema:** `episode_id`, `team_id`, `agent_id`, role, parent/action/message relationships, timestamps, policy ID/version, and token ownership.
- [ ] **Multi-policy registry:** independent trainable/frozen policies, optimizers, engines, synchronization, checkpoint identity, and routing.
- [ ] **Team scheduler:** spawn/wait/join/cancel, DAG dependencies, message queues, workspace, concurrency limits, and resource budgets.
- [ ] **Time-aware execution:** monotonic deadlines, anytime rewards, critical-path latency, timeout attribution, and cleanup.
- [ ] **Credit/advantage plugins:** team broadcast, agent-wise normalization, counterfactual/difference credit, bucket brigade, and tree-value propagation.
- [ ] **Team-aware masking:** update only tokens generated by the selected trainable policy; keep received messages/tool observations as context.
- [ ] **Counterfactual environments:** deterministic reset, snapshot/restore, branch identity, and reproducible paired interventions.
- [ ] **Role-aware distillation:** target/alternative-role teacher scores, history visibility, and teacher/policy identity in data and checkpoints.
- [ ] **Skills and curriculum state:** versioned task pools, skill ledgers, proposer state, difficulty statistics, and phase transitions.
- [ ] **Nested training jobs:** isolated inner-loop updates, resource accounting, model-state restoration, and downstream evaluation.
- [ ] **Artifact lineage:** generated-data provenance, deduplication, policy/teacher/judge versions, and replayable experiment manifests.
- [ ] **Team/RSI evaluation:** fixed held-out tasks; task success, contribution, coordination cost, latency, token/tool budgets, transfer, and forgetting.

## 11. Checklist Boundaries

- Current OPD is **implemented with a complete fixed text recipe**; its remaining accelerator/learning acceptance is not the same as missing implementation.
- Existing async scheduling, text packing, TITO foundations, MoE replay, Zero-TIM kernels, image DPO, and discrete diffusion SFT are **not new from-zero work**.
- Multimodal RL, multi-policy RL, swarm training, and RSI are **distinct extensions**; a general metadata field or environment API does not complete them.
- Shared-policy team training, independent-role co-training, frozen-worker orchestration, and offline role training require **different objectives and state ownership**.
- External project benchmark results remain **source-reported results**, not Tunix performance promises.

[m-backends]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends
[m-platforms]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/platforms
[m-loss]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/loss
[m-opd]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/on_policy_distillation
[m-centering]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/score_centering
[m-support]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/sampling-support-replay.md
[m-vlm]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/geo3k_vlm
[m-media]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/backends/training_utils/data
[m-diffusion]: https://github.com/radixark/miles_diffusion/tree/221f965fa68e4d865665a48e8478932a231dc4ad
[m-async]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/fully_async
[m-agentic]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/user-guide/agentic-rollout.md
[m-r3]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/miles-router.md
[m-top]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features/true_on_policy
[m-advanced]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced
[m-ft]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/fault-tolerance.md
[m-cluster]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/advanced/cluster-backend.md
[m-infra]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/infra_features
[m-multipolicy]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/multi_policy
[m-multilora]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/multi_lora
[m-examples]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/README.md
[m-rollout]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/rollout
[m-dashboard]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/miles/dashboard
[m-env]: https://github.com/radixark/miles/blob/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/docs/user-guide/environments.md
[m-selfdistill]: https://github.com/radixark/miles/tree/88cf4fc932fca650aebc409ccbb5bc9dd7f59afe/examples/on_policy_distillation/qwen3_5_35b_selfdistill
