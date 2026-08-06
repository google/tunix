# Design — canon-zero-tim 可分发包

Status: 稳定(2026-08-05)。改动本文需给出推翻对应理由的新事实。

## 1. 目标与最终用途

**最终用途(用户 2026-08-05 明确)**:在**大集群 multi-host** 上跑 **deepswe** 与 **frozenlake**。

这决定了包的定位。它不是"实验代码的归档",而是:

> **一套搬到新集群时,用来判定 zero-TIM 在该拓扑上可不可达、还需要重钉哪些自由度的准入工具。**

推论:**可移植性是功能,不是整洁度**。`sys.path.insert("/mnt/disks/tunix-data/claude_work")`
这类硬编码在本机只是不雅,在新集群上是**静默失效 ⇒ 走 stock ⇒ 假绿灯**,属于阻断级缺陷。

## 2. 分层架构:哪一层"任何人都能跑"

| 层 | 依赖 | 内容 | 分发形态 |
|---|---|---|---|
| **T0** | 纯 CPU,秒级 | `rpa_diff_chunked` fp64 自测(值/梯度/FD oracle)、多序列 VJP2 gate | ✅ **代码 + 单命令** |
| **T1** | ≥4 TPU 芯,**无模型无镜像**,秒级 | minrepro 四件套(f4 / thirdprog / topo / mesh2d)+ **新集群准入探针** | ✅ **代码 + 单命令** |
| **T2** | 镜像 + 模型 checkpoint + 4 芯 v5p | G1–G4、G7–G9 release gates | ⚠️ **配方 + 预期输出行 + SHA 清单** |
| **T3** | 上述 + 多小时 | P26 训练阶梯 | ⚠️ 同上 |

**关键设计判断**:不试图打包全部。**T0+T1 恰好就是科学核心所在** ——
任何人用任意 4 芯 TPU,秒级即可亲眼看到「4-way 脏 / 2-way 净 / F4 令其归零 / 第三程序漂移」。
这比给他一个跑不动的 32B 配方有价值得多。

T2/T3 强行"让人也跑一遍"是错误目标:它们需要特定镜像与 checkpoint,
正确形态是**可核对的配方**——读者跑出结果后,用 SHA 清单核对自己是否得到同一份数字。

## 3. 两个 codebase,两组 patch 锚点

改动横跨两个互不相同的代码库:

```
① tpu_inference  (vLLM TPU 后端,存在于 docker 镜像 tunix_frozenlake_image:vllm-tpu0.25.0)
   → 以 :ro mount 覆盖镜像内文件的方式生效
   → patch 锚点 = tpu_inference 包内路径
   → 涉及:layers/jax/linear.py · layers/jax/embed.py · 注意力接口 · worker/tpu_runner.py

② tunix          (训练侧,git repo: sequence_packing/tunix)
   → patch 锚点 = tunix repo 根
   → 涉及:rl/alignment.py · rl/rollout/vllm_rollout.py · rl/trainer.py · sft/peft_trainer.py
            · rl/agentic/* · rl/algo_core.py · rl/canonical_qwen3_adapter.py
            · rl/inference/inference_worker.py + 4 个测试文件
```

`src/rpa_diff_chunked.py` 不属于任何一方 —— 它是**独立模块**,被 ① 的 patch import。
因此它必须**无硬编码路径**,靠 `PYTHONPATH` 或包内相对路径解析。

## 4. 关键设计决策

### D-1 · 真 diff,不是整文件副本

**为什么**:整文件副本让改动**不可见**。读者拿到 406 行的 `linear_patched.py`,
无法回答"这个文件相对上游改了什么"。

**怎么做**:从本地镜像 `418dc632edd8` 抽出 stock 原件,与现有副本生成 diff,
按**语义**拆成 6 个 patch(每个 patch = 一项根因修复),而不是按文件拆。

**验收(P1 gate,最强形式)**:`stock + patch` 的产物与现有 `*_patched.py`
**逐字节相同**(SHA-256 相等)。这一条同时证明 diff 无损、无遗漏、无夹带。

### D-2 · `git worktree`,不是 `git checkout -b`

`sequence_packing/tunix` 当前有 **14 个 dirty 文件**,**它们就是产出 P26 G3 release 证据
的那份代码**。`checkout -b` 会波及这个 worktree。

```
✅ git worktree add <新路径> -b yuxzhang/canon-zero-tim main
   在另一个目录开新分支,当前 dirty worktree 一个字节不动
```

### D-3 · 修复只在包内,不动冻结 runner

两个活地雷:
- `run_p19x6.sh:23` 硬编码 `-e CANON_RPA_VJP=1` —— 旧的 **prefill-only** rpa_diff,
  其反向对 chunk+cache 路径的 **kv 梯度恒为 0**(R4)
- 同行 `-e P18_SKIP_ENV_CHECK=${SKIPENV:-1}` —— **默认 1**,按坑 #22 会把配置断言
  降级成警告一路跑过去

**不就地修**,因为这些 runner 承载已签字的 release 证据,改了会让旧 artifact 无法回溯到
产出它的确切脚本。**包内提供默认安全的新 wrapper**,并在 `KNOWN_FOOTGUNS.md` 写明旧脚本的陷阱。

### D-4 · fix 开关与 diagnostic 开关必须可区分

```
FIX(进包,README 记载):
  CANON_RPA_D / CANON_RPA_P / CANON_RPA_M      RPA v3 block 钉死
  CANON_FIXED_AR                               o_proj/down_proj 固定序 AR
  CANON_FIXED_AR_EMBED                         vocab-sharded embed gather 固定序
  MIN_TOKEN_BUCKET                             all-M 对齐
  CANON_LOGPROB_M / CANON_PROMPT_PROCESSED_LOGPROBS   processed-logprob 同 caller 边界
  CANON_RPA_VJP2 / CANON_VJP2_MAX_SEQS         可微反向
  CANON_EXPECT_MODEL_MESH_IDS                  mesh order 断言
  XLA: --xla_allow_excess_precision=false

DIAGNOSTIC(不进包,仅附录列名):
  CANON_CUT / CANON_BARRIER / CANON_BARRIER_ALL / CANON_TAIL
  CANON_MM_ALGO / CANON_MM_ALGO_PRESET / CANON_ATTN_DUMP / CANON_FORCE_MIXED
  CANON_RPA_VJP(旧,已知 kv 梯度为零 —— 列入 KNOWN_FOOTGUNS)
```

### D-5 · `CLUSTER_ADMISSION.md` 是本包对最终目标的实质贡献

其余部分都是"把已有的整干净";这一份是**为新集群造的新工具**。它必须回答:

1. **way-count 探测** —— 目前只测过 2-way(净) / 4-way(脏),**8-way 未知**,而 tp8 是目标配置
2. **mesh order 双侧断言** —— 复用已有的 `CANON_EXPECT_MODEL_MESH_IDS`
   (`tpu_runner.py:907-923`,不匹配直接 `raise`)
3. **多 slice 分层归约探测** —— ICI vs DCN 两级归约是本项目**零覆盖**的新程序分裂来源
4. **bucket 合同重推导** —— `num_tokens_paddings_per_dp = [p // dp_size]`(`tpu_runner.py:1101`),
   dp 下每 replica 看到的 M 要重算
5. **F4 成本模型** —— 当前实现通信量 `(n-1)·|out|` vs ring 的 `2(n-1)/n·|out|`,
   比值 ~`n/2`;tp4 为 2×(已接受),**tp8 为 4×**(未实测)

## 5. round-trip 验证(P6)—— 为什么它是最强 gate

以**陌生使用者**身份,只用包里的东西,复现**已签字的数字**。判据是「**逐字节相同**」,
不是「跑通了」。任何一个数字对不上 ⇒ 包漏了东西。

```
P6a 便宜层(分钟级)   G1: K2.abc 0/10240 · 0/303872 · THIRDPROG 0/303872
                     G2: 四边界 0 字节 · w=r=w·r=1 · clip/TIS=0
                     覆盖 R1/R2/R3/R4
P6b 完整层(~1h)      P26 G1a: N_action=248 · 三边界 0 字节
                     · 梯度 0.2502315640449524 · verdict P26_GSM8K_G1A_PASS
                     覆盖全部六项 + 训练侧 hooks + classifier
```

## 6. 风险

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| **副本与 stock 不同源** | 镜像里的上游文件版本与当初做副本时不一致 | P1 逐字节 gate 会直接暴露;红即停,不继续 |
| **误伤 dirty worktree** | 用 `checkout -b` 而非 `worktree add` | D-2 写死;P0 第一步即建 worktree |
| **去掉硬编码路径后 import 断裂** | `rpa_diff_chunked` 找不到 | P2 gate 要求在**非 `/mnt/disks`** 路径下 import 成功且 PATHTRACE 照打 |
| **包"看起来完整"但漏了东西** | 语义拆分时遗漏某处修改 | P1 逐字节 + P6 round-trip 双重拦截 |
| **8-way / 多 slice 结论被外推** | 本机测不了却写成已验证 | P4 强制标 `UNVERIFIED — 待目标集群`;`principles.md` R4/R8 同源纪律 |
| **artifact 路径失效** | EVIDENCE.md 引用的文件被清理 | P5 gate 用**脚本**核对存在性 + SHA,不靠人眼 |
