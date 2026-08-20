# 🚀 DeepSWE 128-Chip Evaluation & Data Washing Campaign Report

**Campaign Tag**: `p46r01a0`  
**Evaluation Profile**: `cluster/profiles/qwen3-4b-dp-parity-deepswe-eval.env` (Qwen3-4B-Instruct, Topology 128)  
**Time Span**: 2026-08-18T08:54:49Z – 2026-08-20T15:13:48Z (Full Campaign Duration: ~54 Hours)  

---

## 1. 📊 Executive Summary (核心清洗成果)

| 维度 | 指标数值 | 说明 |
| :--- | :--- | :--- |
| **总评测轨迹数 (Total Rollouts)** | **22,918** | 在真实 Python 代码库沙箱中运行的完整多轮交互轨迹 |
| **Golden 正样本数 (`reward = 1.0`)** | **2,280** | 成功定位 Bug、修改代码并**100%通过全部测试用例**的有效解 |
| **单次采样通过率 (Pass@1)** | **9.95%** (~10.0%) | 接近 10% 的高难度真实 SWE 解决率 |
| **覆盖独立难题数 (Unique Problems)** | **1,136** | 涵盖 9 大主流开源 Python 库真实历史 Issue / PR |
| **成功攻克独立难题数 (Pass@k Coverage)** | **612 / 1,136 (53.87%)** | **超过 53.8% 的难题产出了至少一条正向有效修复轨迹** |
| **总环境交互步数 (`env.step`)** | **868,386** | 包含搜索、文件编辑、命令执行的多轮 Agent 交互 |
| **产出 DPO 偏好对比对数** | **4,374 对** | 同一 `task_id` 下高质量 Chosen (1.0) 与 Rejected (0.0) 对 |

---

## 2. 📂 Repository-Level Performance Breakdown (各代码库难度与成功率分布)

| 仓库名称 (Benchmark Repo) | 采样轨迹总量 | 成功样本数 (`reward=1.0`) | Pass@1 成功率 | 难度评估与特征分析 |
| :--- | :---: | :---: | :---: | :--- |
| **`scrapy_final`** | 1,850 | **339** | **18.3%** | **最高通过率**。异步网络爬虫架构清晰，测试定位快 |
| **`coveragepy_final`** | 1,120 | **133** | **11.9%** | 工具类单点功能 Bug 多，代码补丁精炼 |
| **`pillow_final`** | 2,944 | **320** | **10.9%** | C 扩展与 Python 图像接口，格式化修改成功率稳定 |
| **`pyramid_final`** | 1,376 | **150** | **10.9%** | Web 路由与上下文配置相关，测试用例执行迅速 |
| **`aiohttp_final`** | 3,385 | **364** | **10.8%** | 协程网络库，处理 HTTP 协议边界与异常捕获有效 |
| **`orange3_final`** | 2,619 | **238** | **9.1%** | 数据挖掘与 GUI 逻辑复杂，涉及多组件交互 |
| **`tornado_final`** | 543 | **44** | **8.1%** | 异步事件循环与 WebSocket 特殊边界 Bug |
| **`numpy_final`** | 3,308 | **264** | **8.0%** | 底层张量计算与 C 接口密集，数值精度与形状校验严格 |
| **`pandas_final`** | 5,773 | **428** | **7.4%** | **最复杂代码库**。索引、分组、类型推导等深层 Bug 较多 |
| **总计 / 全局** | **22,918** | **2,280** | **9.95%** | **整体分布呈现出合理且符合工业认知的难度梯度** |

---

## 3. 🔍 结果合理性评估 (Reasonableness Analysis)

1. **难度阶梯非常合理**：
   - 偏向顶层应用与工具类的 `scrapy` (18.3%) 和 `coveragepy` (11.9%) 成功率明显高于底层重度数据处理库 `numpy` (8.0%) 和 `pandas` (7.4%)。
   - 这与业界在 SWE-bench / R2E-Gym 上的基准评测难度分布完全吻合。
2. **Pass@k (53.87%) 说明模型拥有强大的解空间探索能力**：
   - 虽然单次 Pass@1 为 9.95%，但通过对 1,136 个问题各采样 15~20 条 Rollout，模型成功攻克了其中 **612 个独立问题**，证明模型具备解决超过半数问题的潜力，非常适合作为 RL 探索基础。
3. **工具调用的合理性**：
   - `file_editor` 占比最高（284,177 次），说明 Agent 在密集地查看与修改代码。
   - `search`（89,476 次）紧随其后，符合先检索定位后修改的逻辑。
   - `execute_bash`（42,642 次）主要用于运行 pytest/unittest 验证补丁。

---

## 4. 📦 产出的 RL / SFT 数据集清单

所有数据集已保存至本目录下的 `datasets/` 文件夹：

1. **`datasets/deepswe_golden_sft_2280.jsonl`** (2,280 条):
   * 仅包含 `reward = 1.0` 的有效修复轨迹。
   * 适用于 **SFT 冷启动（Warmup）**，让模型学习标准的 SWE 交互与补丁生成。
2. **`datasets/deepswe_dpo_pairs.jsonl`** (4,374 对):
   * 同一问题下的 `chosen` (成功解) 与 `rejected` (失败解) 配对。
   * 适用于 **DPO (Direct Preference Optimization)** / KTO / ORPO 偏好对齐训练。
3. **`datasets/deepswe_full_rl_22918.jsonl`** (22,918 条):
   * 全量评测多轮轨迹（含成功与失败）。
   * 适用于 **PPO / GRPO / Trajectory-level RL** 进行强化学习探索训练。
