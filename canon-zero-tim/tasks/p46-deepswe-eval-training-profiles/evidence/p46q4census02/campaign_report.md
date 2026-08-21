# 🚀 DeepSWE 128-Chip Evaluation & Data Washing Campaign Report

**Campaign Tag**: `p46q4census02`  
**Harness Commit**: `87793e07c330f75365f7a6082b20acc733618d12`  
**Sampling Fingerprint / Source Commit**: `stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`  
**Model**: `Qwen/Qwen3-4B-Instruct-2507` (128-Chip Topology, DP16xTP8, Temperature 1.0)  
**Dataset & Whitelist**: R2E-Gym 4,578 tasks, Whitelist 1,851 tasks (`sha256=2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7`)  
**Terminal Exit Marker**: `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 summary_sha256=efa83e0f85600ceb7bc72bc973321271f93ae54cc47b2f5ac848df17d3e6d259`

---

## 1. 📊 Executive Summary (清洗与评测核心成果)

| 维度 | 指标数值 | 说明 |
| :--- | :--- | :--- |
| **评测任务总量 (Total Evaluated Tasks)** | **1,851** | R2E-Gym 经过数据门禁过滤后的高质量白名单任务 |
| **单任务采样次数 ($N$)** | **16** | 严格 $N=16$ 采样，覆盖全解空间探索 |
| **有效多轮评测轨迹数 (Valid Trajectories)** | **29,616** | $1,851 \times 16 = 29,616$ 条在真实代码沙箱中交互的轨迹 |
| **成功解决样本数 (`reward = 1.0`)** | **3,038** | 成功修复 Bug、通过全部单元测试 Golden 样本 |
| **单次采样解决率 (Pass@1)** | **10.26%** | `3038 / 29616`，Qwen3-4B 在 R2E-Gym 真实 Python 库的表现 |
| **可学习任务数 (Q4 Learnable, Pass@16)** | **1,012 / 1,851 (54.67%)** | **超过 54.6% 的任务产出了至少 1 条成功轨迹，且未完全打顶 (1..15/16)** |
| **全失败难题数 (All Fail)** | **839 / 1,851 (45.33%)** | 16 次采样均未通过的困难任务，适合作为更大模型 (Q32) 的探索挑战 |
| **全通过平凡任务数 (All Pass, 16/16)** | **0 / 1,851 (0.00%)** | 证明白名单内无漏网的 trivial/cheatable 任务 |
| **逻辑分片数 (Logical Shards)** | **58** | 58 个分片全部通过一致性校验与报告生成 |

---

## 2. 📦 产出的 Canonical 清洗数据集与交付清单

所有数据集文件已生成并归档：

| 交付文件 | 文件大小 | SHA-256 校验和 | 数据集用途与训练建议 |
| :--- | :---: | :--- | :--- |
| **`p46-campaign.q4_learnable.jsonl`** | 675 KB | `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` | **4B 强化学习黄金探索池**：包含全部 1,012 个具备梯度学习空间的任务 |
| **`p46-campaign.q32_candidates.jsonl`** | 1.3 MB | `38299afc5ca49990a8d5d568b52f06c887a6ed5423fd3300f9ab683cbcafe0bb` | **32B 进阶模型探索池**：包含可学习任务及全失败任务，供大模型攻坚 |
| **`p46-campaign.all_fail.jsonl`** | 560 KB | `325a7627f99bc8a627004220a338de217a64f8911baf85de7b6b151933e1a04f` | **困难负样本库**：839 个全败任务，供分析模型瓶颈与能力边界 |
| **`p46-campaign.all_pass.jsonl`** | 0 B | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 空文件，4B 模型无 16/16 全对任务 |
| **`p46-campaign.summary.json`** | 32 KB | `efa83e0f85600ceb7bc72bc973321271f93ae54cc47b2f5ac848df17d3e6d259` | 全局 58 个 Shard 报告合并摘要与校验元数据 |
| **`trajectory.SHA256SUMS`** | 13 KB | - | 58 个分片原始轨迹文件的完整 SHA-256 清单 |
| **`trajectory.WC`** | 9.3 KB | - | 原始轨迹文件行数清单 (精准对应 29,616 行) |

---

## 3. 💾 持久化原始数据存储路径

按照 `P46_DEEPSWE_PROFILES_RUNBOOK.md` 规范，29,616 条全量交互多轮轨迹保存在集群持久化存储卷中：
- **持久化根目录**: `/mnt/disks/linchai_data/deepswe_eval/p46q4census02/`
- **原始轨迹目录**: `/mnt/disks/linchai_data/deepswe_eval/p46q4census02/outputs/trajectories/`
- **分片报告目录**: `/mnt/disks/linchai_data/deepswe_eval/p46q4census02/outputs/reports/`
- **运行日志归档**: `/mnt/disks/linchai_data/deepswe_eval/p46q4census02/logs/`
