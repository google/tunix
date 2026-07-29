# Plan — fix_accum_fp32
Status: IN-PROGRESS (Scaffolded & Approved). Approved: yes.

## Workstream: 验证 `yuxzhang/fix_accum_fp32` 分支的核心机制（fp32 累加器精度与 bf16 动量降转 cast-back 的显存/算子/时长表现）

### P1.1 实验脚本移植与改造 — gate: 能在本地/TPU成功运行四组实验（支持命令行动态指定 accum depth 与 accum dtype）并输出 [[MEM]] 显存标记
1) 准备能够动态测跑并收集测试指标的专项实验 Python / Bash 脚本（包含对 `gradient_accumulator_dtype` 与 `gradient_accumulation_steps` 的支持）。
2) 本地验证语法正确性与参数传递正确性。

### P1.2 TPU 裸金属环境 (rjx-v5p-8) 自动测跑 4 Arms — gate: 跑通 optax_d1, stream_d1, stream_d4_fp32, stream_d4_bf16 四个关键对照组
1) 触发 SSH 在远端 `rjx-v5p-8` 上执行 4 Arms 对比实验。
2) 收集全量的 `[[MEM]] peak_hbm_gb` 内存峰值和单步训练耗时并存入凭证文件。

### P1.3 定量闭环与结果归档 — gate: 定量数据严格对齐公式，xprof 转入 GCS 桶且 Update 算子被验证为最优 multiply-add
1) 验证：
   - `stream_d1` 必须与 `optax_d1` 一致（~11.72 GB HBM）。
   - `stream_d4_fp32` 仅增加 fp32 累加器开销（约 16.10 GB HBM）。
   - `stream_d4_bf16` 降为 bf16 累加器开销（约 13.97 GB HBM）。
2) 将 xprof 数据转存到 `gs://yuxzhang-tunix-models/issue21_repro_xprof/fix_accum_fp32/` 并验证算子融合形式。
3) 更新实验总结报告或文档并 push 到远端仓库。
