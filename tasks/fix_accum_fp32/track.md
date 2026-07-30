# Track Log — fix_accum_fp32
> Running log of every action for easy revert + debug.
> Newest entry at the TOP. Each entry: What / Commands / Verify / Revert / Status.

---

## 2026-07-29 — 官方 Optax MultiSteps 4-Arm 静态显存镜像对齐完成，Xprof 存入 GCS
What:     在 rjx-v5p-8 跑完官方 optax_d1, optax_d4_bf16_accum, optax_d4_fp32_accum, optax_d4_fp32_moments 四组对照，存入 optax_4arm_hbm.log 和 reported_issue.md。
Commands: gcloud storage cp -r /mnt/workspace/mem_repro_xprof/optax_4arm/* gs://yuxzhang-tunix-models/issue21_repro_xprof/optax_4arm/ && git commit ...
Verify:   实测静态显存：optax_d1=7.30GB(3W), optax_d4_fp32_accum=10.38GB(4W), optax_d4_bf16_accum=10.38GB(4W), optax_d4_fp32_moments=16.85GB(6W)。与我们自定义累加器的静态显存对决 100% 镜像一致！且证明我们自定义 d1 快速通道比官方 MultiSteps(k=1) 更优！
Revert:   git reset --hard HEAD~1
Status:   DONE

## 2026-07-29 — 触发原生 Optax MultiSteps 4-Arm 定量对齐验证 (rjx-v5p-8)
What:     添加 `experimental/mem_repro_optax_4arm_docker.sh` 并在裸金属 TPU `rjx-v5p-8` 启动对齐 Optax 原版 `MultiSteps` 的 4 组对照。
Commands: gcloud compute tpus tpu-vm ssh rjx-v5p-8 ... mem_repro_optax_4arm_docker.sh
Verify:   完成静态 HBM 显存对齐与 Xprof 数据收集。
Revert:   git revert c57dcfcf
Status:   DONE


## 2026-07-29 — 4-Arm 定量验证完成，HBM 精准对齐模型与公式，Xprof 存入 GCS
What:     在 rjx-v5p-8 跑完了 d1_default, d4_fp32_accum, d4_bf16_accum, d4_fp32_moments 四组对照并提取内存指标，上传 xprof 到 GCS，记录至 fix_accum_fp32_hbm.log 和 reported_issue.md。
Commands: gcloud storage cp -r /mnt/workspace/mem_repro_xprof/* gs://yuxzhang-tunix-models/issue21_repro_xprof/fix_accum_fp32/ && git commit ...
Verify:   实测 d1=11.72GB(0W accumulator), d4_fp32=15.46GB(2W accum, bf16 moments), d4_bf16=13.27GB(1W accum), d4_fp32_moments=19.81GB(+2W fp32 moments)。与理论完全严丝合缝一致！
Revert:   git reset --hard HEAD~1
Status:   DONE

## 2026-07-29 — 提交测试架构并触发 TPU (rjx-v5p-8) 4-Arm 定量对比验证
What:     添加 `--gradient_accumulator_dtype` 与测试脚本 `experimental/mem_repro_fix_accum.sh` 并通过 SSH 在 `rjx-v5p-8` 启动 4 组对比。
Commands: git push origin HEAD:yuxzhang/fix_accum_fp32 && gcloud compute tpus tpu-vm ssh rjx-v5p-8 ...
Verify:   完成数据收集并验证显存峰值。
Revert:   git revert 97e6a9f3
Status:   DONE


## 2026-07-29 — 初始化任务管理体系与四臂测试计划 (Scaffold tasks file model)
What:     按照 @phase-workflow 要求创建 todo.md, plan.md, track.md 与 lessons.md。
Commands: write_to_file tasks/...
Verify:   阅读确认各个 md 文件格式符合规范、内容准确反映了分支测试计划。
Revert:   rm -rf tasks/
Status:   DONE
