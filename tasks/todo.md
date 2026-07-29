# Todo
> Human-owned. High-level goals only, no implementation steps.

1. [x] 验证并论证 `yuxzhang/fix_accum_fp32` 分支对 fp32 累加器和 Adam 动量缓存精度修降 (Cast back) 的 HBM / 耗时 / 算子融合表现
   - outcome: 2026-07-29 于 rjx-v5p-8 完成 4-Arm 实测！d1_default=11.72GB(零累加开销), d4_fp32=15.46GB(精确增加2W), d4_bf16=13.27GB(精简1W), d4_fp32_moments=19.81GB(+4W)。数据全维对齐理论并存入 GCS 桶。
