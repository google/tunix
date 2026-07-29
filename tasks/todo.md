# Todo
> Human-owned. High-level goals only, no implementation steps.

1. 验证并论证 `yuxzhang/fix_accum_fp32` 分支对 fp32 累加器和 Adam 动量缓存精度修降 (Cast back) 的 HBM / 耗时 / 算子融合表现
   - outcome: 能够定量证明该分支在 Depth=1 保持 11.72GB 基线显存和最佳算子融合，在 Depth=4 能实现 float32 高精累加且不提升动量 HBM，并可通过配置选装 bf16 极致显存。
