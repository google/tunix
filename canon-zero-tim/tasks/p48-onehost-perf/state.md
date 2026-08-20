# perf 线程 · 任务状态(tasks/p48-onehost-perf)

> 线程 #2 `perf`(见 `../../THREADS.md`)。本目录是**分支内**的 perf 载具与证据落点;
> 逐 phase 的执行台账在外层仓 `tasks/p48_onehost_perf_optimization/`、
> `tasks/p49_onehost_fl_perf/`、`tasks/p50_vjp2_dispatch/`、
> `tasks/p51_onehost_gsm8k_xprof/`、`tasks/p52_reverse_scaffold/`(不进版本控制)。
> 更新:2026-08-18。

## 当前状态

一宿主两条战线都已收官并推送;下一门是 DP16 一发。

| 优化 | 开关 | 实测 | 状态 |
|---|---|---|---|
| P47a(rollout 不请求 prompt logprobs) | 无(合并即生效) | 首 rollout call 21.7→15.0s | 已推送 |
| 证据取回批量化 | `CANON_BATCHED_EVIDENCE=1` | ~6s/步 | GSM8K profile 默认开 |
| P50-E report 窗合并 + P50-F grouped 移植 | `CANON_P28_BATCHED_REPORT=1` | FL 一宿主 -14.5%;DP16 预估 -250s/步(未测) | GSM8K profile 默认开 |
| P52 反向脚手架合并 | `CANON_P28_BATCHED_REVERSE=1` | GSM8K 一宿主 94.3→81.8s(-13.3%),issue 22.6→12.4s | 一宿主认证完;DP16 待 grouped 移植 |
| 层方向 scan | `CANON_P28_LAYER_SCAN` | =1 净负 -5%(**否决**);=verify/verify_rev 保留为仪器 | 见 FLAGS 否决区 |

## 本目录内容

```
P51_XPROF_RUNBOOK.md              一宿主 GSM8K xprof/perfetto 画像:一条命令 + 旋钮表 + 看图三法
scripts/run_onehost_fl_p31_profile.sh    FL 8B P31 画像载具(P48/P49/P50 的靶场)
scripts/run_onehost_gsm8k_xprof.sh       GSM8K 1.7B 真几何画像载具(P51/P52 的靶场;含捕获窗)
scripts/persist_p51_xprof_gcs.sh         捕获导出 GCS(抄 P38 持久化模式;本机无凭据,未实测上传)
scripts/run_onehost_canon_cached.sh      编译缓存载具
scripts/extract_perf.py                  [PERF] 行 → JSON 画像
evidence/p48g7/                          64 卡 GSM8K 性能日志 + 提取 JSON
```

## P54/P55:官方 tunix/perf 栈 + update 捕获窗 + 训练段 span(2026-08-20)

| 件 | 实现 | 产物 |
|---|---|---|
| 语义时间线 | `CANON_PERF_TRACE_DIR` → PerfMetricsConfig → learner 内建 v2 span;G6 训练段=每步一条扁平官方 `peft_train`(与 weight_sync 同落位,p55d 认证;首版三嵌套 span 画糊已 revert e8d4caaf,判决入 census 负控) | `perf/perfetto_trace_v2_<ts>.pb`(~20KB) |
| 器件织物 step 窗 | 官方 `tunix.sft.profiler.Profiler` 步边界驱动 | xplane ~1.9GB——**实为 engine 前 ~25s decode**(device 缓冲 ~283 万事件/核填满即丢,p54final 解剖实证;backward 不在其中) |
| 器件织物 update 窗 | `CANON_XPROF_PHASE=update`:G6 update 入口起窗→ 步完成点 | xplane ~1.5GB,**完整 backward**(census: block_pullback×1758/adjoint×17/optimizer 事务;decode 零) |
| 普查件 | `scripts/census_xplane_modules.py`(device 侧,pip xprof)/ `census_semantic_trace.py`(span 侧含落位判定,容器内跑) | CENSUS_GREEN/RED 判词 |

认证 run:`p54final`(step 窗基线)、`p55a/p55a2`(缩窗否决证据,
rollout_update land-and-revert)、`p55b`(CL5 默认路径回归)、`p55c`
(update 窗认证,自证行 2/2/3)、`p55d`(扁平 span+update 窗复验)——全部 3/3 步、51/51 全零,SHA 见
EVIDENCE run index。tracer 钉 python=0/host=1。运行方法唯一权威:
`P51_XPROF_RUNBOOK.md`(2026-08-19 P55 版:含缓冲截断必读节)。

## 下一个门

**DP16 一发**(等卡 + 用户渲染):用新 tip 渲染 64 卡 GSM8K,读
`[PERF] stage=p32_vag_reverse … adjoint=` 验 E 在 grouped 路径的收益,
并给 P52 的 grouped 移植定量(逐 rank 循环 ×16 杠杆)。

## 登记的开口(不追,但不装作不存在)

- 训练进程的 xprof 捕获需 `CANON_XPROF_PYTHON_TRACER=0` 才有 device plane;
  该结论的实验同时动了捕获步序(单变量机理隔离未做,操作配方已实证)。
- P52 的 P32 grouped 移植未做(一宿主判决已绿,移植是下一条小 CL)。
- GCS 导出脚本未在本机验证(无凭据);首跑看 `[P51.GCS] UPLOADED/COMPLETE` 与回读 cmp。
