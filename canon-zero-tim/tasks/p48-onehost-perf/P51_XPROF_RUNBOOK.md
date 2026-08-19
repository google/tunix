# P51 一宿主 GSM8K xprof/perfetto 画像 Runbook(2026-08-18)

> perf 线程(THREADS #2)载具文档。状态看 `state.md`,开关语义看 `../../FLAGS.md`,
> 产出证据登记在 `../../EVIDENCE.md` 的 run index。Rule zero(`../../AGENTS.md`)照旧:
> 失败 run 目录一律保留,基建失败记 INCONCLUSIVE,不删不改。

一条命令,在授权 v5p-8(`t1v-n-4a77ebd0-w-0`)上跑真几何 GSM8K 训练若干步,
捕获其中的 warm step,产出 XProf xplane + perfetto 双产物(含 device 轨)。

## 一条命令

```bash
cd <worktree>   # yuxzhang/canon-zero-tim 的任意 checkout
# 默认:3 步,捕 step2,python_tracer=0(device 轨必需,勿改回)
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <fresh-label>

# 推荐口径(skip 更多、测后段 warm step;发货优化全开):
P51_MAX_STEPS=6 P51_XPROF_SKIP_STEPS=4 P51_XPROF_STEPS=2 \
P51_BATCHED_REVERSE=1 \
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <fresh-label>
# ↑ 跑 6 步(0-5),捕获 step4-5 两个后段 warm step;约 20 分钟。
```

label 只允许 `[a-zA-Z0-9_-]`;同 label 不可复用(run dir 已存在即拒绝)。
长跑纪律:先 `cp` 冻结副本再发射(脚本运行中禁止编辑仓内 runner)。

## 旋钮表

| env | 默认 | 说明 |
|---|---|---|
| `P51_MAX_STEPS` | 3 | 总步数(step0 付编译 ~370s,此后 ~80-105s/步) |
| `P51_XPROF_SKIP_STEPS` | 2 | 完成 N 步后开始捕获(即捕 stepN 起) |
| `P51_XPROF_STEPS` | 1 | 捕获跨越的步数(1 步 xplane ≈1.9GB;2 步 ≈4GB) |
| `P51_XPROF_PYTHON_TRACER` | **0** | **保持 0**:开 python tracer 会把 device 轨挤掉(host-only 936MB 教训);要 python 栈时才临时开,并接受丢 device |
| `P51_XPROF_HOST_TRACER` | **1** | 与 python tracer 同理:jax 默认的 2 会用 host 事件把 device 采集挤掉。三发实测:python 开+host 2 → 纯 host 936MB;python 0+host 1 → 8 个 device plane / 2300 万事件;python 0+host 2 → 纯 host 32MB。要 host 细节时才调 2,并接受丢 device 轨 |
| `P51_BATCHED_REVERSE` | (关) | P52 优化;profile 当前最优配置时置 1 |
| `P51_ONEHOST_TIMEOUT_SECONDS` | 2700 | 6 步 ≈17 分钟,余量足;加步数需同步加 |

超参里**不用也不能动**的(合同 fail-closed 强制,错即秒红):
prompt/response=1024/1024、batch4×gen8(32 轨迹)、mini/micro/logps=4
(更新几何 32→16×2)、`CANON_VJP2_MAX_SEQS=1`、mesh fsdp1×tp4、
`max_num_seqs=32`、`batched_tokens=256`、beta=0.04(TRAIN 自动)。
发货 perf envs(`CANON_BATCHED_EVIDENCE=1`、`CANON_P28_BATCHED_REPORT=1`)
已内置。

## 产物与判读

```
/mnt/disks/tunix-data/logp_probe_1host/p51_gsm8k_xprof_<label>/train/xprof/plugins/profile/<ts>/
├── *.xplane.pb              # XProf 原生(device 轨在此)
├── perfetto_trace.json.gz   # ui.perfetto.dev 直接拖入
└── *.trace.json.gz          # Chrome trace 格式
```

绿判据(全部必须成立,任一不成立打印 `RED` 并非零退出):
`docker_exit=0`、`steps_done == MAX_STEPS`、**mesh 自证行**
(程序自己打印的 `axis_names=('fsdp', 'tp') axis_types=(Auto, Auto)`)、
以及 `P51_CAPTURE=1` 时的 `xprof_started=1 xprof_stopped=1 xplane>0 perfetto>0`。
数值门照旧:[CANON_ALIGN] 任何一红即停。

**为什么读程序自报的 mesh 而不是读 env**:env 断言写在 `bash -lc "..."` 里若不转义,
会被宿主 shell 提前展开成恒真(实测:容器里 `FL_SHARED_MESH=1,4` 时未转义断言仍
看到空串)。容器内断言已转义并会打印实际值,gate 再用程序自报的 mesh 行复核。

**看图**:
```bash
pip install --user xprof   # 一次性;解析/服务不需要 TPU
~/.local/bin/xprof <run_dir>/train/xprof --port 8791
# 本地: ssh -L 8791:localhost:8791 t1v-n-4a77ebd0-w-0 → http://localhost:8791
```
或 perfetto_trace.json.gz 拖 ui.perfetto.dev(零安装)。

## 导出到 GCS(另一台机器/另一个 agent 消费)

```bash
# 随 run 自动导出(推荐):
P51_GCS_EXPORT=1 P51_MAX_STEPS=6 P51_XPROF_SKIP_STEPS=4 \
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <label>

# 或对已完成的 run 单独导出:
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/persist_p51_xprof_gcs.sh \
  /mnt/disks/tunix-data/logp_probe_1host/p51_gsm8k_xprof_<label>
```

抄自 P38 的持久化模式(`tasks/p38-…/scripts/persist_p38_gcs.sh`):
gcloud→gsutil→google-cloud-storage 三级 fallback、SHA256SUMS 清单、
**上传后下载回读 cmp 校验**、`COMPLETE.json` 完成标记(已存在即拒绝重传)。
目的地:`gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p51/<label>/<session>/`
(`P51_GCS_PREFIX` 可覆盖,但必须留在该 bucket 根下)。
上传内容:`*.xplane.pb`(含 device 轨)、`perfetto_trace.json.gz`、
`*.trace.json.gz`、`driver.log`、`raw.log`、`SHA256SUMS`。

导出是证据处理,**不是捕获判据**:上传失败只报不炸,本地产物照旧可用。
**未在本机验证**(此宿主无 GCS 凭据/网络路径);首次在有凭据的机器上跑时,
先看 `[P51.GCS] UPLOADED`/`COMPLETE` 行与回读 cmp 是否通过。

消费端(另一台机器):
```bash
gcloud storage cp -r gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p51/<label> .
cd <label> && sha256sum -c <session>/SHA256SUMS
pip install --user xprof && ~/.local/bin/xprof <session> --port 8791
```

**脚本化读数**(host 织物聚合,产出 P52 那种 top-op 表):
`xprof.profile_data.ProfileData.from_file(<xplane>)` 逐 plane/line/event
聚合;host C++ 事件过滤 `$` 前缀(python tracer 行);device 轨看
`/device:TPU:*` 的 `XLA Modules`/`XLA Ops` line。

## 已知事实(引用 run 证据)

- 捕获 run 的步时**只看形状,不进 A/B**(tracer 扰动 ~+14s/步)。
- python_tracer=0 是 device 轨的前提(r6 vs rX 对照;归因保留一行:
  该实验同时动了捕获步序,机理级单变量隔离未做,操作配方已实证)。
- 训练捕获曾五探针排查(体积/env/引擎/补丁全洗清)——档案见
  外层 tasks/p51_onehost_gsm8k_xprof/phase1.md。
- 本载具钉授权宿主 `t1v-n-4a77ebd0-w-0`(脚本 preflight 断言)。
- **本载具不传 `FL_SHARED_MESH`**(2026-08-19 起):1×4 形状由
  `--mesh_fsdp/--mesh_tp` 给定,而该 env 在当前 tip 上不是"形状"——它切换整套
  mesh 程序(轴名 `(data,model)`、Explicit 轴类型、参考模型换 vocab 分片、
  `data_sharding_axis` 换轴),一宿主会死在 lm-head 的
  `Mesh for all inputs should be equal`。容器启动前有
  `test -z "${FL_SHARED_MESH:-}"` 自证,日志有
  `mesh_regime=fsdp,tp axis_types=Auto` 一行。
  **副作用**:alignment 报告 JSON 的 `context.mesh` 字段会是空串
  (`tunix/rl/alignment.py` 拿该 env 当标签),无判据读它。
  **其余一宿主载具(pair / FL P31 / P35 / P38 onehost / resident)仍在传该
  env**,在当前 tip 上会撞同一堵墙——尚未验证,见下方"未决"。
- **未决**:L3 pair 中立性门(`p41-optimizer-residency/scripts/run_onehost_pair.sh`)
  是否已被上述 mesh 制式改动打断,只有实跑能定;它是 perf 开关字节中立性的
  签字载具,建议在 TPU 空闲时跑一发确认。

## DP16/Pathways 注意

钩子(`CANON_XPROF_DIR` 等)在 learner 里、理论上随 profile .env 可达
64 卡,但 **Pathways/IFRT proxy 下未测**:device 采集经代理是否工作未知,
多 host 需 pathwaysutils 配合;用户既有裁决为 DP16 不做 xprof。真要试,
只在 canary 渲染上加 env、预期最多拿到 host 轨,不要用于任何判决。
