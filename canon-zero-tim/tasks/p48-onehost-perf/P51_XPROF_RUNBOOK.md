# P51 一宿主 GSM8K xprof/perfetto 画像 Runbook(2026-08-18)

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
| `P51_XPROF_STEPS` | 1 | 捕获跨越的步数(1 步 xplane ≈2GB;2 步 ≈4GB) |
| `P51_XPROF_PYTHON_TRACER` | **0** | **保持 0**:开 python tracer 会把 device 轨挤掉(host-only 936MB 教训);要 python 栈时才临时开,并接受丢 device |
| `P51_XPROF_HOST_TRACER` | (jax 默认) | 1=精简 host 事件;2=含更多(默认足够) |
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

绿判据(driver.log 尾部):`SUMMARY steps_done=N align_pass=…
xprof_started=1 xprof_stopped=1 xplane_files>=1 perfetto_files>=1` +
`GREEN artifacts under …`。数值门照旧:[CANON_ALIGN] 任何一红即停。

**看图**:
```bash
pip install --user xprof   # 一次性;解析/服务不需要 TPU
~/.local/bin/xprof <run_dir>/train/xprof --port 8791
# 本地: ssh -L 8791:localhost:8791 t1v-n-4a77ebd0-w-0 → http://localhost:8791
```
或 perfetto_trace.json.gz 拖 ui.perfetto.dev(零安装)。

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

## DP16/Pathways 注意

钩子(`CANON_XPROF_DIR` 等)在 learner 里、理论上随 profile .env 可达
64 卡,但 **Pathways/IFRT proxy 下未测**:device 采集经代理是否工作未知,
多 host 需 pathwaysutils 配合;用户既有裁决为 DP16 不做 xprof。真要试,
只在 canary 渲染上加 env、预期最多拿到 host 轨,不要用于任何判决。
