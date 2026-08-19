# P51 一宿主 GSM8K 性能画像 Runbook(2026-08-19 重写:官方 tunix/perf 栈)

> perf 线程(THREADS #2)载具文档。状态看 `state.md`,开关语义看 `../../FLAGS.md`,
> 证据登记在 `../../EVIDENCE.md`。Rule zero(`../../AGENTS.md`)照旧:失败 run
> 目录一律保留;profile run 的步时**只看形状,不进 A/B**。

一条命令,在授权 v5p-8(`t1v-n-4a77ebd0-w-0`)跑真几何 GSM8K 训练若干步,
经**官方 tunix/perf 栈**同时产出两台仪器的产物。

## 两台仪器(一次 run 全出)

| 仪器 | 实现 | 产物 | 体积 | 看什么 |
|---|---|---|---|---|
| **语义时间线**(seqpack 同款,官方 docs 的 Metrics/Perfetto) | `tunix/perf` v2 spans(learner 内建 + G6 训练段补装)+ `PerfMetricsExport.from_cluster_config` | `train/perf/perfetto_trace_v2_<ts>.pb` | ~20KB | 阶段结构:rollout / environment / reference_inference / weight_sync / data_loading / **peft_train(G6 update 整段,内嵌 segmented_value_and_grad 与 gradient_commit)**,ui.perfetto.dev 直接拖。训练段 span 是本分支补的:官方内建训练 span 长在 `PeftTrainer.train()`,G6 segmented 路径不进去 |
| **器件织物**(XProf) | `P51_XPROF_PHASE=step`:官方 `tunix/sft/profiler.Profiler` 步边界窗;`=update`:learner 在 G6 update 入口起窗 | `train/xprof/plugins/profile/<ts>/{*.xplane.pb, *.trace.json.gz}` | step ~1.9GB(缓冲上限)/ update 远小 | **step 模式的 device 轨只有 engine 前 ~25s decode**(缓冲截断,见"窗口语义");看 trainer forward/backward/commit 用 `update` 模式。xplane 进 XProf UI,trace.json.gz 拖 ui.perfetto.dev(注意它也只含缓冲保住的部分) |

与旧版(≤2026-08-18)的差别:自制 start/stop 钩子退役,窗口由官方
`Profiler` 管(日志行变为 absl 的 `Starting/Stopping JAX profiler at step N`);
不再产 `perfetto_trace.json.gz`(那是 jax `create_perfetto_trace` 的产物,
官方类不传该参)——器件侧拖 perfetto 用 `trace.json.gz` 即可。

## 一条命令

```bash
cd <worktree>
# 默认:3 步,捕 step2(整步:rollout+update),语义 trace 恒开
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <fresh-label>

# 推荐口径(更稳态、发货优化全开):
P51_MAX_STEPS=6 P51_XPROF_SKIP_STEPS=4 P51_XPROF_STEPS=1 P51_BATCHED_REVERSE=1 \
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <fresh-label>

# 看 trainer backward(forward+16 dispatch+reverse+commit,rollout 不入镜):
P51_XPROF_PHASE=update \
bash canon-zero-tim/tasks/p48-onehost-perf/scripts/run_onehost_gsm8k_xprof.sh <fresh-label>
```

label 只允许 `[a-zA-Z0-9_-]`,不可复用;发射前先 `cp` 冻结副本(#11)。

## 旋钮表

| env | 默认 | 说明 |
|---|---|---|
| `P51_MAX_STEPS` | 3 | 总步数(step0 编译 ~366s,warm ~86-94s/步) |
| `P51_XPROF_SKIP_STEPS` | 2 | 完成 N 步后开窗(捕 stepN 起)。**≥1**:窗口开在"步完成"边界,step0 够不着;0/非法值发射前 REFUSING |
| `P51_XPROF_STEPS` | 1 | 窗口跨步数;须 `skip+steps<=MAX_STEPS`,否则 REFUSING(拼错≠"不想采")。**`update` 模式强制 =1**(>1 会跨进下一步 rollout,decode 重新灌满缓冲;载具与 learner 双层 REFUSING) |
| `P51_CAPTURE` | 1 | 0=只训练不采 xprof(**显式声明**;此时不传 `CANON_XPROF_DIR`)。语义 trace 不受此开关影响,恒开 |
| `P51_XPROF_PYTHON_TRACER` | **0** | 经 ProfilerOptions 传入。**本 image 上开高任一 tracer 都会把 device plane 挤掉**,三发对照:python1/host2→936MB 纯 host;python0/host1→1.85GB 满血 8 planes;python0/host2→32MB 纯 host |
| `P51_XPROF_HOST_TRACER` | **1** | 同上;要 host 细节调 2 需接受丢 device 轨 |
| `P51_XPROF_PHASE` | step | `step`=整步窗(device 内容被缓冲截断为 engine 前 ~25s,**profile rollout 用它**);`update`=G6 update 入口→步完成(**profile backward 用它**,rollout 不进镜头) |
| `P51_ONEHOST_TIMEOUT_SECONDS` | 2700 | 6 步 ≈17 分钟,余量足 |

合同钉死项(fail-closed 强制,错即秒红,不用背):prompt/response=1024/1024、
batch4×gen8(32 轨迹)、mini/micro/logps=4(更新几何 32→16×2)、
`CANON_VJP2_MAX_SEQS=1`、mesh fsdp1×tp4、`max_num_seqs=32`、
`batched_tokens=256`、beta=0.04。发货 perf envs
(`CANON_BATCHED_EVIDENCE=1`、`CANON_P28_BATCHED_REPORT=1`)已内置。

## 窗口语义

官方 `Profiler` 由 learner 在 `Global step N completed` 边界驱动
(G6 segmented 路径不进 `PeftTrainer.train()`,trainer 级激活在本载具
永不触发;且窗口必须罩 rollout):

```
step0 完成 → 计数1(step0 编译步,采不到)
step1 完成 → 计数2==SKIP → absl "Starting JAX profiler at step 2"
step2 全程 → 被采集(整步:rollout+update)
step2 完成 → 计数3==SKIP+STEPS → absl "Stopping JAX profiler at step 3"
```

两 tracer 压低后采集扰动 ≤0.5s(86.3s 采 vs 86.8s 不采)。窗口由步完成点
切分,rollout 由后台 producer 预取,窗口内可能混入邻步少量流水工作:
窗口级归因不受影响,逐步精确记账以 `[PERF]` 行为准。

### device 缓冲截断(step 窗的真实内容,必读)

**窗口时间跨度 ≠ device 轨内容**。TPU device trace 缓冲每核 ~283 万 op
事件;engine decode ~11 万 op/s/核,**~25 秒填满,之后静默丢弃**。
p54final 1.9GB xplane 解剖实证:每核 XLA Modules 行 13 种模块全是
decode 家族(`jit_run_model`/`jit_sample`/`jit_compute_and_gather`…),
跨度 24.9s,XLA Ops 283 万/核——**整步窗里 rescore、vag forward、
backward、adam 全部被 drop**。这也是 xplane 恒 ~1.9GB 的原因(缓冲
上限指纹;95/94/66s 三种窗长实测 1941/1995/1887MB 不变)。
TPU 侧无文档化的缓冲加大键(advanced_configuration 只有
tpu_trace_mode 等,收下未知键≠生效,不可赌)。

### `update` 窗(profile backward 的正解)

start 锚在 `_run_p28_g6_update` 调用前(=语义 `peft_train` span 打开处,
device 窗≡语义 span),stop 在步完成点;decode 不入镜,缓冲全花在
update 上:vag_forward + 16 个异步 dispatch + 流水化 reverse + adam
commit 完整保留。自证行(gate 逐一断言步号;此模式无 absl 行):

```
[P51.XPROF] phase=update armed step=<SKIP>
[P51.XPROF] phase=update started step=<SKIP> anchor=update_entry
[P51.XPROF] phase=update stopped step=<SKIP+STEPS> anchor=step_completed
```

## 绿判据(任一不成立打印 `RED` 并非零退出)

`docker_exit=0` · `steps_done==MAX_STEPS` · **mesh 自证行**(程序自打印的
`axis_names=('fsdp', 'tp') axis_types=(Auto, Auto)`)· `perf_v2_traces>=1`
(语义 trace 静默消失=红)· `P51_CAPTURE=1` 时按 phase 分支:`step` 读
absl Starting/Stopping 行,`update` 读三条自证行(armed/started==SKIP、
stopped==SKIP+STEPS,**精确步号**),另 xplane>0 + trace.json.gz>0。
数值门照旧:[CANON_ALIGN] 一红即停。

## 看图

```bash
# 器件织物(XProf UI;解析/服务不需要 TPU):
pip install --user xprof
~/.local/bin/xprof <run_root>/train/xprof --port 8791
# 本地: ssh -L 8791:localhost:8791 t1v-n-4a77ebd0-w-0 → http://localhost:8791

# 语义时间线 / 器件 trace:两个文件都能直接拖进 ui.perfetto.dev(零安装)
#   <run_root>/train/perf/perfetto_trace_v2_<ts>.pb      ← 阶段结构
#   <run_root>/train/xprof/.../t1v-…trace.json.gz        ← 器件+host 全量
```

脚本化读数:`xprof.profile_data.ProfileData.from_file(<xplane>)` 逐
plane/line/event 聚合(host C++ 事件滤 `$` 前缀);语义 pb 用
`perfetto.protos...perfetto_trace_pb2.Trace` 解析 track_event。

### 验证"backward 在场"(update 模式跑完必做)

```bash
pip install --user xprof   # 仅解析,不需要 TPU
python3 canon-zero-tim/tasks/p48-onehost-perf/scripts/census_xplane_modules.py <run_root>
# 期望:全部 8 个 TensorCore plane 逐一 backward=present + decode=absent
#       → CENSUS_GREEN rc=0;任一 plane 不满足 → CENSUS_RED rc=1
# step 模式跑同一脚本必 CENSUS_RED(decode only)——那不是坏,是该模式的定义
# 语义脚本同样口径:缺任一训练段 span → CENSUS_RED rc=1

# 语义 span 普查(peft_train/segmented_value_and_grad/gradient_commit 各 2×步数):
sudo docker run --rm -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$PWD":"$PWD" --entrypoint python3 tunix_frozenlake_image:vllm-tpu0.25.0 \
  "$PWD/canon-zero-tim/tasks/p48-onehost-perf/scripts/census_semantic_trace.py" <run_root>
```

注意:**别用 `grep -a` 在 xplane 二进制里搜模块名当判据**——host plane
不受 device 缓冲限制,任何模式下都提到 trainer 名字(p54final 负控:
grep 命中 7-8 处但 device Modules 行为零)。必须用上面的逐 plane 普查。

## 导出到 GCS(另一台机器/agent 消费)

`P51_GCS_EXPORT=1` 随 run 自动导出,或事后
`bash scripts/persist_p51_xprof_gcs.sh <run_root>`(P38 持久化模式:三级
fallback + SHA256SUMS + 回读 cmp + COMPLETE 防重传;目的地
`gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p51/<label>/…`)。
本机无 GCS 凭据 ⇒ 上传路径在有凭据机器首跑时看
`[P51.GCS] UPLOADED/COMPLETE` 与回读 cmp。

## 在别的宿主直接跑 demo(不经本载具)必读

tracer/窗口 pin **只活在本载具**。直接跑 `qwen3_grpo_demo.py` 想拿同样产物,
必须自设(否则官方默认 host=2/python=1,在本 image 实测必得 host-only):

```bash
CANON_XPROF_DIR=<dir> CANON_XPROF_SKIP_STEPS=2 CANON_XPROF_STEPS=1 \
CANON_XPROF_HOST_TRACER=1 CANON_XPROF_PYTHON_TRACER=0 \
CANON_XPROF_PHASE=update \
CANON_PERF_TRACE_DIR=<dir2> \
python3 examples/math_gsm8k/qwen3_grpo_demo.py …
```

(`CANON_XPROF_PHASE` 省略=step 整步窗——记住那是 engine 前 25s 织物;
要 backward 必须 `update`。)

mesh 制式按宿主定(勿互抄):本宿主 **不设** `FL_SHARED_MESH`
(engine 兼容 `(fsdp,tp)+Auto`);engine 为 `(data,model)+Explicit` 的宿主
(如 maxtext-single-host)当时需设 `FL_SHARED_MESH=1,4` 并配 embedder
out-sharding——但**该组修法已 land-and-revert**(`6daec65e`→`e26b70b3`),
本分支现状不支持 explicit-engine 宿主跑参考模型 encode;细节与逐宿主表:
`P51_GSM8K_ONEHOST_SHARDING_ERROR_REPORT.md` §4。

## 已知事实

- 本载具钉授权宿主(preflight 断言 hostname);容器内断言
  `FL_SHARED_MESH` 缺席并打印 `mesh_regime=fsdp,tp axis_types=Auto`;
- alignment 报告 JSON 的 `context.mesh` 字段在本载具为空串(该 env 不再传),
  无判据读它;
- 其余一宿主载具(pair / FL P31 / P35 / P38 onehost / resident)仍传
  `FL_SHARED_MESH`,统一处理挂起中;
- device plane 被高档 tracer 挤掉的机理未根除(五探针阶梯已档,
  外层 tasks/p51_onehost_gsm8k_xprof/phase1.md),操作配方已实证;
- **缩窗死路(已实证否决)**:壁钟 delay 子步窗(rollout_update,
  land-and-revert)不缩 xplane——95/94/66s 窗 → 1941/1995/1887MB,
  trace.json.gz 恒 ~41MB;根因即上文缓冲截断:decode 25s 就把缓冲
  填满,窗再长再短文件都 ~1.9GB。要小文件/要 backward 都走
  `P51_XPROF_PHASE=update`,不要再试时间平移。
