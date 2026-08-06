# Phase 3 — T0 层(纯 CPU,秒级)

Status: **✅ 闭环(2026-08-05)**
Scope: 把 fp64 数学 gate 打包成任何人拿到就能跑的单命令,并证明它 fail-closed。

---

## Finding

T0 是包里**唯一真正零依赖**的一层:不需要 TPU、模型、镜像、网络。
两个 gate 都只 import `rpa_diff_chunked`(`p19_vjp2_multiseq_gate.py` 79 行,按模块名 import,
无硬编码路径),因此搬进包后不需要任何改动。

**T0 证明什么、不证明什么**(必须写清,否则会被过度引用):
- ✅ 证明:VJP2 反向所微分的那个纯 JAX 合同,与 full-prefill oracle 是**同一个数学函数**
  —— fp64 下值相同、梯度到舍入级一致、有限差分交叉验证通过。这正是"VJP2 是合法 VJP
  而非 surrogate"的依据。
- ❌ 不证明:与真实 Mosaic kernel 的任何关系。kernel 同一性是 T1/T2 的职责。

## Execution

- [x] **P3.1** `p19_vjp2_multiseq_gate.py` 搬入 `tests/t0_cpu/`
- [x] **P3.2** 写 `tests/t0_cpu/run.sh`:fail-closed runner,**先断言每条测量行存在**再判值
- [x] **P3.3** 预注册门限并写进脚本(移动门限 = 作弊,须记 FAIL 而非放宽)
- [x] **P3.4** 写 `tests/t0_cpu/negative_control.sh` 并实跑

## GATE 结果

```
== T0.1  chunked-vs-full-prefill fp64 oracle ==
  [selftest] value |chain-oracle| = 0.000e+00      门限: 必须【恰好】0.000e+00
  [selftest] grad rel = 5.039e-16                  门限: <= 1e-12
  [selftest] FD best rel = 1.106e-08               门限: <= 1e-6
  [selftest] VERDICT: PASS

== T0.2  ragged multi-sequence VJP2 vs per-seq autodiff ==
  [msq] 3 seqs q_lens=[24,40,32] kv_lens=[40,56,32]
  [msq] value |Δ|=0.000e+00                        门限: 必须【恰好】0.000e+00
  [msq] grad rel: dq=4.492e-17 dk=6.828e-17 dv=5.082e-17   门限: <= 1e-12
  [msq] VERDICT: PASS

===== T0 PASS (2 gates, 7 measurements) =====   exit=0
```

实测值与 `SKILL.md` §4 G5 记载的一致(值 0 / grad 5e-16 / FD 1.1e-08 / 多序列 ~5e-17)。

**负控(判据自身的检验)**

```
== negative control for T0 run.sh ==
  REJECTED (exit 1)   N1 silent gate (prints nothing)
  REJECTED (exit 1)   N2 nonzero value residual
  REJECTED (exit 1)   N3 gradient error above threshold
  REJECTED (exit 1)   N4 good numbers but VERDICT: FAIL
===== NEGATIVE CONTROL PASS -- run.sh rejects all 4 arms =====
```

N1 直接对应坑 #24(**gate 没打印任何行 = 没测,不等于通过**);
N4 对应"只看数字不看 verdict"的漏判。

**⇒ P3 GATE PASS**

## Result log

**2026-08-05 · P3 闭环**

已验证 by:`bash tests/t0_cpu/run.sh` 实跑 exit=0,7 项测量全部落在预注册门限内且与
SKILL.md G5 记载值一致;`bash tests/t0_cpu/negative_control.sh` 实跑 exit=0,四个坏臂
**全部被拒**(每个 arm 用 stub 模块替换真实模块,再跑一份指向 stub 的 run.sh 副本)。

未验证:与真实 Mosaic kernel 的数值关系(不在 T0 范围,归 T1/T2)。

产物:`tests/t0_cpu/{run.sh,negative_control.sh,p19_vjp2_multiseq_gate.py}`

预注册门限(**不得为让 run 通过而放宽**):
`value == 0.000e+00`(无容差)· `grad rel <= 1e-12` · `FD rel <= 1e-6` · `msq rel <= 1e-12`
