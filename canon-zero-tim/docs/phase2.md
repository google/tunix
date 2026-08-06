# Phase 2 — 可移植性修复

Status: **✅ 闭环(2026-08-05)**
Scope: 消除 shim 链的硬编码部署路径;把 22 个运行时模块搬进包;提供 `install.sh` 组装。

---

## Finding

P0.F2 已证 shim 链的**每一层**都把下一层的位置写死成绝对路径:

```
linear_p22xk.py:12   BASE_PATH = "/mnt/disks/tunix-data/claude_work/linear_p22xi.py"
linear_p22xi.py:11   → linear_p22xf.py
linear_p22xf.py:11   → tpu_inference_src/linear_patched.py
qwen3_p22xk.py:11    → qwen3_p22xh.py
qwen3_p22xh.py:17    → tpu_inference_src/qwen3.py          (两个模型变体各一处)
qwen2_p22xk.py:12    → qwen2_p22xj.py
qwen2_p22xj.py:11    → qwen2_p22xg.py
qwen2_p22xg.py:11    → tpu_inference_src/qwen2_patched.py
```

共 **9 处**。这是阻断级缺陷而非整洁度问题:换主机后 import 失败 ⇒ 引擎回落 stock 模块 ⇒
**run 全绿但根本没跑过干预**(与坑 #6「无 PATHTRACE = 干预未命中」同类,但更隐蔽,
因为连 PATHTRACE 的代码都没被加载)。

**模型特化**(P0.F2):`p22xf_contract.py` 与 `qwen3_p22xh.py` 三处副本互不相同 ——

| 文件 | claude_work | qwen1p7b | qwen8b |
|---|---|---|---|
| `p22xf_contract.py` | 182 行 `0c3bfb40…` | **160 行 `a1624d49…`** | **176 行 `8c3a5f08…`** |
| `qwen3_p22xh.py` | 139 行 `04cdf89b…` | **167 行 `2615cad6…`** | **161 行 `11b6ecbd…`** |

release 配置用的是**模型目录里的那份**(PYTHONPATH `$Q17:$Q8:$REPO:$W` 顺序遮蔽 +
`-v "$Q17/qwen3_p22xh.py":"$W/qwen3_p22xh.py":ro` 覆盖绝对路径)。
claude_work 里的通用版**不被任何 release 配置使用**,不进包。

## Execution

- [x] **P2.1** 把 20 个模型无关运行时模块 + 4 个模型特化模块搬进 `src/engine_shims/`
- [x] **P2.2** 新增 `canon_shim_root.py`:按 `$CANON_SHIM_ROOT` → 自身所在目录 两级解析
- [x] **P2.3** 把 9 处 `BASE_PATH` 改写为 `__import__("canon_shim_root").resolve("<sibling>")`
      —— **每个文件恰好改一行**,不新增 import
- [x] **P2.4** 写 `install.sh`:抽 stock → 打 6 个 patch → 铺 shim 链 → 按 MANIFEST 校验
- [x] **P2.5** 生成 `MANIFEST.sha256`(29 条 = 21 shim + 2 模型特化 + 6 patch 产物)

## GATE 结果

**① 每个文件相对源文件的改动行数**

```
13 个支撑模块(rpa_diff, rpa_diff_chunked, p22_pallas_*, p22x*_contract,
              p22xi_padded_matmul, p22xj_padded_swiglu, p22xk_vjp_ops)   改动 0 行  ✅
 7 个 shim 层(linear_p22x{f,i,k}, qwen3_p22xk, qwen2_p22x{g,j,k})        改动 2 行  ✅
 2 个模型特化 p22xf_contract                                             改动 0 行  ✅
 2 个模型特化 qwen3_p22xh                                                改动 2 行  ✅
   (改动 2 行 = diff 的 1 删 + 1 增,即恰好一行被替换)
```

**② 包内零硬编码部署路径**

```
grep -rn "/mnt/disks" src/engine_shims --include='*.py'  →  无输出   ✅
```

> 首次跑该 gate 时命中了 `canon_shim_root.py` docstring 里作为说明文字的字面路径。
> **未放宽判据**,改写了 docstring —— 让"零硬编码路径"这句话字面为真。

**③ 语法与解析器**

```
py_compile   26/26 通过                                                  ✅
解析器在非 /mnt/disks 路径下:默认解析到自身目录 / env 覆盖生效           ✅
```

**⇒ P2 GATE PASS**

## 未验证(推迟到 P6)

- 链在**真实引擎内**能否 import 成功、`[PATHTRACE]` 是否照打 —— 需要镜像内的
  `tpu_inference` 包,主机上无法 import(链底文件 `from tpu_inference... import` )。
  这是 P6 round-trip 的职责,**不得**用 `py_compile` 冒充。

## Result log

**2026-08-05 · P2 闭环**

已验证 by:`diff | grep -c '^[<>]'` 逐文件计数(0 或 2,无第三种);
`grep -rn "/mnt/disks"` 空输出;`py_compile` 26/26;`PYTHONPATH=<tmp> python3 -c` 实测解析器
两条路径(默认 = 自身目录、`CANON_SHIM_ROOT` 覆盖)且断言结果不含 `/mnt/disks`。

未验证:引擎内真实 import 与 PATHTRACE(推迟 P6)。

顺带修正的一处重复:`src/rpa_diff_chunked.py` 与 `src/engine_shims/rpa_diff_chunked.py`
是同一文件的两份副本(会漂移)⇒ 只保留 `engine_shims/` 下那份,`install.sh` 里的冗余 `cp` 已删。

产物:`src/engine_shims/`(25 个 .py)· `install.sh` · `MANIFEST.sha256`(29 条)
