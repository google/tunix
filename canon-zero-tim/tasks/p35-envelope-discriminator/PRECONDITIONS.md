# P35 运行前置条件(硬性,先于任何新 envelope run)

## 1. envelope_probe 的 A 识别启发式已过时,先适配再跑(2026-08-13 起生效)

P47a 之后 rollout(A 臂)采样器请求 `prompt_logprobs=None`
(`tunix/generate/vllm_sampler.py`,cherry-pick c4ec573d 的语义)。
`tunix/rl/envelope_probe.py:362-366` 目前靠 A-records 捕获元数据里的
`num_prompt_logprobs` dict 枚举 A 请求 id 并断言
`native_A_observed == expected_a_rows` —— 这个前提是旧 `=0` 语义在 TPU/JAX
后端"仍然分配全段 prompt logprob 结构"的副产品。`None` 之后 A 记录大概率
不再携带该元数据。

**失效模式:`native_A_observed=False` 假红**(工具拒章,不是数值错误)。
危险方向(假绿)不存在:B 臂 rescore(`vllm_rollout.py` 的
`get_prefill_rescore_logps`)仍显式请求 `prompt_logprobs=0` 并消费返回值,未动。

**跑 P35 前必须做的事**:去读 capture shim 的写入侧 record schema,决定新的
A 识别判据(候选:A = 无 prompt-logprob 元数据的记录,恰与 B 相补——
`None` 语义下这个判别子比旧的更干净),改 `envelope_probe.py` 并配套
`canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py:469`
一类的 fixture,再进真实 run。

不适配就跑的后果:烧一轮 debug 追一个已知假红,且该轮 attestation 不可入证据链。

来源:P48 Phase 1(P47a)消费面审计,裁决记录见
`tasks/p48_onehost_perf_optimization/plan.md`(外层任务树)与该 CL 描述的缺点栏。
