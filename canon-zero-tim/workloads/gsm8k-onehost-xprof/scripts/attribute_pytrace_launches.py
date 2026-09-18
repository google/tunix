#!/usr/bin/env python3
"""Attributes one captured update's device launches and host copies to Python frames.

Needs a capture taken with CANON_XPROF_PYTHON_TRACER=1 (an attribution
capture, not certification evidence).  Host PJRT_LoadedExecutable_Execute
events on the main line are zipped in order with the TPU:0 XLA Modules
events (one host execute per device launch), and each is attributed to the
innermost tunix-side Python frame active at that host time; host
shard_args events (implicit reshards and relabels) are attributed the same
way.  Usage: attribute_pytrace_launches.py <run_root>
"""
import glob, sys, bisect, collections, re
import numpy as np
from xprof.profile_data import ProfileData
root = sys.argv[1]
f = glob.glob(f"{root}/train/xprof/plugins/profile/*/*.xplane.pb")[0]
p = ProfileData.from_file(f)
TUNIX = re.compile(r"adapter|learner|gsm8k|grpo|rollout|cluster|vllm|runner|canon|tunix|qwen|demo|sampler|packing|reward|dataset|trainer|optim|peft|attention|metadata")
allf = []; tun = []; host_exec = []; shard_args = []; modules = []
for plane in p.planes:
    if plane.name == "/device:TPU:0":
        for line in plane.lines:
            if line.name == "XLA Modules":
                modules = sorted((e.start_ns, e.name) for e in line.events)
    if plane.name != "/host:CPU": continue
    for line in plane.lines:
        if line.name.startswith("main/"):
            for e in line.events:
                if e.name.startswith("PJRT_LoadedExecutable_Execute"): host_exec.append(e.start_ns)
        elif line.name == "python3":
            for e in line.events:
                n = e.name
                if n.startswith("$"):
                    if n.startswith("$builtins") or n.startswith("$<unknown>") or n.startswith("$jaxlib.utils") or n.startswith("$_operator"): continue
                    allf.append((e.start_ns, e.start_ns + e.duration_ns, n))
                    if TUNIX.search(n.split(":")[0]): tun.append((e.start_ns, e.start_ns + e.duration_ns, n))
                elif n.startswith("shard_args"): shard_args.append(e.start_ns)
host_exec.sort(); shard_args.sort()
class Stab:
    def __init__(self, frames):
        frames.sort(); self.S = np.array([a for a, _, _ in frames], dtype=np.int64); self.E = np.array([b for _, b, _ in frames], dtype=np.int64); self.N = [n for _, _, n in frames]
        n = len(frames); self.k = max(1, n.bit_length()); self.st = [self.E.copy()]
        for j in range(1, self.k):
            prev = self.st[-1]; w = 1 << (j - 1)
            cur = prev.copy(); cur[w:] = np.maximum(prev[w:], prev[:-w]); self.st.append(cur)  # st[j][i] = max E over (i-2^j, i]
    def innermost(self, t):
        idx = int(np.searchsorted(self.S, t, side="right")) - 1
        if idx < 0: return None
        pos = idx
        for j in range(self.k - 1, -1, -1):
            if pos < 0: break
            if self.st[j][pos] < t: pos -= (1 << j)
        return self.N[pos] if pos >= 0 and self.E[pos] >= t else None
    def chain(self, t, depth=14):
        out = []; idx = int(np.searchsorted(self.S, t, side="right")) - 1; pos = idx
        while pos >= 0 and len(out) < depth:
            for j in range(self.k - 1, -1, -1):
                if pos >= 0 and self.st[j][pos] < t: pos -= (1 << j)
            if pos < 0 or self.E[pos] < t: break
            out.append(self.N[pos]); pos -= 1
        return out[::-1]
tstab = Stab(tun); astab = Stab(allf)
stem = lambda name: re.sub(r"[(.].*", "", name).strip()
n = min(len(host_exec), len(modules))
by_frame = collections.Counter(); by_frame_mod = collections.Counter(); samples = collections.defaultdict(list)
for t, (ms, mname) in zip(host_exec[:n], modules[:n]):
    fr = tstab.innermost(t) or "<no tunix frame>"; m = stem(mname)
    by_frame[fr] += 1; by_frame_mod[(fr, m)] += 1
    if len(samples[(fr, m)]) < 1: samples[(fr, m)].append(t)
print(f"== device launches by innermost tunix frame (n={n})")
for fr, c in by_frame.most_common(30):
    mods = ", ".join(f"{m}x{k}" for (f2, m), k in by_frame_mod.most_common() if f2 == fr)[:160]
    print(f"{c:7d}  {fr[:70]:70s} {mods}")
print("\n== call chains (outermost -> innermost, jax frames included) for selected launches")
for key in [("$canonical_qwen3_adapter.py:9904 _p32_forward_group", "jit__multi_slice"), ("$canonical_qwen3_adapter.py:9849 _p32_group_chunk_inputs", "jit__multi_slice"), ("$agentic_rl_learner.py:2023 <lambda>", "jit_gather"), ("$agentic_rl_learner.py:2023 <lambda>", "jit_broadcast_in_dim"), ("$canonical_qwen3_adapter.py:14923 _transform_value", "jit_convert_element_type"), ("$canonical_qwen3_adapter.py:9904 _p32_forward_group", "jit_convert_element_type"), ("$canonical_qwen3_adapter.py:10405 _p32_reverse_group", "jit__multi_slice")]:
    for t in samples.get(key, [])[:1]:
        print(f"  {key[1]} @ {key[0][:50]}:"); 
        for fr in astab.chain(t): print("      ", fr[:110])
sa = collections.Counter((tstab.innermost(t) or "<no tunix frame>") for t in shard_args)
print("\n== shard_args host copies by innermost tunix frame")
for fr, c in sa.most_common(12): print(f"{c:8d}  {fr[:100]}")
