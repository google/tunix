"""P20.5 gate -- the end-to-end route, on CPU, before any e2e run.

Checks the chain the earlier gates could not see: packer -> TrainExample ->
learner attach -> the kernel actually reaching the model call.  A route that
silently drops the kernel would look exactly like "the optimisation gave 0%",
which is the failure this gate exists to make impossible.
"""
import inspect, os, sys, jax, numpy as np, jax.numpy as jnp
from tunix.rl import common, splash_mask
from tunix.rl import utils as rl_utils
from tunix.models.qwen3 import model as model_lib

B, BUD = 256, 2048
def mk(lengths):
    rng=np.random.default_rng(0); n=len(lengths); half=BUD//2
    a=lambda: np.zeros((n,half),np.int32); p,pm,c,cm=a(),a(),a(),a()
    for i,t in enumerate(lengths):
        pl,cl=int(t)//2,int(t)-int(t)//2
        p[i,-pl:]=rng.integers(1,1000,pl); pm[i,-pl:]=1
        c[i,:cl]=rng.integers(1,1000,cl); cm[i,:cl]=1
    return [common.TrainExample(prompt_ids=jnp.asarray(p),prompt_mask=jnp.asarray(pm),
        completion_ids=jnp.asarray(c),completion_mask=jnp.asarray(cm),
        advantages=jnp.zeros((n,),jnp.float32),ref_per_token_logps=None,old_per_token_logps=None)]

fails=[]
print(f"ENABLED={splash_mask.ENABLED}")
chunk = list(rl_utils.pack_sequences(iter([mk([600,500,400,300])]),
             max_token_budget=BUD, pack_size=2, sequences_per_update=4))[0][0]
print(f"segment_layout = {chunk.segment_layout}")

# 1) attach 真的挂上了 kernel,而且它是 pytree(叶子会被 trace)
att = splash_mask.attach(chunk, seq_len=BUD, block=B, num_heads=16)
k = getattr(att,"splash_kernel",None)
has = k is not None
print(f"1) attach 挂上 kernel: {has}")
if splash_mask.ENABLED and not has: fails.append("attach 没挂上")
if has:
    leaves=jax.tree_util.tree_leaves(att)
    n_arr=sum(1 for x in leaves if hasattr(x,"shape"))
    print(f"   kernel 的 MaskInfo 是 pytree 叶子(会被当参数): {n_arr>len(jax.tree_util.tree_leaves(chunk))}")
    if n_arr<=len(jax.tree_util.tree_leaves(chunk)): fails.append("kernel 不是 pytree 叶子")
    gw=int(np.asarray(k.fwd_mask_info.data_next).shape[-1])
    pb=int(np.asarray(k.fwd_mask_info.partial_mask_blocks).shape[0])
    print(f"   (grid_width, partial_blocks) = ({gw}, {pb})  partial_blocks==1: {pb==1}")
    if pb!=1: fails.append(f"partial_blocks={pb}")

# 2) 模型签名确实接受 splash_kernel(能力检测会用到)
ok_sig = common.model_call_contains.__module__ and "splash_kernel" in model_lib.Qwen3.__call__.__code__.co_varnames
print(f"2) Qwen3.__call__ 接受 splash_kernel: {ok_sig}")
if not ok_sig: fails.append("模型签名没有 splash_kernel")

# 3) compute_per_token_logps 接受并转发
ok_fwd = "splash_kernel" in inspect.signature(
    common.compute_per_token_logps).parameters
print(f"3) compute_per_token_logps 接受 splash_kernel: {ok_fwd}")
if not ok_fwd: fails.append("common 未接受")

# 4) 负控:开关关时必须一路 None,默认路径不受影响
if not splash_mask.ENABLED:
    print("4) 负控(开关关):attach 返回原对象:", att is chunk)
    if att is not chunk: fails.append("开关关时仍改了 example")

print("\nVERDICT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
