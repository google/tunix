This files lists the changes I made to the Qwen3 code to turn it into Qwen3-VL.
The primary goal of this file is to evaluate the possibility to merge the two
models. Main argument for or against the merge is whether we expect the two
models to diverge.

1. Extended config (`ModelConfig.vision_config`).
1. Vision encoder (`Qwen3VL.visual`).
1. More rules in `params.py`.
1. Added `param_dtype` (need to add to Qwen3 anyway).
