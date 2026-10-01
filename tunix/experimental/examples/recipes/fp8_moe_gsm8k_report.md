# FP8 routed-expert (MoE) weights in RL rollout / trainer: GSM8K comparison on v5p (Qwen3.5-35B-A3B)

> **Scope: FP8 applies only to the MoE routed-expert weights, not the full model.**
>
> - **FP8:** the 256 routed experts' `wi_0` / `wi_1` / `wo` in each of the 40 MoE layers.
> - **bf16 in every variant:** everything else — attention and GDN layers, the shared expert, the router, norms,
>   embeddings and the LM head — plus all FP32 paths (logits, gate logits).
> - The routed experts hold ~90% of the parameters (~32B of ~35B) but only ~1/3 of the per-token matmul compute
>   (8 of 256 experts are active per token).
> - Results here say nothing about full-model FP8.

All three variants ran 30 GRPO steps from the same checkpoint with the same prompts. No step had a non-finite
gradient and no update was skipped.

## TL;DR

Variant names read `t<trainer bits>r<rollout bits>` **for the routed experts only**; the rest of the model is bf16
in all three. The run script, pods and W&B runs use the names in the last column.

| variant | rollout: routed experts | trainer: routed experts | rest of the model (rollout and trainer) | script / W&B name |
|---|---|---|---|---|
| **t16t16** | bf16 | bf16 | bf16 | `bf16` |
| **t16r8** | FP8 (W8A8) | bf16 | bf16 | `r8` |
| **t8r8** | FP8 (W8A8) | bf16 master weights, FP8 fake-quant in the forward pass (straight-through gradients) | bf16 | `r8t8` |

Means over steps 1-29 (step 0 includes compilation, ~720-735 s for all three):

| variant | step time s | exposed gen s | train s | weight sync s | gen tok/s (step-level) | completion len | `tis/is_oob_ratio` | token logdiff absmean | token logdiff mean | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| t16t16 | 50.0 | 21.5 | 16.5 | 11.9 | 7328 | 352 | 0.582 | 0.0115 | -0.00095 | 0.901 |
| t16r8 | 47.7 | 20.0 | 16.1 | 11.6 | 8560 | 400 | 0.631 | 0.0138 | -0.00132 | 0.895 |
| t8r8 | 48.4 | 21.5 | 15.4 | 11.4 | 8215 | 391 | 0.588 | 0.0129 | -0.00106 | 0.886 |

Findings:

- **All three variants learn at the same rate.** This is one seed per variant.
  - They start from the same point: step-0 reward 0.59 / 0.58 / 0.60 (t16t16 / t16r8 / t8r8).
  - Each first reaches reward 0.9 at step 5, and completions shorten from ~500-600 tokens at steps 1-2.
  - Steady state (steps 9-29): reward 0.941 / 0.933 / 0.918, completion length 320 / 392 / 379 tokens.
  - t8r8's steady-state reward is ~0.02 below t16t16. With one seed and step-to-step swings of ±0.08, that gap
    is not resolved.
  - Gradient norms are finite throughout, with similar medians (0.131 / 0.117 / 0.139) and no skipped steps.
- **Token-level trainer/sampler mismatch.** `token_logdiff_absmean` is the mean |log p_trainer - log p_sampler|
  per token, so it doesn't depend on sequence length.
  - The FP8 rollout raises it by 20% (t16r8 0.0138 vs t16t16 0.0115). The mean bias grows from -0.00095 to
    -0.00132.
  - Fake-quant in the trainer removes ~40% of that excess (t8r8 0.0129, bias -0.00106). The trainer then computes
    with the same FP8-grid expert weights the rollout uses. What's left is mostly the rollout's activation
    quantization (W8A8), which the trainer doesn't model.
  - The same order holds at steps 0-2, before the policies diverge: 0.0177 / 0.0170 / 0.0179 (t16t16),
    0.0213 / 0.0199 / 0.0191 (t16r8), 0.0203 / 0.0180 / 0.0180 (t8r8).
  - Part of the steady-state gap may come from the trajectories themselves. The FP8 variants' completions are
    ~20% longer than t16t16's, and logdiff falls as completions get shorter and more confident.
- **OOB.** `is_oob_ratio` is the fraction of sequences whose geometric-mean trainer/sampler ratio falls outside
  the [0.999, 1.002] band, i.e. [-0.0010, +0.0020] in log space.
  - t8r8 (0.588) is back at t16t16's level (0.582). t16r8 is ~0.05 higher (0.631), in line with its larger
    mean bias.
  - On GSM8K, OOB is high mainly because the mean token log ratio sits on the band's lower edge: at -0.00095,
    roughly half of the sequences fall below 0.999.
  - Short sequences add noise on top. With a per-token spread of ~0.014 (from t16t16's mean |logdiff| of 0.0115),
    the mean log ratio of a ~350-token sequence has a standard deviation of ~0.0008, against a band 0.003 wide.
  - Across steps, OOB tracks the mean logdiff (correlation -0.44 / -0.61 / -0.35) rather than completion length
    (-0.03 / 0.28 / -0.05).
  - Don't compare these numbers with DeepSWE's, which uses the same band. Averaging over much longer trajectories
    shrinks the noise by √length, so OOB there is much lower for the same per-token mismatch.
- **Weight sync** is ~0.35-0.55 s faster with FP8 experts (11.6 s and 11.4 s vs 11.9 s). Halving the bytes of the
  expert tensors saves more time than quantizing them on sync costs.
- **Training time** doesn't go up with fake-quant (16.5 / 16.1 / 15.4 s). Per-step training time varies a lot
  (about 7-20 s), so these differences are noise.
- **Step time** is dominated by generation, and generation time depends on completion length and on the slowest
  sequence in each step. See the next section before reading speed into the step-time or tok/s columns.

### Generation speed: what the RL runs can and cannot tell

- **Step-level tok/s** = (completion_length_mean × seqs) / exposed_generation_time.
  - It is confounded by completion length and tail latency: a step waits for the slowest of 256 sequences.
  - The FP8 variants show ~12-17% higher step-level tok/s (8560 and 8215 vs 7328), but they also generate
    ~20% longer completions. Longer completions amortize the fixed per-step latency, so this is not a decode
    speedup measurement.
  - At steps 1-2, where lengths are closest (~500-600 tokens for all three), the numbers are close: 1885 / 1662
    (t16t16), 1943 / 1935 (t16r8), 1691 / 1522 (t8r8). t8r8's rollout is identical to t16r8's, so the spread
    between those two is noise.
- **vLLM's own 10 s "Avg generation throughput" windows** are mostly partly idle, because generation lasts only
  ~6-90 s per step, so there are too few clean full-batch windows to use.

**Conclusion:** the RL runs do not give a reliable decode-throughput number. The right follow-up is a controlled
benchmark: same prompts, fixed output length, fixed concurrency, one 2x2x1 v5p rollout slice, bf16 vs `fp8_moe`.

## Setup

- **Model and task:** Qwen3.5-35B-A3B on GSM8K, 30 steps, recipe `gsm_8k_35b_256.sh`:
  - GRPO-LOO, batch of 16 prompts × 16 generations;
  - max prompt and response length 1024, lr 1e-6;
  - `seq-mask-tis` with ratio band [0.999, 1.002];
  - seed 42, so all variants see the same prompts at each step.
- **Cluster:** `bodaborg-v5p-nap` (europe-west4), namespace `trellis`. Each variant uses 96 chips:
  - a 64-chip v5p trainer (4x4x4, Pathways, fsdp 32 × tp 2, tokamax `gmm_v2` with the recipe's fixed tiles);
  - 8 rollout replicas on v5p 2x2x1 (vLLM, expert parallelism 4);
  - raiden weight sync through the MaxText-to-MaxText weight converter (prefused MoE).
  - The three variants ran side by side, 2026-10-01 03:42-04:19 UTC.
- **FP8 recipe: MoE routed experts only, not the full model.** Only the routed experts' `wi_0` / `wi_1` / `wo` are
  FP8.
  - Format float8_e4m3fn, with per-output-channel scales of shape `(E, 1, out)` = amax / 448.
  - Attention and GDN layers, the shared expert, the router, norms, embeddings and the LM head stay bf16, and all
    FP32 paths are unchanged.
  - The rollout's fused MoE kernel runs W8A8.
  - t8r8's trainer rounds its routed-expert weights to the same grid in the forward pass (`fp8_moe_fake_quant`).
    Activations and gradients stay bf16, and the GMM backward kernels run in bf16 with f32 accumulation.
- **W&B:** entity `google-trellis`, project `trellis-gsm8k`, runs `igfp8v2-{bf16,r8,r8t8}-gsm8k-v5p`:
  [bf16](https://wandb.ai/google-trellis/trellis-gsm8k/runs/cofkd2xs),
  [r8](https://wandb.ai/google-trellis/trellis-gsm8k/runs/aqganc35),
  [r8t8](https://wandb.ai/google-trellis/trellis-gsm8k/runs/i529nji5).

Per-step metric keys in W&B (all under the `train/` prefix):

- `orchestrator/{step_time_sec,exposed_generation_time,policy_training_time,weight_sync_time}`
- `rollout/{completion_length_mean,global_valid_seqs}`
- `trainer/tis/is_oob_ratio`
- `trainer/{grad_norm,step_skipped}`
- `trainer/sampler_is/token_logdiff_{absmean,mean}`
- `rewards/mean`

## Per-step data

t16t16 (`bf16`):

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean    ld_mean  grad_norm  reward
    0   734.9  155.5    565.5   13.85      819.1  497.5 0.5996     0.01771  -0.001823     0.3356   0.593
    1   100.7  81.26    7.306   11.76       1885  598.3 0.5722     0.01703  -0.001424     0.5783  0.5309
    2   106.8  84.27    10.22   12.28       1662    547 0.5838     0.01789  -0.001601     0.4212  0.5977
    3   66.24   40.1    13.71   12.37       2828  442.9 0.5786     0.01435  -0.001312     0.2546   0.775
    4   80.77  54.84    13.68   12.19       1855  397.3   0.66     0.01437  -0.001037     0.2702  0.8355
    5   59.14  31.94     15.3   11.84       2855  356.2  0.643     0.01342  -0.001221     0.2368  0.9238
    6   60.83  34.29    14.32   12.17       2887  386.6 0.5568       0.012 -0.0008843     0.2829  0.8711
    7   58.96  32.27    14.72   11.91       2846  358.7 0.5838     0.01181 -0.0008323    0.06675  0.9852
    8   62.15   31.7    18.77   11.62       3256  403.2    0.6     0.01402  -0.001176     0.2375  0.8344
    9   37.94  9.991    16.37   11.51       8925  348.3 0.5661     0.01224 -0.0009053     0.1089  0.9691
   10   39.52  8.944    18.46   12.04  1.079e+04  377.1 0.5753     0.01151 -0.0006227     0.2834  0.9121
   11   38.47  9.623    17.06   11.72       9499  357.1 0.5599     0.01157  -0.000819    0.05019  0.9426
   12   61.55  30.45     18.2   12.84       2841  337.9 0.5477     0.01145 -0.0009943     0.1581  0.9445
   13   36.34  10.52    13.97   11.76       8435  346.8  0.676     0.01161  -0.001127     0.1257  0.9113
   14   56.59  27.66    17.09   11.79       2822  304.9 0.5742     0.01037 -0.0008653          0       1
   15   37.85  7.443    18.67   11.67  1.002e+04  291.3 0.6016    0.009663  -0.000861    0.02296  0.9965
   16   36.78   8.75    15.88   12.06       9671  330.6  0.605     0.01128  -0.001042     0.7488  0.8668
   17   37.55  6.992    18.78   11.71       9685  264.5 0.6473     0.01052  -0.001025    0.09492   0.975
   18   37.83  7.367    18.86   11.52  1.035e+04  297.8 0.6315     0.01022 -0.0009976    0.05598  0.9566
   19   37.96  8.931    17.22   11.74       9632    336 0.5919     0.01102 -0.0007571     0.1355  0.8957
   20   37.76  7.729    17.82   12.12  1.029e+04  310.7 0.6084     0.01113 -0.0009627    0.07806  0.9883
   21   56.39  29.14    15.28   11.91       3033  345.3  0.565     0.01108  -0.001245     0.9571  0.9449
   22   38.15  6.936     19.4   11.75  1.166e+04  315.9 0.5491     0.00994 -0.0006033    0.09931  0.8734
   23   37.36  5.842    19.03    12.4   1.19e+04  271.5 0.6094    0.009721 -0.0009706          0  0.9437
   24    38.3  9.519    16.38   12.34       9368  348.3 0.5534    0.009754 -0.0007246     0.4829  0.8363
   25   37.54  6.918    18.46   12.09  1.032e+04  278.9 0.5767    0.009699 -0.0007912     0.1182  0.9328
   26   37.76   9.08    16.88   11.73       9393  333.2 0.5215    0.008495  -0.000574     0.1054  0.9574
   27   36.42  5.809    18.91   11.64  1.167e+04  264.9 0.5352    0.008598 -0.0008347          0       1
   28   37.77  7.444    18.56   11.69  1.042e+04  302.9 0.5357    0.009058 -0.0007384    0.08011  0.9664
   29   38.82  7.683    19.14   11.93  1.171e+04  351.4 0.4779    0.008508 -0.0006854     0.1366  0.9477
```

t16r8 (`r8`):

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean    ld_mean  grad_norm  reward
    0   728.4  157.7    557.4   13.29      863.4  531.9 0.6876     0.02126  -0.002487     0.3279   0.577
    1   101.7  77.77    12.21   11.68       1943  590.2 0.6944     0.01987  -0.002106     0.4356  0.5211
    2    86.3   64.8    10.28   11.16       1935  489.8 0.7202     0.01915  -0.001976     0.3896  0.6602
    3   64.21  40.37    12.32   11.46       3003  473.6 0.6474     0.01685  -0.001917     0.2275  0.7406
    4   61.54  41.87     8.54   11.06       2508  410.2 0.6545     0.01717   -0.00141     0.3276  0.8098
    5   82.13  52.27    18.64   11.16       1556  317.8 0.6929     0.01475  -0.001215     0.1318  0.9457
    6   58.57  31.07    15.56   11.87       3031  367.9 0.6231     0.01422  -0.001529     0.1419  0.8773
    7   37.69  9.232    16.78   11.61       9582  345.6 0.6237     0.01362  -0.001412     0.1239  0.9703
    8   61.14  34.09    14.67   12.32       2899  385.9 0.6714     0.01686  -0.001336     0.2059  0.8285
    9   37.78  11.67    14.07   11.95       7951  362.6 0.6427     0.01409  -0.001287    0.07001  0.9824
   10    37.2  9.911    15.67   11.55  1.007e+04  389.7 0.6318     0.01464  -0.001427     0.2152  0.9066
   11   37.99  8.673    17.57   11.67   1.15e+04  389.6 0.5877     0.01434  -0.001347    0.03035  0.9504
   12   38.17  8.965    16.91   12.23  1.062e+04  371.8 0.6649     0.01359  -0.001598    0.07567  0.9582
   13    35.8  15.73    8.712   11.29       6686  410.7 0.6345      0.0133  -0.001164      0.166  0.8988
   14   57.19  26.85    18.93   11.35       3464  363.3 0.5748     0.01284  -0.000921    0.04629  0.9883
   15   37.93   7.44    18.52    11.9  1.177e+04  342.2 0.6133      0.0118  -0.001125          0       1
   16   36.02   9.17    15.44   11.34  1.118e+04  400.4 0.6631     0.01432  -0.001131     0.0755  0.8438
   17   36.38  6.713     18.3    11.3   1.21e+04  317.2 0.5703     0.01205  -0.001101    0.09439  0.9859
   18   38.88  8.359    18.18   12.28  1.137e+04  371.3  0.625     0.01166  -0.001309     0.6069  0.9434
   19   38.66  9.098    17.01   12.49  1.164e+04  413.7  0.644     0.01333  -0.001125     0.1496  0.8586
   20   38.14  8.643    17.84   11.59  1.181e+04  398.6 0.6135     0.01308 -0.0008791    0.09636  0.9559
   21   56.13  28.95    15.95   11.17       3970    449 0.5852     0.01273  -0.001451    0.03471   0.943
   22   37.59  8.204    17.96   11.34  1.345e+04  431.2 0.6328     0.01226  -0.001305     0.1317  0.8516
   23   37.82  7.372    19.25   11.13  1.282e+04  369.2 0.6523      0.0127  -0.001252          0  0.9437
   24   37.11  10.91    14.76   11.35  1.146e+04  488.5 0.6092     0.01363  -0.001316     0.4679  0.7859
   25   37.32  8.081    17.95    11.2  1.221e+04  385.5 0.6341     0.01272   -0.00132    0.04694  0.9324
   26   38.32  9.361    17.28    11.6  1.165e+04    426 0.5647      0.0107  -0.001002     0.1103  0.9602
   27   38.12  7.716     18.5   11.82  1.163e+04  350.6 0.6367     0.01272   -0.00118          0       1
   28      38  8.263    18.44   11.23  1.163e+04  375.4 0.6055     0.01105  -0.001218    0.08725  0.9789
   29   39.92  8.471    19.53   11.85  1.281e+04  423.8 0.5911     0.01101  -0.001008     0.1085  0.9336
```

t8r8 (`r8t8`):

```
 step  step_s  gen_s  train_s  sync_s  gen_tok_s  compl    oob  ld_absmean    ld_mean  grad_norm  reward
    0   721.7  159.3    549.9   12.49      823.5  512.4 0.6003     0.02032  -0.001545       0.44  0.6043
    1   104.9  86.61    7.613   10.64       1691  572.2 0.6172     0.01805  -0.001742     0.4605  0.5715
    2   104.1  83.64    9.231   11.15       1522  497.3 0.6063     0.01804  -0.001364     0.7916  0.6617
    3   63.79  38.42    14.18   11.14       2811  421.8  0.582     0.01581  -0.001209     0.2744  0.7812
    4   80.11  58.48    9.696   11.87       1748  399.4 0.5956     0.01491 -0.0008749     0.4168  0.8164
    5   60.59  30.69    18.54   11.31       2828    339 0.5322      0.0135  -0.001283     0.1513  0.9375
    6   58.58  29.44    17.85   11.24       3445  396.2 0.6125     0.01308  -0.001295     0.1285  0.8676
    7   61.95  31.07    18.66   12.16       2997  363.7 0.5579     0.01278  -0.001005     0.1023  0.9586
    8   37.19  9.378    16.36   11.39  1.069e+04  391.5 0.5784     0.01447  -0.001258     0.1933  0.8289
    9   57.12  32.53    12.31   12.23       2854  362.7 0.5797     0.01209 -0.0008657     0.1711  0.9715
   10   57.18  31.96    13.11   12.04       3073  383.6 0.6347     0.01257 -0.0009196    0.06601  0.8992
   11   37.33  8.051    17.94   11.28  1.196e+04    376 0.5618     0.01238  -0.001104     0.1344  0.9125
   12   36.05  8.108    16.91   10.97  1.135e+04  359.5 0.5657     0.01204  -0.001031     0.1337  0.9234
   13      35  16.45    7.007   11.47       6187  397.7 0.5568     0.01304 -0.0006134      0.361  0.8719
   14   35.82  7.996    16.48   11.27  1.041e+04  325.2 0.5823     0.01158 -0.0009319    0.05063  0.9887
   15   36.54  6.552    18.19   11.72  1.204e+04  308.1 0.5446     0.01102 -0.0009486    0.04895  0.9926
   16   35.44  9.087       15   10.99   1.16e+04  411.8 0.7065     0.01326 -0.0009339     0.1556   0.816
   17   34.91  6.221    17.33    11.3  1.143e+04  277.7 0.6719     0.01119  -0.001266     0.1442  0.9859
   18   37.09  7.704    17.91   11.41   1.17e+04  352.1 0.6185     0.01196  -0.001193     0.1201  0.9453
   19   37.42  7.982    18.28    11.1   1.39e+04  433.3 0.6097     0.01368  -0.001277     0.2207   0.841
   20   38.71  8.002    18.66   11.97  1.164e+04  363.9 0.6413      0.0126  -0.001295    0.03596  0.9609
   21   56.58  28.42    16.32   11.78       3607  400.4 0.6198      0.0134  -0.001181          0  0.9418
   22    36.2  7.315    17.56   11.26  1.297e+04  370.7 0.5776     0.01115 -0.0008786      1.556  0.8629
   23   36.38  7.178    18.28   10.85  1.215e+04  340.7 0.6062     0.01135  -0.001105    0.03902  0.9359
   24    36.8  14.85    11.07    10.8       8863    514 0.5031     0.01275  -0.000837     0.3571  0.7305
   25    37.8  7.917    18.56   11.25  1.249e+04  386.3 0.6246     0.01165    -0.0009     0.0607  0.9203
   26   37.39  11.93    14.36   11.02       9086  423.6 0.5393     0.01078 -0.0008307    0.08495  0.9477
   27   36.09  8.135    15.93   11.94  1.087e+04  345.3 0.5273     0.01083 -0.0008038          0       1
   28   36.94  9.565    16.29   11.03   1.07e+04  399.9 0.5182     0.01178  -0.001004    0.07782  0.9379
   29   38.03  9.354    17.35   11.26  1.163e+04  424.8 0.5773      0.0115 -0.0007246     0.9624  0.8891
```

## Reproduction

Code:

- **maxtext** `atwigg/mlperf` @ `f09d1fc94`. It has per-channel FP8 MoE, `rollout_fp8_moe`, `fp8_moe_fake_quant`
  and the weight converter support (#5473). It also has the `gmm_v2` lhs K-tail mask (#5482), which this recipe's
  trainer (tp 2, fixed tiles) needs when `TRAINER_FP8=true`.
- **tunix** `igorts/fp8-gsm8k-v5p` @ `84c141f07`, on top of `atwigg/mlperf` `a5f36057f`:
  - the `ROLLOUT_FP8` / `TRAINER_FP8` recipe switches (`fp8_moe.sh`, already on `atwigg/mlperf`);
  - run script `recipes/fp8_gsm8k_35b_v5p.sh` and report script `recipes/fp8_gsm8k_report.py`;
  - `requirements/maxtext_requirements.txt` pinned to maxtext `f09d1fc94`.
  - This report lives on the same branch, next to the scripts, one commit after `84c141f07`; it adds no code.

### 1. Build the image

```bash
B=/tmp/fp8/imgctx; rm -rf $B && mkdir -p $B/maxtext $B/tunix
(cd ~/git/maxtext && git archive f09d1fc94 | tar -x -C $B/maxtext)
(cd ~/git/tunix && git archive 84c141f07 | tar -x -C $B/tunix)
cat > $B/Dockerfile <<'EOF'
FROM gcr.io/cloud-tpu-multipod-dev/atwigg/trellis@sha256:490e18e45582edd9191f52fc7d08fe5928f26653102ef83bf7517e6a8e1e608b
COPY maxtext /opt/src/maxtext
RUN uv pip install --no-deps --reinstall /opt/src/maxtext /opt/src/maxtext/src/maxtext/integration/vllm
# /app/tunix is an editable install. Keep the base image's generated protobuf modules (gitignored).
RUN cd /app/tunix && find . -name '*_pb2*.py' -exec cp --parents {} /tmp/ \; -print && mkdir -p /tmp/pb2 && cd /tmp && find . -maxdepth 1 -name experimental -exec mv {} /tmp/pb2/ \; && rm -rf /app/tunix
COPY tunix/tunix /app/tunix
RUN cp -rn /tmp/pb2/. /app/tunix/ && rm -rf /tmp/pb2 && python -c 'from tunix.experimental.distributed.runtime.discovery import discovery_service_pb2'
# /app/examples must match the tunix pin too.
COPY tunix/examples /app/examples
COPY tunix/pyproject.toml /app/pyproject.toml
COPY tunix/requirements /app/requirements
EOF
cd $B && docker build -t gcr.io/cloud-tpu-multipod-dev/<you>/trellis:fp8 . && docker push gcr.io/cloud-tpu-multipod-dev/<you>/trellis:fp8
```

The runs above used `gcr.io/cloud-tpu-multipod-dev/igorts/trellis:fp8-v2`
(`sha256:f9159b8077ecd35767022d521b623fa86ca4cfa2c0da479b9f1f625bce1635d0`), built with exactly these steps. The
run script uses it by default; set `TUNIX_IMAGE` to use your own.

### 2. Launch (from a tunix checkout at `84c141f07`)

Each variant uses 96 v5p chips (288 for all three). The launcher switches the kubectl context to
`bodaborg-v5p-nap`.

```bash
R=~/git/tunix/tunix/experimental/examples/recipes
cd /tmp   # the launcher writes scratch files to the cwd
export RUN_PREFIX=<you>-fp8   # pod and W&B run names: $RUN_PREFIX-<variant>; default igfp8v2
for v in bf16 r8 r8t8; do $R/fp8_gsm8k_35b_v5p.sh $v start --dry-run > dry_$v.yaml; done   # optional: inspect
for v in bf16 r8 r8t8; do $R/fp8_gsm8k_35b_v5p.sh $v start; done
# stop (also needed after a run finishes: the trainer and rollouts keep running):
#   $R/fp8_gsm8k_35b_v5p.sh <variant> stop
```

What the variants set:

- `ROLLOUT_FP8=true` adds `"fp8_moe": true` to the vLLM `maxtext_config` and `rollout_fp8_moe=true` to the
  trainer's MaxText flags.
- `TRAINER_FP8=true` adds `fp8_moe_fake_quant=true`.
- `MAX_STEPS` (default 30) can be overridden from the environment.

A run takes ~50 min: ~15 min for the pods to be scheduled and start, ~12 min for step 0 (compilation), then
~50 s per step.

### 3. Collect metrics

```bash
pip install wandb pandas            # any venv; needs WANDB_API_KEY for the google-trellis entity
python $R/fp8_gsm8k_report.py -v --prefix=$RUN_PREFIX   # per-step table + means over steps >= 1
```

While a run is going, look at pods `$RUN_PREFIX-<variant>-{orch,train,roll-N}` in namespace `trellis`. The
orchestrator logs one `Train step N` line per step, and `Weight sync finished in X seconds`.
