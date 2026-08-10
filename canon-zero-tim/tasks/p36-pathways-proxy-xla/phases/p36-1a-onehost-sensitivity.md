# P36.1a — Direct-attached one-host sensitivity control

- Status: passed

## Finding

- Confirmed: `aaron-v5p-node6` was a READY and HEALTHY v5p-8 with no user TPU workload running.
- Confirmed: Both arms used image ID `418dc632edd8` and probe SHA-256
  `faf65c53223c8ccf1b7d5545084aefe1eabb0918d88ea43127e61ecc577b602f`.
- Controlled variable: the OFF arm used `--xla_cpu_max_isa=AVX2`; the ON arm additionally used
  `--xla_allow_excess_precision=false`.

## Result

Both arms completed all eight registered depths over four TPU devices.

| Depth | OFF differing bytes | ON differing bytes | Reduction |
|---:|---:|---:|---:|
| 4 | 70,556/262,144 | 8,438/262,144 | 88.04% |
| 8 | 92,633/262,144 | 8,130/262,144 | 91.22% |
| 12 | 103,437/262,144 | 7,862/262,144 | 92.40% |
| 14 | 106,973/262,144 | 7,819/262,144 | 92.69% |
| 15 | 108,524/262,144 | 7,711/262,144 | 92.89% |
| 16 | 109,593/262,144 | 7,652/262,144 | 93.02% |
| 20 | 114,026/262,144 | 7,102/262,144 | 93.77% |
| 24 | 116,607/262,144 | 7,234/262,144 | 93.80% |

The flag is a strong carrier in this direct-attached generic THIRDPROG probe, but it is not the
only carrier: the ON arm remains non-bitwise. This probe does not include the complete Qwen
F4-embed, Pallas and canonical-operator configuration and cannot predict that the Pathways arm
will become exactly zero.

## Evidence

- `artifacts/p36_onehost_excess_off.raw.log`, SHA-256
  `a24139e233e435939fbc694e53c180ba9098b79f26ca35cd594769697a9581f2`
- `artifacts/p36_onehost_excess_on.raw.log`, SHA-256
  `25673f20e07249e2660b8a1374a831457b14eeace06b2e010412dbce66b880ba`
- `artifacts/p36_onehost_excess_pair.driver.log`, SHA-256
  `991e99e7b102fde3eaf3655bf76c7143ff389491f4f79c819d5f78c9572e8406`

## Next

Run P36.2 on the Pathways proxy. Do not promote the one-host control into a Pathways verdict.
