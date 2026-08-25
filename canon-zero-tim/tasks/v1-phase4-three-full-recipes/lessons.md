# Lessons

- 2026-08-25 — Do not promote an overflow-safe observer into a numerical repair without magnitude and scaling evidence. A max-scaled L2 can distinguish finite values from NaN/Inf, but it can also make a wildly mis-scaled finite gradient appear admissible. Freeze the expected DP/TP/loss/accumulator algebra, locate the first bad boundary, and keep the optimizer transaction disabled until that boundary is explained.
