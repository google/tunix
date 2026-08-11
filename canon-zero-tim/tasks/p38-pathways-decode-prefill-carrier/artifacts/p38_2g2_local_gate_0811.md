# P38.2g2 local construction gate — 2026-08-11

Status: PASS locally; target Pathways numerical run NOT RUN.

## Source identities

- pinned image ID:
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
- patch 08 SHA-256:
  `3486a601c14a52b8748fb3a773dd497e411bf2b6344161e356d8a6d61c787996`;
- patch 09 SHA-256:
  `b2cd3a002749f0e02c1adda698472221a6b85a22743ca45e88b63fa5a5a0bfa1`;
- installed `tpu_runner_p21_l30.py` SHA-256:
  `adacdaeeda7a73a44c8f0af4a6866e6aebd077f7a90338537da7d4fd0faa579a`;
- installed `attn_iface_patched.py` SHA-256:
  `804d33649b8585a26b005822b26625d463b6c558f4d8b87bf8b7284d339b3e97`.

## Reproduction

From the source worktree:

```bash
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_extract_p38_serving_archive.py
python3 -m unittest discover \
  -s canon-zero-tim/tests/p38_serving -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_postflight.sh
bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh
sudo docker run --rm \
  -v "$PWD:/workspace" -w /workspace -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
git diff --check
```

Observed results:

- serving classifier: 18/18 PASS;
- archive transport: 4/4 PASS;
- stock/U renderer: 5/5 PASS;
- postflight:
  `[P38.SERVING] POSTFLIGHT_PASS exact_stop=accepted red_stop=rejected stock_hit=rejected unified_missing=rejected unified_exact=accepted`;
- exact-image Qwen3-1.7B overlay: 29/29 manifest, runtime 13/13;
- exact-image Qwen3-8B overlay: 29/29 manifest, runtime 13/13;
- exact-image terminal marker:
  `P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`;
- complete P33 CPU terminal marker:
  `[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1 p35_stage_probe=1`;
- new preflight marker:
  `[P38.SERVING] PREFLIGHT_PASS bounded=accepted partial_and_unbounded=rejected`;
- `git diff --check`: PASS.

## Claim boundary

This proves only that the patches install from the pinned source, the capture
contract fails closed, the manifests isolate stock and U, and evidence can be
recovered from a pod log. It does not prove that a production continue-decode
record was captured or that U changes any logprob. A clean write-only RPA v3
arm is not present.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR` and `CANON_KV_UNIFIED` unset. The stock
attention and runner branches remain selected. Discard patches 08/09 and their
manifest, renderer, test, and documentation entries to remove the diagnostic.
