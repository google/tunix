# Thin overlay on Yixuan's runner image for the Qwen3.5-35B-A3B GRPO run.
#
# One change, applied as a patch rather than a file copy so that a base image whose
# maxtext_engine.py has moved fails the build instead of silently reverting whatever
# moved. Nothing else in the image is touched: same tunix, maxtext, tpu-inference,
# vLLM, Raiden wheel and JAX pins as the base.
ARG BASE_IMAGE=gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8
FROM ${BASE_IMAGE}

COPY 0001-clear-payload-metadata.patch /tmp/

# `--forward` off and no fuzz: an already-applied or near-miss patch is a signal that the
# base image changed under us, and the run that follows would not be the one documented.
RUN cd /app/maxtext \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/0001-clear-payload-metadata.patch \
 && rm /tmp/0001-clear-payload-metadata.patch \
 && grep -q 'dataclasses.replace(payload, metadata={})' src/maxtext/training_engine/maxtext_engine.py \
 && python3 -c "import ast,sys; ast.parse(open('src/maxtext/training_engine/maxtext_engine.py').read())"
