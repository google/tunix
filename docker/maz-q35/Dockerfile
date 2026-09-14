# Thin overlay on Yixuan's runner image for the Qwen3.5-35B-A3B GRPO run.
#
# One change, applied as a patch rather than a file copy. A file copy would overwrite any
# other edit the base image has made to maxtext_engine.py; a patch fails the build instead.
# Nothing else in the image is modified: same tunix, maxtext, tpu-inference, vLLM, Raiden
# wheel and JAX pins as the base.
ARG BASE_IMAGE=gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8
FROM ${BASE_IMAGE}

COPY 0001-clear-payload-metadata.patch /tmp/

# `--forward` off and no fuzz. An already-applied or inexactly-matching patch means the
# base image differs from the one this overlay was written against, so the resulting run
# would not match the documented configuration.
RUN cd /app/maxtext \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/0001-clear-payload-metadata.patch \
 && rm /tmp/0001-clear-payload-metadata.patch \
 && grep -q 'dataclasses.replace(payload, metadata={})' src/maxtext/training_engine/maxtext_engine.py \
 && python3 -c "import ast,sys; ast.parse(open('src/maxtext/training_engine/maxtext_engine.py').read())"
