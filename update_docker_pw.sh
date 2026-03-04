cd ~/deepswe
docker build -f Dockerfile -t gcr.io/cloud-tpu-multipod-dev/sizhi-deepswe-dev-image:latest .
docker push gcr.io/cloud-tpu-multipod-dev/sizhi-deepswe-dev-image:latest
