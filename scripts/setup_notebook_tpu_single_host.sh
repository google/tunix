#!/bin/bash

#
# Example usage:
#   bash setup_notebook_tpu_single_host.sh --tpu-zone us-central1-a --tpu-project cloud-tpu-inference-test
#   bash setup_notebook_tpu_single_host.sh --tpu-name ${USER}-v5e-8 --tpu-zone us-central1-a --tpu-project cloud-tpu-inference-test
# Then open http://localhost:8889/tree on your local machine
# The Jupyter server log is here: ~/jupyter.log

# --- Default values ---
DEFAULT_TPU_ZONE="us-central1-a"
DEFAULT_TPU_PROJECT="cloud-tpu-inference-test"
DEFAULT_IMAGE="" # Will be set based on ACCELERATOR if not provided
DEFAULT_ACCELERATOR="v5litepod-8"
DEFAULT_TPU_NAME="$USER-tpu-v5e-8"
DEFAULT_VM_USER="$USER"

# --- Initialize variables with defaults ---
export TPU_ZONE="$DEFAULT_TPU_ZONE"
export TPU_PROJECT="$DEFAULT_TPU_PROJECT"
export IMAGE="$DEFAULT_IMAGE"
export ACCELERATOR="$DEFAULT_ACCELERATOR"
export TPU_NAME="$DEFAULT_TPU_NAME"
export VM_USER="$DEFAULT_VM_USER"

# --- Parse named arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --image) IMAGE="$2"; shift ;;
        --accelerator) ACCELERATOR="$2"; shift ;;
        --tpu-name) TPU_NAME="$2"; shift ;;
        --tpu-zone) TPU_ZONE="$2"; shift ;;
        --tpu-project) TPU_PROJECT="$2"; shift ;;
        --vm-user) VM_USER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Conditional logic for IMAGE based on ACCELERATOR (if IMAGE was not explicitly set) ---
if [ -z "$IMAGE" ]; then # Only apply default logic if IMAGE was not passed as an argument
  if [[ "$ACCELERATOR" == *"v5litepod"* || "$ACCELERATOR" == *"v5e"* ]]; then
    IMAGE="v2-alpha-tpuv5-lite"
  elif [[ "$ACCELERATOR" == *"v6litepod"* || "$ACCELERATOR" == *"v6e"* ]]; then
    IMAGE="v2-alpha-tpuv6e"
  else
    # Fallback to a general default if accelerator doesn't match specific conditions
    # and no explicit IMAGE was given. You might want to make this more specific
    # or remove it if all your accelerators should match the above.
    IMAGE="v2-alpha-tpuv5-lite" # Or some other sensible default
    echo "Warning: ACCELERATOR type not recognized for specific IMAGE default. Using $IMAGE."
  fi
fi


# --- Output selected values (optional, for debugging) ---
echo "--- Using the following configuration ---"
echo "TPU_ZONE: $TPU_ZONE"
echo "TPU_PROJECT: $TPU_PROJECT"
echo "IMAGE: $IMAGE"
echo "ACCELERATOR: $ACCELERATOR"
echo "TPU_NAME: $TPU_NAME"
echo "VM_USER: $VM_USER"
echo "---------------------------------------"

# --- Execute gcloud command ---
tpu_exec() {
  local CMD=$1
  shift  # remaining args are extra SSH options

  # Base gcloud args
  local base_args=(
    alpha compute tpus tpu-vm ssh "$TPU_NAME"
    --zone="$TPU_ZONE"
    --project="$TPU_PROJECT"
    --verbosity=debug
    --command="$CMD"
  )
  gcloud "${base_args[@]}" -- "$@"
}

install_deps() {
  tpu_exec "source venvs/jupyter/bin/activate && pip install -y $1"
}

setup_venv() {
  echo "Setting up virtual environment now..."
  gcloud compute os-login ssh-keys add --key="$(ssh-add -L | grep publickey)" --project=$TPU_PROJECT
  tpu_exec "mkdir -p venvs && \
  python3 -m venv venvs/jupyter && \
  source venvs/jupyter/bin/activate && \
  pip install --upgrade pip && pip install jupyter && pip install jax[tpu]"
}

# Start the jupyter server. Quote escape is tricky
prepare_jupyter(){
echo "Preparing Jupyter Notebook server now..."
tpu_exec "bash -c 'cat > ~/run_jupyter.sh <<EOF
#!/bin/bash
pgrep -f jupyter-notebook | xargs -r kill
source venvs/jupyter/bin/activate
jupyter notebook --no-browser --ip=0.0.0.0 --port=8889 --NotebookApp.token=''
EOF
chmod +x ~/run_jupyter.sh'"
}

run_jupyter() {
echo "Running Jupyter server now..."
tmux kill-session -t jupyter-session 2>/dev/null || true
tmux new -d -s jupyter-session "$(declare -f tpu_exec); tpu_exec 'bash ~/run_jupyter.sh' '-L 8889:localhost:8889' > ~/jupyter.log 2>&1"
}

create_ssd_and_move_docker_dir() {
  DISK_NAME=$USER-docker-disk-${RANDOM:0:1}
  gcloud compute disks create $DISK_NAME \
    --size=100GB \
    --type=pd-ssd \
    --zone="$TPU_ZONE" \
    --project="$TPU_PROJECT"

  gcloud alpha compute tpus tpu-vm attach-disk "$TPU_NAME" \
    --zone="$TPU_ZONE" \
    --project="$TPU_PROJECT" \
    --disk=$DISK_NAME \
    --mode=read-write

  # Format the disk, depends where your disk is mounted, change /dev/sdb if needed
  MOUNT_POINT=/dev/sdb
  MOUNT_FOLDER=/mnt/docker
  tpu_exec '
      sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0 $MOUNT_POINT &&
      sudo mkdir -p $MOUNT_FOLDER &&
      sudo mount -o discard,defaults $MOUNT_POINT $MOUNT_FOLDER &&
      echo "$MOUNT_POINT $MOUNT_FOLDER ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
    '

  # Move Docker data to the new SSD
  tpu_exec '
      sudo systemctl stop docker &&
      sudo mv /var/lib/docker $MOUNT_FOLDER &&
      sudo ln -s $MOUNT_FOLDER /var/lib/docker &&
      sudo systemctl start docker
    '
}

# Google colab is tricky to run on TPU VM, but it is possible.
run_colab() {
  tmux new -s gcloud-session " \
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --privileged \
    --net=host \
    -v /dev:/dev \
    -v /run:/run \
    -v /var/lib/cloud:/var/lib/cloud \
    -e TPU_NAME="local" \
    --zone=$TPU_ZONE \
    --project=$TPU_PROJECT \
    --command=\"sudo docker pull us-docker.pkg.dev/colab-images/public/runtime; \
    sudo docker stop colab || true && sudo docker rm colab || true && \
    sudo docker run --name colab -p 127.0.0.1:9100:8080 us-docker.pkg.dev/colab-images/public/runtime; \
    sleep 5 && sudo docker logs colab \" \
    -- -o ProxyCommand='corp-ssh-helper -relay=mtv5.r.ext.google.com %h %p ' -L 9100:localhost:9100 && bash \
  "
}

# --- Run the functions ---
# create_ssd_and_move_docker_dir
setup_venv
prepare_jupyter
run_jupyter
