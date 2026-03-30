import time
import json
import subprocess
from google.cloud import pubsub_v1

# --- CONFIG ---
PROJECT_ID = "cloud-tpu-multipod-dev"
SUBSCRIPTION_ID = "gcs-tpu-subscription"
# The actual heavy-lifting script that uses the TPU
PROCESS_SCRIPT = "math_eval_nb.py"

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

PREFIX = "gs://linchai-bucket-dev/rl/checkpoints/deepscaler_ckpt/vllm_old_logpbs_bs64_kl_no_logpbs/01/actor"
def callback(message):
    try:
        # 1. Parse the GCS event data
        data = json.loads(message.data.decode("utf-8"))
        file_path = data.get('name') # e.g., "10/data.bin"
        if PREFIX not in file_path:
            print(f"⚠️ Ignoring file {file_path} as it doesn't match prefix {PREFIX}")
            message.ack()  # Still ack to remove from queue
            return
        
        folder_name = file_path.split('/')[0]
        print(f"📥 Received message for file: {file_path}, extracted folder: {folder_name}")
        
        if folder_name.isdigit():
            print(f"🔔 Found new folder: {folder_name}. Starting TPU task...")
            
            # 2. Run your TPU code
            # We pass the folder name/path as an argument
            subprocess.run(["python3", PROCESS_SCRIPT, file_path, folder_name], check=True)
            
            print(f"✅ Task for {folder_name} complete.")
        
        # 3. "Acknowledge" the message so it's deleted from the queue
        message.ack()
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        # If it fails, we don't 'ack', so it stays in the queue to try again later
        message.nack()

if __name__ == "__main__":
    print(f"🛰️ TPU Listener active on {SUBSCRIPTION_ID}...")
    
    # Limit to 1 message at a time so we don't overwhelm the TPU
    flow_control = pubsub_v1.types.FlowControl(max_messages=1)
    
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control
    )

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("\n🛑 Listener stopped.")