import os
import sys
import urllib.request
import urllib.parse
import json
import time

TOKEN = sys.argv[1]
REPO_ID = "Qwen/Qwen3-1.7B"
LOCAL_DIR = "/mnt/disks/linchai_data/huggingface/models/Qwen/Qwen3-1.7B"
os.makedirs(LOCAL_DIR, exist_ok=True)

# List files using HF REST API
req = urllib.request.Request(
    f"https://huggingface.co/api/models/{REPO_ID}",
    headers={"Authorization": f"Bearer {TOKEN}"}
)
try:
    with urllib.request.urlopen(req) as response:
        repo_info = json.loads(response.read().decode())
except Exception as e:
    print(f"Failed to fetch repo info: {e}")
    sys.exit(1)

files = [f["rfilename"] for f in repo_info["siblings"]]
print(f"Files to download: {files}")

class RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = urllib.request.Request(newurl)
        for h, v in req.headers.items():
            if h.lower() != 'authorization':
                new_req.add_header(h, v)
        return new_req

opener = urllib.request.build_opener(RedirectHandler)
urllib.request.install_opener(opener)

MAX_RETRIES = 5

for filename in files:
    dest_path = os.path.join(LOCAL_DIR, filename)
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
    
    # Get remote file size using HEAD request
    remote_size = 0
    head_req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(head_req) as resp:
            remote_size = int(resp.info().get("Content-Length", 0))
    except Exception as e:
        print(f"Warning: Failed to get HEAD info for {filename}: {e}")
        
    # Check if local file already exists and matches remote size
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        if remote_size > 0 and local_size == remote_size:
            print(f"File {filename} already exists and matches expected size ({local_size / 1024 / 1024:.2f} MB). Skipping.")
            continue
        elif remote_size > 0:
            print(f"File {filename} exists but size mismatch (local: {local_size}, remote: {remote_size}). Re-downloading.")
            try:
                os.remove(dest_path)
            except OSError as e:
                print(f"Failed to remove corrupted file {dest_path}: {e}")
                sys.exit(1)
        else:
            # If we couldn't get remote size, skip only if size > 0 to be safe
            if local_size > 0:
                print(f"File {filename} exists (size: {local_size / 1024 / 1024:.2f} MB). Skipping HEAD check fallback.")
                continue

    # Download loop with retries
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Downloading {filename} (Attempt {attempt}/{MAX_RETRIES})...")
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        downloaded = 0
        try:
            with urllib.request.urlopen(req) as response:
                meta = response.info()
                file_size = int(meta.get("Content-Length", 0))
                if remote_size == 0:
                    remote_size = file_size
                print(f"Expected Size: {remote_size / 1024 / 1024:.2f} MB")
                
                with open(dest_path, "wb") as f:
                    block_size = 8192 * 16
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        if remote_size:
                            percent = downloaded * 100.0 / remote_size
                            sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded / 1024 / 1024:.2f} MB)")
                            sys.stdout.flush()
                print()
                
            # Verify final downloaded size
            if remote_size > 0 and downloaded != remote_size:
                raise ValueError(f"Size mismatch: downloaded {downloaded} of expected {remote_size} bytes")
                
            print(f"Successfully downloaded {filename}")
            break # Break retry loop on success
            
        except Exception as e:
            print(f"\nError downloading {filename} on attempt {attempt}: {e}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print(f"Failed to download {filename} after {MAX_RETRIES} attempts.")
                sys.exit(1)

print("All downloads finished successfully!")
