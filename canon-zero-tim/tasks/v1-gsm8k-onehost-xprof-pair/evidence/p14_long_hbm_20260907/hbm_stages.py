import sys, json, re
root = sys.argv[1]
prev = None
for line in open(f"{root}/train/raw.log", errors="replace"):
    m = re.search(r"\[V2\.FL\.HBM_STAGE\] (\{.*\})", line)
    if not m:
        continue
    try:
        d = json.loads(m.group(1))
    except Exception:
        continue
    dev = d["devices"][0]
    peak = dev["peak_bytes_in_use"] / 2**30
    use = dev["bytes_in_use"] / 2**30
    jump = "" if prev is None or peak - prev < 0.5 else f"  <-- +{peak - prev:.2f}"
    print(f"{str(d.get('stage')):40} g={str(d.get('group')):>3} in_use={use:6.2f} peak={peak:6.2f}{jump}")
    prev = peak
