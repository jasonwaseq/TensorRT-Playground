import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def load_gpu_latency(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        sys.exit(1)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {path}.")
        sys.exit(1)

    # GPU compute latency per inference (ms)
    latencies = [
        entry.get("endComputeMs", 0) - entry.get("startComputeMs", 0)
        for entry in data
        if "startComputeMs" in entry and "endComputeMs" in entry
    ]

    return np.array(latencies)

if len(sys.argv) < 2:
    print("Usage: python3 plot_latency.py <perf_data.json> [label]")
    sys.exit(1)

json_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(json_path)

lat = load_gpu_latency(json_path)

if len(lat) == 0:
    print(f"Error: No valid latency data found in {json_path}")
    sys.exit(1)

print(f"--- {label} ---")
print(f"  samples: {len(lat)}")
print(f"  mean:    {lat.mean():.4f} ms")
print(f"  p50:     {np.percentile(lat, 50):.4f} ms")
print(f"  p90:     {np.percentile(lat, 90):.4f} ms")
print(f"  p99:     {np.percentile(lat, 99):.4f} ms")

plt.figure(figsize=(8, 5))
plt.hist(lat, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("GPU Compute Latency (ms)")
plt.ylabel("Count")
plt.title(f"TensorRT GPU Latency Histogram – {label}")
plt.grid(True, alpha=0.3)

out_dir = os.path.dirname(json_path)
out_name = os.path.splitext(os.path.basename(json_path))[0] + "_hist.png"
out_path = os.path.join(out_dir, out_name)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
