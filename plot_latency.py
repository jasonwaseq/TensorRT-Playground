import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_gpu_latency(path):
    with open(path, "r") as f:
        data = json.load(f)

    # GPU compute latency per inference (ms)
    latencies = [
        entry["endComputeMs"] - entry["startComputeMs"]
        for entry in data
        if "startComputeMs" in entry and "endComputeMs" in entry
    ]

    return np.array(latencies)

if len(sys.argv) < 2:
    print("Usage: python3 plot_latency.py perf_base.json [label]")
    sys.exit(1)

json_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else json_path

lat = load_gpu_latency(json_path)

print(f"{label}")
print(f"  samples: {len(lat)}")
print(f"  mean: {lat.mean():.4f} ms")
print(f"  p50: {np.percentile(lat, 50):.4f} ms")
print(f"  p90: {np.percentile(lat, 90):.4f} ms")
print(f"  p99: {np.percentile(lat, 99):.4f} ms")

plt.figure(figsize=(8, 5))
plt.hist(lat, bins=50)
plt.xlabel("GPU Compute Latency (ms)")
plt.ylabel("Count")
plt.title(f"TensorRT GPU Latency Histogram – {label}")
plt.grid(True)

out = json_path.replace(".json", "_hist.png")
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved plot to {out}")
