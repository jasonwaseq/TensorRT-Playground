import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define files to compare
files = {
    "baseline": "results/perf_base.json",
    "cuda-graph": "results/perf_cudagraph.json",
    "stream-2": "results/perf_stream2.json",
    "stream-4": "results/perf_stream4.json",
}

# Collect data
plt.figure(figsize=(10, 6))
found_data = False

for label, path in files.items():
    if not os.path.exists(path):
        print(f"Warning: File {path} not found. Skipping {label}.")
        continue

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {path}. Skipping {label}.")
        continue

    # Extract latencies
    lat = [
        e.get("endComputeMs", 0) - e.get("startComputeMs", 0)
        for e in data
        if "startComputeMs" in e and "endComputeMs" in e
    ]

    if not lat:
        print(f"Warning: No valid latency data found in {path}. Skipping {label}.")
        continue

    found_data = True
    plt.hist(
        lat,
        bins=50,
        histtype="step",
        linewidth=2,
        label=label
    )
    print(f"Added {label} (n={len(lat)})")

if not found_data:
    print("Error: No data found to plot.")
    sys.exit(1)

plt.xlabel("GPU Compute Latency (ms)")
plt.ylabel("Count")
plt.title("TensorRT GPU Latency Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = "results/latency_compare.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
