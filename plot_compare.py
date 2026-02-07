import json
import numpy as np
import matplotlib.pyplot as plt

files = {
    "baseline": "results/perf_base.json",
    "cuda-graph": "results/perf_cudagraph.json",
    "stream-4": "results/perf_stream4.json",
}

plt.figure(figsize=(8, 5))

for label, path in files.items():
    with open(path) as f:
        data = json.load(f)

    lat = [
        e["endComputeMs"] - e["startComputeMs"]
        for e in data
    ]

    plt.hist(
        lat,
        bins=50,
        histtype="step",
        linewidth=2,
        label=label
    )

plt.xlabel("GPU Compute Latency (ms)")
plt.ylabel("Count")
plt.title("TensorRT GPU Latency Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/latency_compare.png", dpi=150)
print("Saved results/latency_compare.png")
