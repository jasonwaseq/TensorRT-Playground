#!/usr/bin/env python3
import argparse
import json
import time
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--saveEngine', type=str, help='Path to save engine')
    parser.add_argument('--loadEngine', type=str, help='Path to load engine')
    parser.add_argument('--exportTimes', type=str, help='Path to export timing JSON')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16')
    parser.add_argument('--int8', action='store_true', help='Enable INT8')
    parser.add_argument('--builderOptimizationLevel', type=int, help='Optimization level')
    parser.add_argument('--profilingVerbosity', type=str, help='Profiling verbosity')
    parser.add_argument('--timingCacheFile', type=str, help='Timing cache file')
    parser.add_argument('--skipInference', action='store_true', help='Skip inference')
    parser.add_argument('--warmUp', type=int, help='Warm up duration')
    parser.add_argument('--duration', type=int, help='Benchmark duration')
    parser.add_argument('--streams', type=int, help='Number of streams')
    parser.add_argument('--useCudaGraph', action='store_true', help='Use CUDA Graph')

    # Parse known args, ignore others
    args, unknown = parser.parse_known_args()

    print("[Mock trtexec] Running with args:", args)

    # Simulation - Building Engine
    if args.saveEngine:
        print(f"[Mock trtexec] Building engine from {args.onnx}...")
        time.sleep(1) # Simulate build time
        with open(args.saveEngine, 'wb') as f:
            f.write(b'mock_engine_content')
        print(f"[Mock trtexec] Engine saved to {args.saveEngine}")

    # Simulation - Benchmarking
    if args.exportTimes:
        print(f"[Mock trtexec] Benchmarking engine {args.loadEngine}...")
        time.sleep(1) # Simulate benchmark time
        
        # Generator dummy traces
        traces = []
        n_samples = 100
        for i in range(n_samples):
            start = i * 10
            duration = random.uniform(2.0, 3.0) # 2-3 ms latency
            traces.append({
                "startComputeMs": start,
                "endComputeMs": start + duration
            })
        
        with open(args.exportTimes, 'w') as f:
            json.dump(traces, f, indent=4)
        print(f"[Mock trtexec] Trace saved to {args.exportTimes}")

if __name__ == "__main__":
    main()
