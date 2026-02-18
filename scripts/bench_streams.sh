#!/bin/bash
set -e

# Define paths
TRTEXEC_BIN="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ENGINE_FILE="engines/resnet50_fp16.engine"

# Check if trtexec exists
if [ ! -x "$TRTEXEC_BIN" ]; then
    echo "Error: trtexec not found at $TRTEXEC_BIN"
    exit 1
fi

# Check if engine exists
if [ ! -f "$ENGINE_FILE" ]; then
    echo "Error: Engine file not found at $ENGINE_FILE"
    echo "Run scripts/build_fp16_engine.sh to build it first."
    exit 1
fi

# Create results directory
mkdir -p results

# Run benchmarks for streams 2 and 4
for s in 2 4; do
    OUTPUT_FILE="results/perf_stream${s}.json"
    echo "Running benchmark with $s streams..."
    $TRTEXEC_BIN \
      --loadEngine="$ENGINE_FILE" \
      --streams=$s \
      --warmUp=500 \
      --duration=30 \
      --exportTimes="$OUTPUT_FILE"
    
    echo "Benchmark for $s streams complete. Results saved to $OUTPUT_FILE"
done

echo "All stream benchmarks complete."
