#!/bin/bash
set -e

# Define paths
TRTEXEC_BIN="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ENGINE_FILE="engines/resnet50_fp16.engine"
OUTPUT_FILE="results/perf_base.json"

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

# Run benchmark
echo "Running baseline benchmark..."
$TRTEXEC_BIN \
  --loadEngine="$ENGINE_FILE" \
  --warmUp=500 \
  --duration=30 \
  --exportTimes="$OUTPUT_FILE"

echo "Benchmark complete. Results saved to $OUTPUT_FILE"
