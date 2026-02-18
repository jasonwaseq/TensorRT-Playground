#!/bin/bash
set -e

# Define paths
TRTEXEC_BIN="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX_MODEL="models/resnet50.onnx"
ENGINE_FILE="engines/resnet50_fp16.engine"
TIMING_CACHE="engines/resnet50.cache"

# Check if trtexec exists
if [ ! -x "$TRTEXEC_BIN" ]; then
  echo "Error: trtexec not found at $TRTEXEC_BIN"
  exit 1
fi

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
  echo "Error: ONNX model no found at $ONNX_MODEL"
  echo "Run 'python export_onnx.py' first."
  exit 1
fi

# Create engines directory if it doesn't exist
mkdir -p engines

# Run trtexec
echo "Building TensorRT engine..."
$TRTEXEC_BIN \
  --onnx="$ONNX_MODEL" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --profilingVerbosity=detailed \
  --timingCacheFile="$TIMING_CACHE" \
  --saveEngine="$ENGINE_FILE" \
  --skipInference

echo "Engine build complete: $ENGINE_FILE"
