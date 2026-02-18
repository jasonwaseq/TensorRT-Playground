#!/bin/bash
set -e

# Define colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function for logging
log() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check dependencies
check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "$1 could not be found. Please install it."
    fi
}

check_command python3
# trtexec might not be in PATH but in /usr/src/tensorrt/bin
if [ ! -x "/usr/src/tensorrt/bin/trtexec" ] && ! command -v trtexec &> /dev/null && [ -z "$TRTEXEC" ]; then
    error "trtexec could not be found."
fi

# Ensure directories exist
mkdir -p models engines results

# Step 1: Export ONNX
log "Step 1: Exporting ONNX model..."
python3 export_onnx.py

# Step 2: Build TensorRT Engine
log "Step 2: Building TensorRT engine..."
./scripts/build_fp16_engine.sh

# Step 3: Run Benchmarks
log "Step 3: Running benchmarks..."
./scripts/bench_base.sh
./scripts/bench_cudagraph.sh
./scripts/bench_streams.sh

# Step 4: Generate Plots
log "Step 4: Generating plots..."
python3 plot_compare.py

# Generate individual latency histograms
for result_file in results/perf_*.json; do
    if [ -f "$result_file" ]; then
        log "Plotting latency for $(basename "$result_file")..."
        python3 plot_latency.py "$result_file"
    fi
done

log "All steps completed successfully!"
ls -l results/*.png
