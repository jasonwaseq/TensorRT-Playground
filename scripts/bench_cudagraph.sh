#!/bin/bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=engines/resnet50_fp16.engine \
  --useCudaGraph \
  --warmUp=500 \
  --duration=30 \
  --exportTimes=results/perf_cudagraph.json

