#!/bin/bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=engines/resnet50_fp16.engine \
  --warmUp=500 \
  --duration=30 \
  --exportTimes=results/perf_base.json

