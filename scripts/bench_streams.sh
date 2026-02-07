#!/bin/bash

for s in 2 4; do
  /usr/src/tensorrt/bin/trtexec \
    --loadEngine=engines/resnet50_fp16.engine \
    --streams=$s \
    --warmUp=500 \
    --duration=30 \
    --exportTimes=results/perf_stream${s}.json
done

