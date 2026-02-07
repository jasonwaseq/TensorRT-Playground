#!/bin/bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/resnet50.onnx \
  --fp16 \
  --builderOptimizationLevel=5 \
  --profilingVerbosity=detailed \
  --timingCacheFile=engines/resnet50.cache \
  --saveEngine=engines/resnet50_fp16.engine \
  --skipInference

