#!/bin/bash
export TRTEXEC="$(pwd)/mock_trtexec.py"
chmod +x mock_trtexec.py
echo "Using mock trtexec at $TRTEXEC"

./run_all.sh
