#!/bin/bash
source /home/till/.zshrc
conda activate pyhpc-bench-gpu
export CUPY_CUDA_COMPILE_WITH_DEBUG=1
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
python profile_cupy.py