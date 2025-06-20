#!/bin/bash

# export TORCH_CUDA_ARCH_LIST='8.0'
export CUDA_VISIBLE_DEVICES="$1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

clear && python train.py