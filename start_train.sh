#!/bin/bash

# export TORCH_CUDA_ARCH_LIST='8.0'
# export CUDA_VISIBLE_DEVICES="3"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


clear && torchrun --nproc_per_node 1 train.py
# clear && python train.py