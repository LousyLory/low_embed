#!/usr/bin/env bash

set -exu

sweep_id=$1
threads=$2

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

echo $OMP_NUM_THREADS
echo $OMP_NUM_THREADS

wandb agent $sweep_id
