#!/bin/bash -l 

#PBS -N 100holo
#PBS -l walltime=6:59:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -l gpu_type=v100
#PBS -A NAML0001
#PBS -q casper

#PBS -j eo
#PBS -k eod

#module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
source ~/.bashrc
conda activate holo_torch

python -u inference.py
