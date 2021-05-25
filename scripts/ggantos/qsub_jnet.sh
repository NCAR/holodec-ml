#!/bin/bash -l
# +
#PBS -N jnetgauss
#PBS -A NAML0001
#PBS -l select=1:ncpus=8:ngpus=1:mem=256GB
#PBS -l walltime=7:59:00
#PBS -q casper

#PBS -j eo
#PBS -k eod
#PBS -m a
#PBS -M ggantos@ucar.edu

module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
source ~/.bashrc
conda activate micro

# -

python -u train_jnet_xy.py ../../config/jnet_xy.yml
