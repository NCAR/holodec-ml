#!/bin/bash -l
#SBATCH --account=NAML0001
#SBATCH -t 14:00:00
#SBATCH --mem=128G
#SBATCH -n 8
#SBATCH --gres=gpu:v100:1
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
module load ncarenv
ncar_pylib 
python conv2d_evaluation_z.py > holodec.log
