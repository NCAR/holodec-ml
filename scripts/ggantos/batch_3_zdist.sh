#!/bin/bash -l
#SBATCH -J ho_zdist
#SBATCH --account=NAML0001
#SBATCH -t 2:00:00
#SBATCH --mem=256G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o conv2d_large_zdist.o
#SBATCH -e conv2d_large_zdist.o
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20200417
export PATH="/glade/work/ggantos/ncar_20200417/bin:$PATH"
    
python train_conv2d_zdist.py ../../config/3particle_zdist.yml
