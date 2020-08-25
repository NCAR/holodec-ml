#!/bin/bash -l
#SBATCH -J casp_nb
#SBATCH --account=NAML0001
#SBATCH -t 8:00:00
#SBATCH --mem=128G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o conv2d_1.o
#SBATCH -e conv2d_1.o
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20200417
export PATH="/glade/work/ggantos/ncar_20200417/bin:$PATH"
    
python train_conv2d.py ../../config/1particle.yml