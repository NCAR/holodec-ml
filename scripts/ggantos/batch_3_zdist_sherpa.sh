#!/bin/bash -l
#SBATCH -J ho_3sherpa
#SBATCH --account=NAML0001
#SBATCH -t 2:00:00
#SBATCH --mem=256G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o conv2d_3_zdist_sherpa.o
#SBATCH -e conv2d_3_zdist_sherpa.o
module load gnu/8.3.0 openmpi/3.1.4 cuda/10.1

source /glade/u/home/ggantos/.bashrc
conda deactivate
conda activate sherpa
export PATH=“/glade/u/home/ggantos/miniconda3/envs/holodec/bin:$PATH”
    
python train_conv2d_zdist_sherpa.py ../../config/3particle_zdist_sherpa.yml
