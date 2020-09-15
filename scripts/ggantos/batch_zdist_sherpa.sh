#!/bin/bash -l
#SBATCH -J hol_sherp
#SBATCH --account=NAML0001
#SBATCH -t 1:00:00
#SBATCH --mem=512G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o zdist_sherpa.o
#SBATCH -e zdist_sherpa.o
module load gnu/8.3.0 openmpi/3.1.4 cuda/10.1

source /glade/u/home/ggantos/.bashrc
conda deactivate
conda activate sherpa
export PATH=“/glade/u/home/ggantos/miniconda3/envs/holodec/bin:$PATH”
    
python train_zdist_sherpa.py ../../config/zdist_sherpa.yml
