#!/bin/bash -l
#SBATCH -J hol_paral
#SBATCH --account=NAML0001
#SBATCH -t 1:00:00
#SBATCH --mem=512G
#SBATCH -n 4
#SBATCH --gres=gpu:v100:1
#SBATCH -o zdist_sherpa_parallel.o
#SBATCH -e zdist_sherpa_parallel.o
module load gnu/8.3.0 openmpi/3.1.4 cuda/10.1

source /glade/u/home/ggantos/.bashrc
conda deactivate
conda activate sherpa
export PATH=“/glade/u/home/ggantos/miniconda3/envs/holodec/bin:$PATH”
    
python sherpa_parallel_runner.py ../../config/zdist_sherpa.yml
