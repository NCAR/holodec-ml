#!/bin/bash -l
#SBATCH -J hol_unet
#SBATCH --account=NAML0001
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --partition=dav
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH -o unet_%j.o
#SBATCH -e unet_%j.o
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ggantos@ucar.edu
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
source ~/.bashrc
conda activate micro
    
python -u train_unet.py ../../config/unet.yml
