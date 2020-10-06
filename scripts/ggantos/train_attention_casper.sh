#!/bin/bash -l
#SBATCH --job-name=attn
#SBATCH --account=NAML0001
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=dav
#SBATCH --mem=256G
#SBATCH --output=hsdata_wrfrt.%j.out
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=dgagne@ucar.edu
module load cuda/10.1
export PATH="/glade/u/home/dgagne/miniconda3/envs/hfip/bin:$PATH"
cd /glade/u/home/dgagne/holodec-ml/scripts/ggantos
python -u train_attn.py ../../config/attn.yml
