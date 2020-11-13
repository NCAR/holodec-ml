#!/bin/bash -l
#SBATCH -J hol_attn
#SBATCH --account=NAML0001
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --partition=dav
#SBATCH --mem=256G
#SBATCH -n 1
#SBATCH --gres=gpu:v100:1
#SBATCH -o attn.o
#SBATCH -e attn.o
module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20200417
export PATH="/glade/work/ggantos/ncar_20200417/bin:$PATH"
    
python -u train_attn.py ../../config/attn.yml
