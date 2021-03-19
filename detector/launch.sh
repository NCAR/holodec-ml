#!/bin/bash -l

#SBATCH -J output
#SBATCH --account=NAML0001
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH --exclude=casper[09,36]
#SBATCH -n 8
#SBATCH --gres=gpu:v100:1
#SBATCH -o test/out
#SBATCH -e test/err

module load ncarenv/1.3 gnu/8.3.0 python cuda/10.1 cudnn/7.6.5 nccl/2.7.5
ncar_pylib /glade/work/$USER/py37
export PATH="/glade/work/$USER/py37/bin:$PATH"

python train.py model.yml
