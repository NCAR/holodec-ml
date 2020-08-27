#!/bin/bash -l
#SBATCH -J output
#SBATCH --account=NAML0001
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH -n 8
#SBATCH --gres=gpu:v100:1
#SBATCH -o out
#SBATCH -e out
#SBATCH --mail-user=schreck@ucar.edu

module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1

ncar_pylib /glade/work/$USER/py37

export PATH="/glade/work/$USER/py37/bin:$PATH"
    
python trainer.py config.yml
