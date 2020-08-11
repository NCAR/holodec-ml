#!/bin/bash -l
#SBATCH -J zmass_callbacks
#SBATCH --account=NAML0001
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH -o conv2d_zmass_callbacks.o
#SBATCH -e conv2d_zmass_callbacks.o
#SBATCH --mail-user=schreck@ucar.edu

module load gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1

ncar_pylib /glade/work/$USER/py37

export PATH="/glade/work/$USER/py37/bin:$PATH"
    
python train_conv2d_zmass.py ../../config/3particle_zmass.yml
