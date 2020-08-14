"""
Run proprocessing of histogram data on cheyenne
"""

import numpy as np

paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_50-100particle_gamma_training.nc',
                'FourierTransform':True,
                'hist_edges':np.logspace(0,3,41),
                'max_hist_count':5000,
                }

exec_file = 'preprocess_hist_1D_data.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))