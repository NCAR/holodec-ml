"""
Run proprocessing of histogram data on cheyenne
"""

import numpy as np

paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_1particle_gamma_training.nc',
                'input_func':{'abs':np.abs},
                'input_scale':{'abs':255},
                'FourierTransform':True,
                'hist_edges':np.logspace(-1,2.8,100),
                'max_hist_count':5000
                }

exec_file = 'preprocess_hist_data.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))