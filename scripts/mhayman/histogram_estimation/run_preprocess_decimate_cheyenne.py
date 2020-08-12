"""
Run proprocessing of histogram data on cheyenne
"""

import numpy as np

paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_50-100particle_gamma_training.nc',
                'input_func':{'abs':np.abs},
                'input_scale':{'abs':255},
                'FourierTransform':True,
                'hist_edges':np.logspace(0,3,40),
                'max_hist_count':10000,
                'n_decimate':4,  # decimation factor for images
                'sigk':4,    # smoothing kernel standard deviation in pixels
                }

exec_file = 'preprocess_hist_data.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))