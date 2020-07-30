"""
Run proprocessing of histogram data on cheyenne
"""

import numpy as np

paths = {   'data':'/h/eol/bansemer/holodec/holodec-ml/datasets/',
            'save':'/scr/sci/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_1particle_gamma_training.nc',
                'input_func':{'abs':np.abs},
                'input_scale':{'abs':255},
                'FourierTransform':True,
                'hist_edges':np.logspace(-1,2.8,100)
                }

exec_file = 'preprocess_hist_data.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))