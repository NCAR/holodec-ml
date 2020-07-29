"""
Run proprocessing of histogram data on cheyenne
"""

paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_1particle_gamma_training.nc',
                'input_func':{'abs':np.abs},
                'input_scale':{'abs':255}
                'FourierTransform':True,
                'hist_edges':np.linspace(0,300,100)
                }

exec_file = 'preprocess_hist_data.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))