"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {
            'data_file':'histogram_training_data_5000count20200731T133118.nc',     # training data file
            'num_epochs':50,    # the number of training epochs
            'conv_chan':[16,32],# list length defines number of operations
            'conv_size':[5,5],  # convolution kernel size
            'max_pool':[4,4],   # maxpool decimation
            'nn_size':[64,32,],  # excludes the output layer (set by the input data)
            'batch_size':64,    # training batch size
            'output_activation':'linear', # output activation function,
            'valid_fraction':0.2,   # fraction of data reserved for validation
            'test_fraction':0.3,    # fraction of data reserved for training
            'loss_function':'KLDivergence',     # loss function
            'h_chunk':64,      # xarray chunk size when loading
            }

exec_file = 'train_psd_convnet.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))