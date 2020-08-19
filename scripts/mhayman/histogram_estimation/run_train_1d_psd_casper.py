"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {
            'data_file': 'histogram_training_data_5000count20200819T094342.nc', #'histogram_training_data_5000count20200818T100751.nc',# 'histogram_training_data_5000count20200814T075733.nc',     # training data file
            'validation_file':'histogram_validation_data_5000count20200819T094342.nc',
            'test_file':'histogram_test_data_5000count20200819T094342.nc',
            'num_epochs':2000,    # the number of training epochs
            'scale_labels':False, # Scale the labels - depending on loss function, may not be desirable
            'conv_chan':[],# list length defines number of operations
            'conv_size':[],  # convolution kernel size
            'max_pool':[],   # maxpool decimation
            'nn_size':[256,128,],  # excludes the output layer (set by the input data)
            'batch_size':256,    # training batch size
            'output_activation':'exponential', # output activation function,
            'valid_fraction':0.1,   # fraction of data reserved for validation
            'test_fraction':0.1,    # fraction of data reserved for training
            'loss_function':'cum_poisson',     # loss function
            'h_chunk':128,      # xarray chunk size when loading
            'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267]  # example outputs
            }

exec_file = 'train_psd_convnet.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))