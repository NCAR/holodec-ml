"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {
            'data_file':'histogram_training_data_10000count20200812T201643.nc',     # training data file
            'num_epochs':50,    # the number of training epochs
            'conv_chan':[32,32],# list length defines number of operations
            'conv_size':[5,5],  # convolution kernel size
            'max_pool':[2,2],   # maxpool decimation
            'nn_size':[128,64,],  # excludes the output layer (set by the input data)
            'batch_size':256,    # training batch size
            'output_activation':'relu', # output activation function,
            'valid_fraction':0.2,   # fraction of data reserved for validation
            'test_fraction':0.1,    # fraction of data reserved for training
            'loss_function':'kstest',     # loss function
            'h_chunk':128,      # xarray chunk size when loading
            'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267]  # example outputs
            }

exec_file = 'train_psd_convnet.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))