"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {
            'data_file': 'histogram_training_data_5000count20200819T091551.nc', #'histogram_training_data_5000count20200818T100751.nc',# 'histogram_training_data_5000count20200814T075733.nc',     # training data file
            'validation_file':'histogram_validation_data_5000count20200819T091551.nc',
            'test_file':'histogram_test_data_5000count20200819T091551.nc',
            'scale_labels':False, # Scale the labels - depending on loss function, may not be desirable
            'output_activation':'exponential', # output activation function,
            'valid_fraction':0.1,   # fraction of data reserved for validation
            'test_fraction':0.1,    # fraction of data reserved for training
            'loss_function':'cum_poisson',     # loss function
            'h_chunk':128,      # xarray chunk size when loading
            'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267],  # example outputs

            # optimization definitions: high, low and initial value
            'learning_rate':[1e-4,1e-2,1e-3], 
            'num_dense_layers':[0,5,1],
            'num_input_nodes':[1,1024,512],
            'num_dense_nodes':[1,1024,512],
            'activation':['relu', 'sigmoid','relu'], # categorical list, last is initial value
            'batch_size':[1,256,64],
            'adam_decay':[1e-6,1e-2,1e-3],
            'epoch_count':[500,5000,1000],

            # gp_minimize arguments
            'n_calls':12,
            'noise':0.01,
            'n_jobs':-1,
            'kappa':5
            }

exec_file = 'optimize_hist_hyperparam.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))