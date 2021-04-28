"""
Train DenseNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/hist-from-vae/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/hist-from-vae/'}

settings = {
            'data_file': 'training_1_25particles_gamma2D_le4.nc', #'histogram_training_data_5000count20200818T100751.nc',# 'histogram_training_data_5000count20200814T075733.nc',     # training data file
            'validation_file':'validation_1_25particles_gamma2D_le4.nc',
            'test_file':'test_1_25particles_gamma2D_le4.nc',
            'scale_inputs':True, # Scale the inputs
            'scale_labels':False, # Scale the labels - depending on loss function, may not be desirable
            'output_activation':'sigmoid', # output activation function,
            'valid_fraction':0.1,   # fraction of data reserved for validation
            'test_fraction':0.1,    # fraction of data reserved for training
            'loss_function':'binary_crossentropy',     # loss function 'cum_poisson', 'kstest'
            'log_input':False,     # include log of the input data as input
            'early_stopping':True,  # implement early stopping
            'h_chunk':32,      # xarray chunk size when loading
            'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267],  # example outputs

            # optimization definitions: high, low and initial value
            'learning_rate':[1e-4,1e-2,1e-3], 
            'num_dense_layers':[0,4,2],
            'num_input_nodes':[1,1028,512],
            'num_dense_nodes':[1,1028,128],
            'activation':['relu', 'sigmoid','relu'], # categorical list, last is initial value
            'batch_size':[16,256,32],
            'adam_decay':[1e-6,1e-2,1e-3],
            'epoch_count':[10,500,100],

            # gp_minimize arguments
            'n_calls':12,
            'noise':0.05,
            'n_jobs':-1,
            'kappa':5
            }

exec_file = 'optimize_2Dhistvae_d_hyperparam.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))