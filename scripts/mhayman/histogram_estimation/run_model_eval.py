"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/',
            'model_path':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/histogram_training_data_5000count20200819T091551_HypOpt_NN_20200831T093951/'}

settings = {
            'data_file': 'histogram_training_data_5000count20200819T091551.nc',
            'test_file':'histogram_training_data_5000count20200814T075733.nc',
            'model_file':'histogram_training_data_5000count20200819T091551_HypOpt_NN_20200831T093951.h5',
            'scale_labels':False, # Scale the labels - depending on loss function, may not be desirable
            'loss_function':'cum_poisson',     # loss function
            'log_input':True,     # include log of the input data as input
            'early_stopping':True,  # implement early stopping
            'h_chunk':128,      # xarray chunk size when loading
            'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267],  # example outputs
            }

exec_file = 'evaluate_psd_convnet.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))