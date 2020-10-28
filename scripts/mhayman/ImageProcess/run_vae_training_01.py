"""
Train ConvNet to estimate PSD on Casper
"""

paths = {   'load_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save_data':'/glade/scratch/mhayman/holodec/holodec-ml-data/vae/'}
settings={
        'datafile':'synthetic_holograms_v02.nc',
        'n_layers':1, # must be greater than 1 because the latent layer counts
        'n_filters':2, # number of input convolutional channels
        'nConv':4, # convolution kernel size
        'nPool':4, # max pool size
        'activation':'relu', # convolution activation
        'kernel_initializer':"he_normal",
        'latent_dim':32,
        'n_dense_layers':2, # number of dense layers in bottom layer
        'loss_fun':'mse',   # training loss function
        'out_act':'linear',  # output activation
        'num_epochs':100,
        'batch_size':16,
        'input_variable':'image',
        'split_fraction':0.8,
        'valid_fraction':0.2,
        'image_rescale':255.0,
        'beta':5e-4,   # KL divergence penalty scalar
        'h_chunk':128,      # xarray chunk size when loading
        'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267]  # example outputs
        }

exec_file = 'train_vae_01.py'
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))