"""
Train variational autoecoder
created 10/27/2020
Matthew Hayman
mhayman@ucar.edu


settings={
        'datafile',:'synthetic_holograms_v02.nc',
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
        'num_epochs':20,
        'batch_size':16,
        'input_variable':'image,
        'split_fraction':0.8,
        'valid_fraction':0.2,
        'image_rescale':255.0,
        'beta':5e-4,   # KL divergence penalty scalar
        'h_chunk':128,      # xarray chunk size when loading
        'holo_examples':[1, 4, 9, 50, 53, 77, 91, 101, 105, 267]  # example outputs
        

        }


"""
# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import xarray as xr

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, SeparableConv2D, UpSampling2D, concatenate, Conv2DTranspose, Lambda, Reshape, Layer
# from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow.keras.metrics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import ml_defs as mldef

run_date = datetime.datetime.now()
run_date_str = run_date.strftime('%Y%m%dT%H%M%S')

num_epochs = settings['num_epochs']

input_variable = settings['input_variable']

split_fraction = settings['split_fraction']
valid_fraction = settings['valid_fraction']

image_rescale = settings['image_rescale']

with xr.open_dataset(paths['load_data']+settings['datafile'],chunks={'hologram_number':settings['h_chunk']}) as ds:
    print(ds.data_vars)
    file_base = 'vae_'+settings['datafile'].replace('.nc','')+'_'+run_date_str
    save_path = paths['save_data']+file_base+'/'
    model_save_path = paths['save_data']
    ml.ensure_path(save_path)


    print('Training dataset attributes')
    for att in ds.attrs:
        print('  '+att+': '+str(ds.attrs[att]))
    
    print('   max particle size: %d'%ds['d'].values.max())
    print('   min particle size: %d'%ds['d'].values.min())
    print()
    
    # Setup training data
    split_index = np.int(split_fraction*ds.sizes['hologram_number'])  # number of training+validation points
    valid_index = np.int(valid_fraction*ds.sizes['hologram_number'])  # number of validation points
    
    scaled_in_data = ds[input_variable]/image_rescale
    print('\ninput dimensions:')
    print(scaled_in_data.dims)
    print(scaled_in_data.shape)
    print()
    print('split index: %d'%split_index)
    print('valid index: %d'%valid_index)
    
    if not 'channel' in scaled_in_data.dims:
        scaled_in_data = scaled_in_data.expand_dims("channel", 3)
    scaled_in_train = scaled_in_data.isel(hologram_number=slice(valid_index,split_index))
    scaled_in_valid = scaled_in_data.isel(hologram_number=slice(None,valid_index))
    scaled_in_test = scaled_in_data.isel(hologram_number=slice(split_index,None))

n_filters = settings['n_filters']
nConv = settings['nConv']
nPool = settings['nPool']
kernel_initializer = settings['kernel_initializer']
latent_dim = settings['latent_dim']
n_dense_layers = settings['n_dense_layers']

# create the model
input_node = Input(shape=scaled_in_data.shape[1:])  
next_input_node = input_node
for _ in range(settings['n_layers']):
    # define the down sampling layer
    conv_1d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(next_input_node)
    act_1d = Activation("relu")(conv_1d)
    conv_2d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(act_1d)
    act_2d = Activation("relu")(conv_2d)
    next_input_node = MaxPool2D(pool_size=(nPool, nPool))(act_2d)
    n_filters = n_filters*2

n_filters = n_filters//2

input_shape = K.int_shape(next_input_node)
zinput = Flatten()(next_input_node)
z_mean = Dense(latent_dim,activation='relu')(zinput)
z_log_var = Dense(latent_dim,activation='relu')(zinput)
for _ in range(np.maximum(n_dense_layers-2,0)):
    # represent the mean and variance branches
    # with separate dense networks
    z_mean = Dense(latent_dim,activation='relu')(z_mean)
    z_log_var = Dense(latent_dim,activation='relu')(z_log_var)
z_mean = Dense(latent_dim,activation='linear')(z_mean)
z_log_var = Dense(latent_dim,activation='linear')(z_log_var)

z = Lambda(mldef.vae_sample)([z_mean,z_log_var])

x1 = Dense(np.prod(input_shape[1:]),activation='relu')(z)
return_node = Reshape(input_shape[1:])(x1)


for _ in range(np.maximum(settings['n_layers'],0)):
    # define the up sampling and feed foward layer
    upsamp_1u = Conv2DTranspose(n_filters, (nConv,nConv), strides=(nPool,nPool),padding="same")(return_node)
    conv_1u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(upsamp_1u)
    act_1u = Activation("relu")(conv_1u)
    conv_2u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(act_1u)
    return_node = Activation("relu")(conv_2u)
    n_filters = n_filters // 2

# add the output layer
z_decoded = Conv2D(2,(1,1),padding="same",activation=settings['out_act'])(return_node)

class CustomVariationalLayer(Layer):
    def vae_loss(self,x,z_decoded,z_mean,z_log_var):
        # calculate image intensity assuming
        # reference field is sqrt(0.5)+0*1j and the scattered
        # field is z_decoded[...,0]+z_decoded[...,1]*1j
        x_hat = K.square(np.sqrt(0.5)+z_decoded[...,0])+K.square(z_decoded[...,1])
        # calculate MSE between decoded image and actual image
        x_mse_loss = K.mean(K.square(x[...,0]-x_hat),axis=[1,2])
        beta = settings['beta']
        kl_loss = -beta*K.mean(1+z_log_var-K.square(z_mean) - K.exp(z_log_var),axis=-1)
        return K.mean(x_mse_loss+ kl_loss)

    def call(self,inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x,z_decoded,z_mean,z_log_var)
        self.add_loss(loss,inputs=inputs)
        return z_decoded

y = CustomVariationalLayer()([input_node,z_decoded,z_mean,z_log_var])


# build and compile the model
mod = Model(input_node, y)
mod.compile(optimizer="adam", loss=None, metrics=['acc'])
mod.summary()

ml.ensure_path(save_path)
plot_model(mod,show_shapes=True,to_file=save_path+file_base+"model_plot.png")

history = mod.fit(scaled_in_train.values,y=None,
                  batch_size=settings['batch_size'], epochs=settings['num_epochs'], verbose=1,
                  validation_data=(scaled_in_valid.values,None))

epochs = np.arange(len(history.history['loss']))+1
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(epochs,history.history['loss'],'bo-',alpha=0.5,label='Training')
ax.plot(epochs,history.history['val_loss'],'rs-',alpha=0.5,label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(b=True)
plt.legend()
plt.savefig(save_path+file_base+"_LossHistory_"+file_base+".png", dpi=200, bbox_inches="tight")

# fig, bx = plt.subplots(1, 1, figsize=(8, 4))
# bx.plot(epochs,history.history['acc'],'bo-',alpha=0.5,label='Training')
# bx.plot(epochs,history.history['val_acc'],'rs-',alpha=0.5,label='Validation')
# bx.set_xlabel('Epoch')
# bx.set_ylabel('Accuracy')
# bx.grid(b=True)
# plt.legend()
# plt.savefig(save_path+file_base+"AccuracyHistory_"+file_base+".png", dpi=200, bbox_inches="tight")


# save the model
save_model(mod, model_save_path+file_base+'.h5', save_format="h5")

# output test examples
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_in_test.values, batch_size=64)
cnn_stop = datetime.datetime.now()
print(f"{scaled_in_test.values.shape[0]} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_in_test.values.shape[0]} seconds per hologram")

preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','xsize','ysize','channel'),
                            coords=scaled_in_test.coords)

for im in settings['holo_examples']:
    fig_obj, ax_obj_lst = plt.subplots(1, 3, figsize=(3*6, 4))
    ax_obj = ax_obj_lst[0]
    im_obj = ax_obj.matshow(scaled_in_test.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('True image')

    ax_obj = ax_obj_lst[1]
    im_obj = ax_obj.matshow(preds_out_da.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('Reconstructed Image')

    ax_obj = ax_obj_lst[2]
    im_obj = ax_obj.imshow(scaled_in_test.isel(hologram_number=im,channel=0)-preds_out_da.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('Difference')
    plt.savefig(save_path+file_base+"_TestExampleCase_%d_"%im+file_base+".png", dpi=200, bbox_inches="tight")
    plt.close('all')
    
# output training examples
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_in_train.values, batch_size=64)
cnn_stop = datetime.datetime.now()
print(f"{scaled_in_train.values.shape[0]} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_in_train.values.shape[0]} seconds per hologram")

preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','xsize','ysize','channel'),
                            coords=scaled_in_train.coords)

for im in settings['holo_examples']:
    fig_obj, ax_obj_lst = plt.subplots(1, 3, figsize=(3*6, 4))
    ax_obj = ax_obj_lst[0]
    im_obj = ax_obj.matshow(scaled_in_train.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('True image')

    ax_obj = ax_obj_lst[1]
    im_obj = ax_obj.matshow(preds_out_da.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('Reconstructed Image')

    ax_obj = ax_obj_lst[2]
    im_obj = ax_obj.imshow(scaled_in_train.isel(hologram_number=im,channel=0)-preds_out_da.isel(hologram_number=im,channel=0))
    plt.colorbar(im_obj, ax=ax_obj)
    ax_obj.set_title('Difference')
    plt.savefig(save_path+file_base+"_TrainExampleCase_%d_"%im+file_base+".png", dpi=200, bbox_inches="tight")
    plt.close('all')