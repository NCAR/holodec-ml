"""
Python code to build (or load) and
execuate training of ConvT UNET

"""

# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')


import sys
import numpy as np
import xarray as xr

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, SeparableConv2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import ml_defs as mldef

# Model Training settings
h_chunk = 256 # size of dask array chunks along hologram_number dimension
num_epochs = 101  # number of training epochs to run
batch_size = 64   # training batch size
split_fraction = 0.7  # fraction of points used for training/validation (not testing)
valid_fraction = 0.1  # fraction of points used for validation


# Training data file
ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v02/'   # linux share
ds_file='UNET_image_256x256_5000count_5particles_5zplanes_v02.nc'

input_variable = 'image_planes'

model_path = '/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/'
model_file = ''  # if empty, creates a new model

### New Model Definitions
nFilters = 32
nPool = 2
nConv = 13
nLayers = 4
loss_fun = mldef.filtered_mae  # definition passed into compiler 
loss_str = "filtered_mae"  # string representation of loss for filename
out_act = "linear" # "sigmoid"


if len(model_file) == 0:
    new_model = True  # Create a new model
    nn_descript = f'UNET_Layers{nLayers}_Conv{nConv}_Pool{nPool}_Filt{nFilters}_'+loss_str+'_'+out_act
    run_num = 0
    model_path = "/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/"+nn_descript+"/"+ds_file.replace(".nc","/")
else:
    new_model = False
    nn_descript = model_file.split('_epochs')[0]
    run_num = np.int(model_file.split('_run').replace('.h5',''))
    
      
# make sure there file structure is in place
# for the model
ml.ensure_path(model_path)

# save path for training results in the git repo
save_path1 = 'results/'+nn_descript+"/"+ds_file.replace(".nc","/")
ml.ensure_path(save_path1)

# save path for model description file
save_path_mod = 'results/'+nn_descript+"/"
ml.ensure_path(save_path_mod)

print('Training UNET on')
print(ds_file)
print('located in')
print(ds_path)

# open the data to train on
ds = xr.open_dataset(ds_path+ds_file,chunks={'hologram_number': h_chunk})


print('Training dataset attributes')
for att in ds.attrs:
    print('  '+att+': '+str(ds.attrs[att]))


# Setup labels
split_index = np.int(split_fraction*ds.sizes['hologram_number'])  # number of training+validation points
valid_index = np.int(valid_fraction*ds.sizes['hologram_number'])  # number of validation points
all_labels = ds['labels'].sel(type=['amplitude','z'])

train_labels = all_labels.isel(hologram_number=slice(valid_index,split_index))
test_labels = all_labels.isel(hologram_number=slice(split_index,None))
val_labels = all_labels.isel(hologram_number=slice(None,valid_index))

scaler = ml.MinMaxScalerX(train_labels,dim=('hologram_number','xsize','ysize'))
scaled_train_labels = scaler.fit_transform(train_labels)
scaled_val_labels = scaler.fit_transform(val_labels)
scaled_test_labels = scaler.fit_transform(test_labels)
scaled_all_labels = scaler.fit_transform(all_labels)

# setup the input to be used
in_data = ds[input_variable]

if not 'channel' in in_data.dims:
    in_data = in_data.expand_dims("channel", 3)

scaled_in_data = in_data

### Build the UNET ###

if new_model:
    # define the input based on input data dimensions
    cnn_input = Input(shape=scaled_in_data.shape[1:])  

    # create the unet
    unet_out = mldef.add_unet_layers(cnn_input,nLayers,nFilters,nConv=nConv,nPool=nPool,activation="relu")

    # add the output layer
    out = Conv2D(scaled_train_labels.sizes['type'],(1,1),padding="same",activation=out_act)(unet_out)

    # build and compile the model
    mod = Model(cnn_input, out)
    mod.compile(optimizer="adam", loss=loss_fun, metrics=['acc'])
    mod.summary()

    ### End UNET Definition ###

    # save a visualization of the net
    plot_model(mod,show_shapes=True,to_file=save_path_mod+nn_descript+".png")

else:
    mod = load_model(model_path+model_file)
    mod.summary()


print()
print('Training dataset attributes:')
for att in ds.attrs:
    print('  '+att+': '+str(ds.attrs[att]))

### Train the UNET ###
history = mod.fit(scaled_in_data[valid_index:split_index].values,
                  scaled_train_labels.values, 
                  batch_size=batch_size, epochs=num_epochs, verbose=1,
                  validation_data=(scaled_in_data[:valid_index].values,scaled_val_labels.values))
run_num+=1

### Save the Training History ###
epochs = np.arange(len(history.history['loss']))+1
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(epochs,history.history['loss'],'bo-',alpha=0.5,label='Training')
ax.plot(epochs,history.history['val_loss'],'rs-',alpha=0.5,label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(b=True)
plt.legend()
plt.savefig(save_path1+"LossHistory_"+f"_epochs{num_epochs}_run{run_num}"+".png", dpi=200, bbox_inches="tight")

fig, bx = plt.subplots(1, 1, figsize=(8, 4))
bx.plot(epochs,history.history['acc'],'bo-',alpha=0.5,label='Training')
bx.plot(epochs,history.history['val_acc'],'rs-',alpha=0.5,label='Validation')
bx.set_xlabel('Epoch')
bx.set_ylabel('Accuracy')
bx.grid(b=True)
plt.legend()
plt.savefig(save_path1+"AccuracyHistory_"+f"_epochs{num_epochs}_run{run_num}"+".png", dpi=200, bbox_inches="tight")

### Save the Model ### 
model_name = nn_descript+f"_epochs{num_epochs}_run{run_num}"+".h5"
save_model(mod, model_path+nn_descript+f"_epochs{num_epochs}_run{run_num}"+".h5", save_format="h5")
print('saved model as')
print(model_path+nn_descript+f"_epochs{num_epochs}_run{run_num}"+".h5")

# Save the training history
res_ds = xr.Dataset({
                    'epochs':epochs,
                    'Training_Loss':history.history['loss'],
                    'Validation_Loss':history.history['val_loss'],
                    'Training_Accuracy':history.history['acc'],
                    'Validation_Accuracy':history.history['val_acc'],
                    'split_index':split_index,
                    'valid_index':valid_index,
                    'input_variable':input_variable
                    })
res_ds.attrs['batch_size'] = batch_size
res_ds.attrs['training_data'] = ds_file
res_ds.attrs['model'] = model_name
res_ds.to_netcdf(model_path+nn_descript+f"_epochs{num_epochs}_run{run_num}_TrainingHistory.nc")

ds.close()