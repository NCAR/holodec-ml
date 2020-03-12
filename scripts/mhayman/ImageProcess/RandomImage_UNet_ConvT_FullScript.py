"""
Python code to execuate training of ConvT
UNET with display plots disabled
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

# if the data files are large, performing analysis on the results
# may cause a memory overrun.  Turn the analysis off by setting to
# False.
analyze_results = False

# Training data file
ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/'   # linux share
figure_path = 'results/'

# ds_path = '/glade/scratch/mhayman/holodec/holodec-ml-data/'  # glade
# figure_path = '/glade/scratch/mhayman/holodec/holodec-ml-data/results/'

# ds_file = 'image_data_256x256_50count.nc'
# ds_file = 'image_data_256x256_5000count.nc'
# ds_file = 'image_data_64x64_5000count.nc'
# ds_file = 'random_image_data_64x64_5000count.nc'
# ds_file = 'random_image_data_64x64_5000count_v02.nc' # 1 um PSF with 1 cm depth
# ds_file = 'random_image_data_64x64_5000count_v03.nc' # 1 um PSF with 10 cm depth
# ds_file = 'random_image_multiplane_data_64x64_5000count.nc' # 1 um PSF with 1 cm depth
# ds_file = 'random_image_multiplane_data_256x256_5000count_1particles_v02.nc' # 10 um PSF with 4 cm depth
# ds_file = 'random_image_multiplane_data_256x256_5000count_1particles_v03.nc' # 10 um PSF with 1 cm depth
ds_file = 'random_image_multiplane_data_256x256_5000count_1particles_v04.nc' # 5 um PSF with 2 cm depth

ds = xr.open_dataset(ds_path+ds_file)

run_num = 0
num_epochs = 150



# Setup labels
split_index = np.int(0.7*ds.sizes['hologram_number'])  # number of training+validation points
valid_index = np.int(0.2*ds.sizes['hologram_number'])  # number of validation points
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
in_data = ds['image']

if not 'channel' in in_data.dims:
    in_data = in_data.expand_dims("channel", 3)

scaled_in_data = in_data

### Define and build the UNET ###

n_filters = 16
nPool = 4
nConv = 5
loss_fun = "mse" #,"mae" #"binary_crossentropy"
out_act = "linear" # "sigmoid"
nn_descript = f'UNET_{n_filters}Filt_{nConv}Conv_{nPool}Pool_'+loss_fun+'_'+out_act
cnn_input = Input(shape=scaled_in_data.shape[1:])  # input

conv_1a = SeparableConv2D(n_filters*1, (nConv, nConv), padding="same", kernel_initializer = "he_normal")(cnn_input)
act_1a = Activation("relu")(conv_1a)
conv_1b = SeparableConv2D(n_filters*1, (nConv, nConv), padding="same", kernel_initializer = "he_normal")(act_1a)
act_1b = Activation("relu")(conv_1b)
pool_1 = MaxPool2D(pool_size=(nPool, nPool))(act_1b)

conv_2a = SeparableConv2D(n_filters*2,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(pool_1)
act_2a = Activation("relu")(conv_2a)
conv_2b = SeparableConv2D(n_filters*2,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(act_2a)
act_2b = Activation("relu")(conv_2b)
pool_2 = MaxPool2D(pool_size=(nPool, nPool))(act_2b)

conv_3a = SeparableConv2D(n_filters*4,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(pool_2)
act_3a = Activation("relu")(conv_3a)
conv_3b = SeparableConv2D(n_filters*4,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(act_3a)
act_3b = Activation("relu")(conv_3b)
pool_3 = MaxPool2D(pool_size=(nPool, nPool))(act_3b)

conv_4a = SeparableConv2D(n_filters*8,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(pool_3)
act_4a = Activation("relu")(conv_4a)

conv_4b = SeparableConv2D(n_filters*8,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(act_4a)
act_4b = Activation("relu")(conv_4b)

upsamp_5 = Conv2DTranspose(n_filters*4, (nConv,nConv), strides=(nPool,nPool),padding="same")(act_4b)
concat_5 = concatenate([upsamp_5,act_3b],axis=3)
conv_5a = SeparableConv2D(n_filters*4,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(concat_5)
act_5a = Activation("relu")(conv_5a)
conv_5b = SeparableConv2D(n_filters*4,(nConv,nConv),padding="same", kernel_initializer = "he_normal")(act_5a)
act_5b = Activation("relu")(conv_5b)


upsamp_6 = Conv2DTranspose(n_filters*2, (nConv,nConv), strides=(nPool,nPool),padding="same")(act_5b)
concat_6 = concatenate([upsamp_6,act_2b],axis=3)
conv_6a = SeparableConv2D(n_filters*2,(nConv,nConv),padding="same",kernel_initializer = "he_normal")(concat_6)
act_6a = Activation("relu")(conv_6a)
conv_6b = SeparableConv2D(n_filters*2,(nConv,nConv),padding="same",kernel_initializer = "he_normal")(act_6a)
act_6b = Activation("relu")(conv_6b)

upsamp_7 = Conv2DTranspose(n_filters, (nConv,nConv), strides=(nPool,nPool),padding="same")(act_6b)
concat_7 = concatenate([upsamp_7,act_1b],axis=3)
conv_7a = SeparableConv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = "he_normal")(concat_7)
act_7a = Activation("relu")(conv_7a)
conv_7b = SeparableConv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = "he_normal")(act_7a)
act_7b = Activation("relu")(conv_7b)

out = Conv2D(scaled_train_labels.sizes['type'],(1,1),padding="same",activation=out_act)(act_7b)


mod = Model(cnn_input, out)
mod.compile(optimizer="adam", loss=loss_fun,metrics=['acc'])
mod.summary()
run_num=0

### End UNET Definition ###

# save a visualization of the net
plot_model(mod,show_shapes=True,to_file=figure_path+"holodec_"+nn_descript+'_'+ds_file.replace(".nc","")+".png")


### Train the UNET ###
history = mod.fit(scaled_in_data[valid_index:split_index].values,
                  scaled_train_labels.values, 
                  batch_size=64, epochs=num_epochs, verbose=1,
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
plt.savefig(figure_path+nn_descript+"_LossHistory_"+ds_file.replace(".nc","")+f"_{num_epochs}epochs_run{run_num}"+".png", dpi=200, bbox_inches="tight")

fig, bx = plt.subplots(1, 1, figsize=(8, 4))
bx.plot(epochs,history.history['acc'],'bo-',alpha=0.5,label='Training')
bx.plot(epochs,history.history['val_acc'],'rs-',alpha=0.5,label='Validation')
bx.set_xlabel('Epoch')
bx.set_ylabel('Accuracy')
bx.grid(b=True)
plt.legend()
plt.savefig(figure_path+nn_descript+"_AccuracyHistory_"+ds_file.replace(".nc","")+f"_{num_epochs}epochs_run{run_num}"+".png", dpi=200, bbox_inches="tight")


### Save the Model ### 
save_model(mod, ds_path+"/models/holodec_"+nn_descript+'_'+ds_file.replace(".nc","")+f"_{num_epochs}epochs_run{run_num}"+".h5", save_format="h5")

if analyze_results:
    ### Run the Model on All Data ###
    cnn_start = datetime.datetime.now()
    preds_out = mod.predict(scaled_in_data.values, batch_size=64)
    cnn_stop = datetime.datetime.now()
    print(f"{scaled_in_data.values.shape[0]} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
    print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_in_data.values.shape[0]} seconds per hologram")

    # create a data array for the output
    preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','xsize','ysize','type'),
                                coords=all_labels.coords)

    # create a data array with the original scaling
    preds_original = scaler.inverse_transform(preds_out_da)


    ### Plot Results ###

    # Scatter plot non-zero amplitude terms
    iscatter = np.nonzero(preds_original.sel(type='amplitude').values.flatten() > 0.2)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for a, clabel in enumerate(all_labels.coords['type'].values):
        ax=axes.ravel()[a]
        ax.scatter(all_labels.sel(type=clabel).values.flatten()[iscatter], preds_original.sel(type=clabel).values.flatten()[iscatter], 1, 'k')
        diag = np.linspace(all_labels.sel(type=clabel).min(), all_labels.sel(type=clabel).max(), 10)
        ax.plot(diag, diag, 'b--' )
        ax.set_title(clabel)
        plt.savefig(figure_path+nn_descript+f"_ScatterPlot"+f"_{num_epochs}epochs_run{run_num}_"+ds_file.replace(".nc","")+".png",dpi=300)


    # Plot the results from some example cases
    index_list = [18,2854,1247,858,3143,832,4021,3921,222,2431,321]

    diff_cmap = plt.get_cmap('seismic')
    diff_cmap.set_bad(color='black')

    for ind in index_list:
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax = ax.ravel()
        inan_mask = np.nonzero((preds_original.sel(type='amplitude',hologram_number=ind).values < 0.1)* \
            (all_labels.sel(type='amplitude',hologram_number=ind).values < 0.1))
        nan_mask = np.ones(preds_original.sel(type='amplitude',hologram_number=ind).values.shape)
        nan_mask[inan_mask] = np.nan
        ax[0].imshow(preds_original.sel(type='amplitude',hologram_number=ind).values,vmin=0,vmax=1)
        ax[1].imshow(all_labels.sel(type='amplitude',hologram_number=ind).values,vmin=0,vmax=1)
        ax[2].imshow(preds_original.sel(type='amplitude',hologram_number=ind).values-all_labels.sel(type='amplitude',hologram_number=ind).values,vmin=-1,vmax=1,cmap=diff_cmap)
        ax[3].imshow(preds_original.sel(type='z',hologram_number=ind).values*nan_mask,vmin=ds.attrs['zmin'],vmax=ds.attrs['zmax'])
        ax[4].imshow(all_labels.sel(type='z',hologram_number=ind).values*nan_mask,vmin=ds.attrs['zmin'],vmax=ds.attrs['zmax'])
        ax[5].imshow((preds_original.sel(type='z',hologram_number=ind).values-all_labels.sel(type='z',hologram_number=ind).values)*nan_mask,vmin=-1e-2,vmax=1e-2,cmap=diff_cmap)
        plt.savefig(figure_path+nn_descript+f"_ExampleImage{ind}"+f"_{num_epochs}epochs_run{run_num}_"+ds_file.replace(".nc","")+".png",dpi=300)

    # Plot the inputs used for the example cases
    channel_number = in_data.sizes['channel']
    for ind in index_list:
        fig, ax = plt.subplots(2, channel_number//2, figsize=(np.minimum(channel_number*3,12), 8))
        for ai in range(channel_number):
            axind = ai//2+np.mod(ai,2)*channel_number//2
            ax[np.mod(ai,2),ai//2].imshow(scaled_in_data.isel(channel=ai,hologram_number=ind),vmin=-2,vmax=2)
        plt.savefig(figure_path+nn_descript+f"_ExampleInput{ind}"+f"_{num_epochs}epochs_run{run_num}_"+ds_file.replace(".nc","")+".png",dpi=300)


    xg,yg = np.meshgrid(preds_original['xsize'].values*ds.attrs['pixel_width'],preds_original['ysize'].values*ds.attrs['pixel_width'])
    for ind in index_list:
        ipart = np.nonzero(preds_original.sel(type='amplitude',hologram_number=ind).values > 0.2)
        amp_p = preds_original.sel(type='amplitude',hologram_number=ind).values[ipart]
        z_p = preds_original.sel(type='z',hologram_number=ind).values[ipart]
        x_p = xg[ipart]
        y_p = yg[ipart]

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z_p,x_p,y_p,c=amp_p,vmin=0,vmax=1)
        ax.set_xlim([ds.attrs['zmin'],ds.attrs['zmax']])
        ax.set_ylim([preds_original['xsize'].values[0]*ds.attrs['pixel_width'],preds_original['xsize'].values[1]*ds.attrs['pixel_width']])
        ax.set_zlim([preds_original['ysize'].values[0]*ds.attrs['pixel_width'],preds_original['ysize'].values[1]*ds.attrs['pixel_width']])
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        plt.savefig(figure_path+nn_descript+f"_Example3DScatter{ind}"+f"_{num_epochs}epochs_run{run_num}_"+ds_file.replace(".nc","")+".png",dpi=300)
