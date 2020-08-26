
# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')

#imports we know we'll need
import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer  

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Input
import tensorflow
from tensorflow.python.keras import backend as K


import sys
import numpy as np
import xarray as xr
import json

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, SeparableConv2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, save_model, load_model, Sequential
from tensorflow.keras.utils import plot_model
import tensorflow.keras.losses
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import ml_defs as mldef


start_time = datetime.datetime.now()

input_variable = 'input_image'  # data variable used as an input
label_variable = 'histogram'  # data variable used labels for training

save_file_base = settings['data_file'].replace('.nc','')+'_HypOpt_NN_'+start_time.strftime('%Y%m%dT%H%M%S')

save_file_path = paths['save_data']+save_file_base+'/'
ml.ensure_path(save_file_path)


separate_files = False # flag indicating how to obtain test and validation data
print('Training UNET on')
print(settings['data_file'])
print('located in')
print(paths['load_data'])
# check if we are using separate files for training, test and validation
if len(settings.get('test_file','')) > 0 and len(settings.get('validation_file','')) > 0:
    separate_files = True
    print('Validating with')
    print(settings['validation_file'])
    print('Testing with')
    print(settings['test_file'])

"""
Load the input data
"""

if settings['loss_function'].lower() == 'kldivergence':
    loss_func = tensorflow.keras.losses.KLDivergence()
elif settings['loss_function'].lower() == 'kstest':
    loss_func = mldef.ks_test
elif settings['loss_function'].lower() == 'poisson':
    loss_func = mldef.poisson_nll
elif settings['loss_function'].lower() == 'cum_poisson':
    loss_func = mldef.cum_poisson_nll
else:
    loss_func = settings['loss_function']

with xr.open_dataset(paths['load_data']+settings['data_file'],chunks={'hologram_number':settings['h_chunk']}) as ds:
    print('Training dataset attributes')
    for att in ds.attrs:
        print('  '+att+': '+str(ds.attrs[att]))

    print('dataset dimensions')
    print(ds.dims)
    print(ds.sizes)

    # split data into training, validation and test data
    # ordered as (validation, test, train)
    test_index = np.int((settings['valid_fraction']+settings['test_fraction'])*ds.sizes['hologram_number'])  # number of training+validation points
    valid_index = np.int(settings['valid_fraction']*ds.sizes['hologram_number'])  # number of validation points
    
    if not separate_files:
        print('test index: %d'%test_index)
        print('validation index: %d'%valid_index)
        print('hologram count: %d'%ds.sizes['hologram_number'])

    # assign test and validation based on datasets used
    all_labels = ds[label_variable]
    if separate_files:
        train_labels = ds[label_variable]
        train_moments = ds['histogram_moments']
        if len(ds[input_variable].dims) == 4:
            train_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        elif len(ds[input_variable].dims) == 3:
            train_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        input_scaler = ml.MinMaxScalerX(train_data)
        scaled_train_input = input_scaler.fit_transform(train_data)

        with xr.open_dataset(paths['load_data']+settings['test_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_test:
            test_labels = ds_test[label_variable]
            test_moments = ds_test['histogram_moments']
            if len(ds_test[input_variable].dims) == 4:
                test_input = ds_test[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
            elif len(ds_test[input_variable].dims) == 3:
                test_data = ds_test[input_variable].transpose('hologram_number','rsize','input_channels')
            scaled_test_input = input_scaler.fit_transform(test_data)
        with xr.open_dataset(paths['load_data']+settings['validation_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_valid:
            valid_labels = ds_valid[label_variable]
            valid_moments = ds_valid['histogram_moments']
            if len(ds_valid[input_variable].dims) == 4:
                valid_input = ds_valid[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
            elif len(ds_valid[input_variable].dims) == 3:
                valid_data = ds_valid[input_variable].transpose('hologram_number','rsize','input_channels')
            scaled_valid_input = input_scaler.fit_transform(valid_data)

    else:
        train_labels = all_labels.isel(hologram_number=slice(test_index,None))
        test_labels = all_labels.isel(hologram_number=slice(valid_index,test_index))
        test_moments = ds['histogram_moments'].isel(hologram_number=slice(valid_index,test_index))
        val_labels = all_labels.isel(hologram_number=slice(None,valid_index))
        
        if len(ds[input_variable].dims) == 4:
            in_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        elif len(ds[input_variable].dims) == 3:
            in_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        
        # input scaling is not totally proper on this branch
        input_scaler = ml.MinMaxScalerX(in_data,dim=in_data.dims[1:])
        scaled_in_data = input_scaler.fit_transform(in_data)
        scaled_train_input = scaled_in_data.isel(hologram_number=slice(test_index,None))
        scaled_test_input = scaled_in_data.isel(hologram_number=slice(valid_index,test_index))
        scaled_valid_input = scaled_in_data.isel(hologram_number=slice(None,valid_index))
        
    if settings.get('scale_labels',True):
        # normalize based on training data
        output_scaler = ml.MinMaxScalerX(train_labels)
        scaled_train_labels = output_scaler.fit_transform(train_labels)
        scaled_val_labels = output_scaler.fit_transform(valid_labels)
        scaled_test_labels = output_scaler.fit_transform(test_labels)
        # output_scaler = ml.MinMaxScalerX(all_labels,dim=all_labels.dims[1:])
        # scaled_all_labels = output_scaler.fit_transform(all_labels)
    else:
        scaled_train_labels = train_labels
        scaled_val_labels = valid_labels
        scaled_test_labels = test_labels


# setup the dimensions for hyperparameter optimization
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=0, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_dense_nodes = Integer(low=1, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_batch_size = Integer(low=1, high=256, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
dim_epoch_count = Integer(low=500,high=2000,name="epoch_count")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay,
              dim_epoch_count
             ]

default_parameters = [1e-3, 1, 512, 128, 'relu', 64, 1e-3, 1000]




def create_model(learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes,activation,adam_decay,input_shape):
    
    """
    My model
    """
    
    mod = Sequential()
    mod.add(Input(shape=input_shape)) # input_shape=scaled_train_input.shape[1:]

    mod.add(Dense(num_dense_nodes,activation=activation))

    # flatten the convolution output for Dense Layers
    mod.add(Flatten())
    for _ in range(num_dense_layers):
        mod.add(Dense(num_dense_nodes,activation=activation))

    # add the output layer
    mod.add(Dense(np.prod(scaled_train_labels.shape[1:]),activation=settings['output_activation']))

    adam = Adam(lr=learning_rate, decay= adam_decay)
    mod.compile(optimizer=adam, loss=loss_func, metrics=['acc'])
    
    # mod.summary()
    
    """
    Provided Model
    """
    
    # #start the model making process and create our first layer
    # model = Sequential()
    # model.add(Dense(num_input_nodes, input_shape= input_shape, activation=activation
    #                ))
    # #create a loop making a new dense layer for the amount passed to this model.
    # #naming the layers helps avoid tensorflow error deep in the stack trace.
    # for i in range(num_dense_layers):
    #     name = 'layer_dense_{0}'.format(i+1)
    #     model.add(Dense(num_dense_nodes,
    #              activation=activation,
    #                     name=name
    #              ))
    # #add our classification layer.
    # model.add(Dense(10,activation='exponential'))
    
    # #setup our optimizer and compile
    # adam = Adam(lr=learning_rate, decay= adam_decay)
    # model.compile(optimizer=adam, loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    return mod

# define the fitness function to use for evaluating the model
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_input_nodes, 
            num_dense_nodes, activation, batch_size, adam_decay, 
            epoch_count):

    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_input_nodes=num_input_nodes,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         adam_decay=adam_decay,
                         input_shape=scaled_train_input.shape[1:]
                        )
    

    blackbox = model.fit(scaled_train_input.values,
                scaled_train_labels.values, 
                batch_size=batch_size, epochs=epoch_count, verbose=1,
                validation_data=(scaled_valid_input.values,scaled_val_labels.values))

    # #named blackbox becuase it represents the structure
    # blackbox = model.fit(x=X_train,
    #                     y=y_train,
    #                     epochs=epoch_count,
    #                     batch_size=batch_size,
    #                     )
    
    # # return the sum of the moment errors
    # m_pred = ((preds_original*(0.5*preds_original.coords['histogram_bin_centers'])**m).sum(dim=('histogram_bin_centers','output_channels')))**(1/m)

    #return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()


    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.v1.reset_default_graph()
    
    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy

# Run this code before every hyperparameter or anything that 
# makes a new Keras/Tensorflow model.
K.clear_session()
tensorflow.v1.reset_default_graph()

# minimize the fitness by tuning the hyper-parameters
gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=12,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 5,
                            x0=default_parameters)


print(gp_result.x)

# evaluate on test data
mod = create_model(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4],gp_result.x[6],scaled_train_input.shape[1:])
mod.fit(scaled_train_input.values,scaled_train_labels.values,batch_size=gp_result.x[5], epochs=gp_result.x[7])
# model.evaluate(X_test,y_test)

# evaluate the test data
print("Evaluating test data...")
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_test_input.values, batch_size=settings['batch_size'])
cnn_stop = datetime.datetime.now()
print(f"{scaled_test_input.sizes['hologram_number']} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_test_input.sizes['hologram_number']} seconds per hologram")

if len(preds_out.shape)==2:
    preds_out = preds_out[...,np.newaxis]

preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','histogram_bin_centers','output_channels'),
                            coords=scaled_test_labels.coords)

if settings.get('scale_labels',True):
    preds_original = output_scaler.inverse_transform(preds_out_da)
else:
    preds_original = preds_out_da

for m in settings.get('moments',[0,1,2,3]):
    m_pred = (preds_original*(0.5*preds_original.coords['histogram_bin_centers'])**m).sum(dim=('histogram_bin_centers','output_channels'))
    try:
        m_label = test_moments.sel(moments=m)
    except KeyError:
        print('No direct moment data')
        print('Approximating moments from histogram data')
        m_label = (test_labels*(0.5*test_labels.coords['histogram_bin_centers'])**m).sum(dim=('histogram_bin_centers','output_channels'))
    one_to_one = [m_label.values.min(),m_label.values.max()]
    fig, ax = plt.subplots() # figsize=(4,4)
    ax.scatter(m_pred,m_label,s=1,c='k')
    ax.plot(one_to_one,one_to_one,color='tab:red',linewidth=0.5)
    ax.minorticks_on()
    ax.grid(b=True)
    ax.grid(which='minor',linestyle=':')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Moment %d'%m)
    plt.savefig(save_file_path+save_file_base+f"_{m}MomentScatter.png", dpi=200, bbox_inches="tight")
    plt.close('all')


for holo_num in settings['holo_examples']:
    plt.figure()
    plt.bar(ds['histogram_bin_edges'].values[:-1],preds_original.isel(hologram_number=holo_num,output_channels=0).values,
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    plt.bar(ds['histogram_bin_edges'].values[:-1],test_labels.isel(hologram_number=holo_num,output_channels=0).values,
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='white',edgecolor='black',fill=False,label='true')
    # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    plt.xlabel('Particle Diameter [$\mu m$]')
    plt.ylabel('Count')
    if np.mean(np.diff(ds['histogram_bin_edges'].values)) != np.diff(ds['histogram_bin_edges'].values[0:2])[0]:
        plt.xscale('log')
    plt.savefig(save_file_path+save_file_base+f"_ExampleHist_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.bar(ds['histogram_bin_edges'].values[:-1],np.cumsum(preds_original.isel(hologram_number=holo_num,output_channels=0).values),
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    plt.bar(ds['histogram_bin_edges'].values[:-1],np.cumsum(test_labels.isel(hologram_number=holo_num,output_channels=0).values),
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='white',edgecolor='black',fill=False,label='true')
    # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    plt.xlabel('Particle Diameter [$\mu m$]')
    plt.ylabel('Count')
    if np.mean(np.diff(ds['histogram_bin_edges'].values)) != np.diff(ds['histogram_bin_edges'].values[0:2])[0]:
        plt.xscale('log')
    plt.savefig(save_file_path+save_file_base+f"_ExampleCDF_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    if scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values.ndim == 2:
        plt.figure()
        plt.imshow(scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values)
        plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")     
    else:
        plt.figure()
        plt.plot(scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values)
        plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")
    plt.close('all')
