
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
import tensorflow.compat.v1
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
from tensorflow.keras.callbacks import EarlyStopping

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

# multiplier and bias
input_dct = {'mu':[1.0,0.0],'logsig':[1.0,0.0]} 
# input_dct = {'z_latent':[1.0,0.0]} 
# dmin=0
input_variable = list(input_dct.keys())
# input_variable = ['mu','logsig']  # data variable used as an input
label_variable = ['dz_hist']  # data variable used labels for training, 'd_hist'
moment_label = 'moments_nolim'  # name of the moment deat in the file

save_file_base = settings['data_file'].replace('.nc','')+'_HypOpt_NN_'+start_time.strftime('%Y%m%dT%H%M%S')

save_file_path = paths['save_data']+save_file_base+'/'
ml.ensure_path(save_file_path)


separate_files = False # flag indicating how to obtain test and validation data
print('Training on')
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
fit_kwargs = {}

if settings.get('early_stopping',False):
    # implement early stopping in fit routine
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    fit_kwargs['callbacks'] =[es]     #model.fit(..., callbacks=cb_list)


if settings['loss_function'].lower() == 'kldivergence':
    loss_func = tensorflow.keras.losses.KLDivergence()
elif settings['loss_function'].lower() == 'kstest':
    loss_func = mldef.ks_test
elif settings['loss_function'].lower() == 'poisson':
    loss_func = mldef.poisson_nll
elif settings['loss_function'].lower() == 'cum_poisson':
    loss_func = mldef.cum_poisson_nll
elif settings['loss_function'].lower() == 'cum_mse':
    loss_func = mldef.cum_mse
else:
    loss_func = settings['loss_function']

with xr.open_dataset(paths['load_data']+settings['data_file'],chunks={'hologram_number':settings['h_chunk']}) as ds:
    print('Training dataset attributes')
    for att in ds.attrs:
        print('  '+att+': '+str(ds.attrs[att]))

    print('dataset dimensions')
    print(ds.dims)
    print(ds.sizes)

    
    
    if not separate_files:
        # split data into training, validation and test data
        # ordered as (validation, test, train)
        test_index = np.int((settings['valid_fraction']+settings['test_fraction'])*ds.sizes['hologram_number'])  # number of training+validation points
        valid_index = np.int(settings['valid_fraction']*ds.sizes['hologram_number'])  # number of validation points

        print('test index: %d'%test_index)
        print('validation index: %d'%valid_index)
        print('hologram count: %d'%ds.sizes['hologram_number'])

    # assign test and validation based on datasets used
    all_labels = ds[label_variable]
    if separate_files:
        train_moments = ds[moment_label]
        train_lab_lst = []
        lab_size_lst = []
        print('training labels:')
        for lab_var in label_variable:
            print(lab_var+': %f,%f'%(ds[lab_var].max(),ds[lab_var].min()))
            train_lab_lst += [ds[lab_var].stack(dense_output=('d_bin_centers', 'z_bin_centers'))]
            lab_size_lst+=[ds[lab_var].shape[1]]
        train_labels = xr.concat(train_lab_lst,dim=('dense_output'))
        train_labels = train_labels.transpose('hologram_number','dense_output')
        hist_axes = ds['d_bin_centers']

        train_data_lst = []
        data_size_lst = []
        print('training inputs:')
        for in_var in input_variable:
            print(in_var+': %f,%f'%(ds[in_var].max(),ds[in_var].min()))
            train_data_lst += [input_dct[in_var][0]*(ds[in_var].rename({ds[in_var].dims[1]:'dense_input'})+input_dct[in_var][1])]
            data_size_lst+=[ds[in_var].shape[1]]
        train_data = xr.concat(train_data_lst,dim=('dense_input'))
        train_data = train_data.transpose('hologram_number','dense_input')

        if settings['scale_inputs']:
            # input_scaler = ml.MinMaxScalerX(train_data,dim=train_data.dims[:-1])
            input_scaler = ml.MinMaxScalerX(train_data)
            scaled_train_input = input_scaler.fit_transform(train_data)
        else:
            scaled_train_input = train_data

        with xr.open_dataset(paths['load_data']+settings['test_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_test:
            test_moments = ds_test[moment_label]

            test_lab_lst = []
            for lab_var in label_variable:
                test_lab_lst += [ds_test[lab_var].stack(dense_output=('d_bin_centers', 'z_bin_centers'))]
            test_labels = xr.concat(test_lab_lst,dim=('dense_output'))
            test_labels=test_labels.transpose('hologram_number','dense_output')

            test_data_lst = []
            for in_var in input_variable:
                test_data_lst += [input_dct[in_var][0]*(ds_test[in_var].rename({ds_test[in_var].dims[1]:'dense_input'})+input_dct[in_var][1])]
            test_data = xr.concat(test_data_lst,dim=('dense_input'))
            test_data=test_data.transpose('hologram_number','dense_input')
            if settings['scale_inputs']:
                scaled_test_input = input_scaler.fit_transform(test_data)
            else:
                scaled_test_input = test_data
        with xr.open_dataset(paths['load_data']+settings['validation_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_valid:
            valid_moments = ds_valid[moment_label]
            
            valid_lab_lst = []
            for lab_var in label_variable:
                valid_lab_lst += [ds_valid[lab_var].stack(dense_output=('d_bin_centers', 'z_bin_centers'))]
            valid_labels = xr.concat(valid_lab_lst,dim=('dense_output'))
            valid_labels=valid_labels.transpose('hologram_number','dense_output')

            valid_data_lst = []
            for in_var in input_variable:
                valid_data_lst += [input_dct[in_var][0]*(ds_valid[in_var].rename({ds_valid[in_var].dims[1]:'dense_input'})+input_dct[in_var][1])]
            valid_data = xr.concat(valid_data_lst,dim=('dense_input'))
            valid_data=valid_data.transpose('hologram_number','dense_input')

            if settings['scale_inputs']:
                scaled_valid_input = input_scaler.fit_transform(valid_data)
            else:
                scaled_valid_input = valid_data

        # train_labels = ds[label_variable]
        # train_moments = ds['histogram_moments']
        # if len(ds[input_variable].dims) == 4:
        #     train_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        # elif len(ds[input_variable].dims) == 3:
        #     train_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        # if settings.get('log_input',False):
        #     train_data = xr.concat((train_data,np.log(train_data)),dim='input_channels')
        
        # input_scaler = ml.MinMaxScalerX(train_data,dim=train_data.dims[:-1])
        # scaled_train_input = input_scaler.fit_transform(train_data)

        # with xr.open_dataset(paths['load_data']+settings['test_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_test:
        #     test_labels = ds_test[label_variable]
        #     test_moments = ds_test['histogram_moments']
        #     if len(ds_test[input_variable].dims) == 4:
        #         test_input = ds_test[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        #     elif len(ds_test[input_variable].dims) == 3:
        #         test_data = ds_test[input_variable].transpose('hologram_number','rsize','input_channels')
        #     if settings.get('log_input',False):
        #         test_data = xr.concat((test_data,np.log(test_data)),dim='input_channels')
        #     scaled_test_input = input_scaler.fit_transform(test_data)
        # with xr.open_dataset(paths['load_data']+settings['validation_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_valid:
        #     valid_labels = ds_valid[label_variable]
        #     valid_moments = ds_valid['histogram_moments']
        #     if len(ds_valid[input_variable].dims) == 4:
        #         valid_input = ds_valid[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        #     elif len(ds_valid[input_variable].dims) == 3:
        #         valid_data = ds_valid[input_variable].transpose('hologram_number','rsize','input_channels')
        #     if settings.get('log_input',False):
        #         valid_data = xr.concat((valid_data,np.log(valid_data)),dim='input_channels')
        #     scaled_valid_input = input_scaler.fit_transform(valid_data)

    else:
        print(' test and validation sets from one file not implemented ')
        # train_labels = all_labels.isel(hologram_number=slice(test_index,None))
        # test_labels = all_labels.isel(hologram_number=slice(valid_index,test_index))
        # test_moments = ds['histogram_moments'].isel(hologram_number=slice(valid_index,test_index))
        # val_labels = all_labels.isel(hologram_number=slice(None,valid_index))
        
        # if len(ds[input_variable].dims) == 4:
        #     in_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        # elif len(ds[input_variable].dims) == 3:
        #     in_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        
        # # input scaling is not totally proper on this branch
        # input_scaler = ml.MinMaxScalerX(in_data,dim=in_data.dims[1:])
        # scaled_in_data = input_scaler.fit_transform(in_data)
        # scaled_train_input = scaled_in_data.isel(hologram_number=slice(test_index,None))
        # scaled_test_input = scaled_in_data.isel(hologram_number=slice(valid_index,test_index))
        # scaled_valid_input = scaled_in_data.isel(hologram_number=slice(None,valid_index))
        
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
dim_learning_rate = Real(low=settings['learning_rate'][0], 
                        high=settings['learning_rate'][1], 
                        prior='log-uniform',
                        name='learning_rate')
dim_num_dense_layers = Integer(low=settings['num_dense_layers'][0], 
                        high=settings['num_dense_layers'][1], 
                        name='num_dense_layers')
dim_num_input_nodes = Integer(low=settings['num_input_nodes'][0], 
                        high=settings['num_input_nodes'][1], 
                        name='num_input_nodes')
dim_num_dense_nodes = Integer(low=settings['num_dense_nodes'][0], 
                        high=settings['num_dense_nodes'][1], 
                        name='num_dense_nodes')
dim_activation = Categorical(categories=settings['activation'][:-1],
                             name='activation')
dim_batch_size = Integer(low=settings['batch_size'][0], 
                        high=settings['batch_size'][1], 
                        name='batch_size')
dim_adam_decay = Real(low=settings['adam_decay'][0],
                        high=settings['adam_decay'][1],
                        name="adam_decay")
dim_epoch_count = Integer(low=settings['epoch_count'][0],
                        high=settings['epoch_count'][1],
                        name="epoch_count")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay,
              dim_epoch_count
             ]

default_parameters = [  settings['learning_rate'][2], 
                        settings['num_dense_layers'][2], 
                        settings['num_input_nodes'][2], 
                        settings['num_dense_nodes'][2], 
                        settings['activation'][-1], 
                        settings['batch_size'][2],
                        settings['adam_decay'][2], 
                        settings['epoch_count'][2]]




def create_model(learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes,activation,adam_decay,input_shape):
    
    """
    My model
    """
    print()
    print('Model design')
    mod = Sequential()
    mod.add(Input(shape=input_shape)) # input_shape=scaled_train_input.shape[1:]
    print('   Input Shape: '+str(input_shape))
    # flatten the convolution output for Dense Layers
    mod.add(Flatten())

    print('   Layer 0: '+str(num_input_nodes)+', '+activation)
    mod.add(Dense(num_input_nodes,activation=activation))

    for lyr_cnt in range(num_dense_layers):
        print(f'   Layer {lyr_cnt+1}: '+str(num_dense_nodes)+', '+activation)
        mod.add(Dense(num_dense_nodes,activation=activation))

    # add the output layer
    print('   Output Layer: '+str(np.prod(scaled_train_labels.shape[1:]))+', '+settings['output_activation'])
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
                validation_data=(scaled_valid_input.values,scaled_val_labels.values),
                **fit_kwargs)

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
    loss_val = blackbox.history['val_loss'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print("Loss: {0:.6}".format(loss_val))
    print()


    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.compat.v1.reset_default_graph()
    
    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy

# Run this code before every hyperparameter or anything that 
# makes a new Keras/Tensorflow model.
K.clear_session()
tensorflow.compat.v1.reset_default_graph()

# minimize the fitness by tuning the hyper-parameters
gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            n_calls=settings['n_calls'],
                            noise= settings['noise'],
                            n_jobs=settings['n_jobs'],
                            kappa = settings['kappa'],
                            x0=default_parameters)


# evaluate on test data
mod = create_model(gp_result.x[0],gp_result.x[1],gp_result.x[2],gp_result.x[3],gp_result.x[4],gp_result.x[6],scaled_train_input.shape[1:])
# save a visualization
plot_model(mod,show_shapes=True,to_file=save_file_path+save_file_base+"_diagram.png")
# train optimal model
# history = mod.fit(scaled_train_input.values,scaled_train_labels.values,batch_size=gp_result.x[5], epochs=gp_result.x[7])
history = mod.fit(scaled_train_input.values,
                scaled_train_labels.values, 
                batch_size=gp_result.x[5], epochs=gp_result.x[7], verbose=1,
                validation_data=(scaled_valid_input.values,scaled_val_labels.values),
                **fit_kwargs)

print()
print('Solution Results')
print(gp_result.x)
print()

### Save the Training History ###
epochs = np.arange(len(history.history['loss']))+1
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(epochs,history.history['loss'],'bo-',alpha=0.5,label='Training')
ax.plot(epochs,history.history['val_loss'],'rs-',alpha=0.5,label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
if np.sum(np.array(history.history['loss'])<=0) == 0:
    ax.set_yscale('log')
ax.grid(b=True)
plt.legend()
plt.savefig(save_file_path+save_file_base+"_LossHistory.png", dpi=200, bbox_inches="tight")

fig, bx = plt.subplots(1, 1, figsize=(8, 4))
bx.plot(epochs,history.history['acc'],'bo-',alpha=0.5,label='Training')
bx.plot(epochs,history.history['val_acc'],'rs-',alpha=0.5,label='Validation')
bx.set_xlabel('Epoch')
bx.set_ylabel('Accuracy')
bx.grid(b=True)
plt.legend()
plt.savefig(save_file_path+save_file_base+"_AccuracyHistory.png", dpi=200, bbox_inches="tight")

### Save the Model ### 
model_name = save_file_base+".h5"
save_model(mod, save_file_path+model_name, save_format="h5")
print('saved model as')
print(save_file_path+model_name)

# evaluate the test data
print("Evaluating test data...")
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_test_input.values, batch_size=gp_result.x[5])
cnn_stop = datetime.datetime.now()
print(f"{scaled_test_input.sizes['hologram_number']} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_test_input.sizes['hologram_number']} seconds per hologram")

# if len(preds_out.shape)==2:
#     preds_out = preds_out[...,np.newaxis]
# 
# preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','dense_output'),
#                             coords=scaled_test_labels.coords)

# diameter
preds_out_da = xr.DataArray(preds_out.reshape(-1,ds['d_bin_centers'].size,ds['z_bin_centers'].size),
                            dims=('hologram_number','d_bin_centers','z_bin_centers'),
                            coords={'hologram_number':scaled_test_labels.coords['hologram_number'],
                                'd_bin_centers':hist_axes})

# z
# preds_out_da = xr.DataArray(preds_out,
#                             dims=('hologram_number','z_bin_centers'),
#                             coords={'hologram_number':scaled_test_labels.coords['hologram_number'],
#                                 'z_bin_centers':hist_axes})

if settings.get('scale_labels',True):
    preds_original = output_scaler.inverse_transform(preds_out_da)
else:
    preds_original = preds_out_da

for m in settings.get('moments',[0,1,2,3]):
    m_pred = (preds_original*(0.5*preds_original.coords['d_bin_centers'])**m).sum(dim=('d_bin_centers','z_bin_centers'))
    try:
        m_label = test_moments.sel(moments=m)
    except KeyError:
        print('No direct moment data')
        print('Approximating moments from histogram data')
        m_label = (test_labels*(0.5*test_labels.coords['d_bin_centers'])**m).sum(dim=('d_bin_centers','z_bin_centers'))
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
    

    fig,ax_lst = plt.subplots(1,2,figsize=(10,5))
    ax_lst[0].pcolormesh(ds['d_bin_edges'],ds['z_bin_edges'],preds_original.isel(hologram_number=holo_num).values.T)
    ax_lst[0].set_xlabel('diameter [$mu m$]')
    ax_lst[0].set_ylabel('z position [$mu m$]')
    ax_lst[0].set_title('Predicted')

    ax_lst[1].pcolormesh(ds['d_bin_edges'],ds['z_bin_edges'],test_labels.unstack('dense_output').isel(hologram_number=holo_num).values.T)
    ax_lst[1].set_xlabel('diameter [$mu m$]')
    # ax_lst[1].set_ylabel('z position [$mu m$]')
    ax_lst[1].set_title('Actual')

    # plt.bar(ds['d_bin_edges'].values[:-1],preds_original.isel(hologram_number=holo_num).values,
    #         np.diff(ds['d_bin_edges'].values),
    #         facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    # plt.bar(ds['d_bin_edges'].values[:-1],test_labels.isel(hologram_number=holo_num).values,
    #         np.diff(ds['d_bin_edges'].values),
    #         facecolor='white',edgecolor='black',fill=False,label='true')
    # # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    # plt.xlabel('Particle Diameter [$\mu m$]')
    # plt.ylabel('Count')
    # if np.mean(np.diff(ds['d_bin_edges'].values)) != np.diff(ds['d_bin_edges'].values[0:2])[0]:
    #     plt.xscale('log')
    plt.savefig(save_file_path+save_file_base+f"_ExampleHist_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    # plt.figure()
    # plt.bar(ds['d_bin_edges'].values[:-1],np.cumsum(preds_original.isel(hologram_number=holo_num).values),
    #         np.diff(ds['d_bin_edges'].values),
    #         facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    # plt.bar(ds['d_bin_edges'].values[:-1],np.cumsum(test_labels.isel(hologram_number=holo_num).values),
    #         np.diff(ds['d_bin_edges'].values),
    #         facecolor='white',edgecolor='black',fill=False,label='true')
    # # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    # plt.xlabel('Particle Diameter [$\mu m$]')
    # plt.ylabel('Count')
    # if np.mean(np.diff(ds['d_bin_edges'].values)) != np.diff(ds['d_bin_edges'].values[0:2])[0]:
    #     plt.xscale('log')
    # plt.savefig(save_file_path+save_file_base+f"_ExampleCDF_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(scaled_test_input.isel(hologram_number=holo_num).values,'.')
    # plt.legend()
    plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    # if scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values.ndim == 2:
    #     plt.figure()
    #     plt.imshow(scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values)
    #     plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")     
    # else:
    #     plt.figure()
    #     for i_chan in range(scaled_test_input.coords['input_channels'].size):
    #         plt.plot(scaled_test_input.isel(hologram_number=holo_num,input_channels=i_chan).values,label='channel %d'%i_chan)
    #     plt.legend()
    #     plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    plt.close('all')


json_dct = {'settings':settings,'paths':paths}
    
with open(save_file_path+save_file_base+"_run_settings.json", 'w') as fp:
    json.dump(json_dct, fp, indent=4)