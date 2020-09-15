import sys 
import logging
import numpy as np
import pandas as pd
from typing import List, Dict

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, AveragePooling2D, Activation, \
    Reshape, Attention
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD

from holodecml.library.losses import *


logger = logging.getLogger(__name__)


custom_losses = {
    "sce": SymmetricCrossEntropy(0.5, 0.5),
    "weighted_mse": wsme,
    "rmse": rmse,
    "r2": R2
}


class Conv2DNeuralNetwork(object):
    """
    A Conv2D Neural Network Model that can support an arbitrary numbers of
    layers.

    Attributes:
        filters: List of number of filters in each Conv2D layer
        kernel_sizes: List of kernel sizes in each Conv2D layer
        conv2d_activation: Type of activation function for conv2d layers
        pool_sizes: List of Max Pool sizes
        dense_sizes: Sizes of dense layers
        dense_activation: Type of activation function for dense layers
        output_activation: Type of activation function for output layer
        lr: Optimizer learning rate
        optimizer: Name of optimizer or optimizer object.
        adam_beta_1: Exponential decay rate for the first moment estimates
        adam_beta_2: Exponential decay rate for the first moment estimates
        sgd_momentum: Stochastic Gradient Descent momentum
        decay: Optimizer decay
        loss: Name of loss function or loss object
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, filters=(8,), kernel_sizes=(5,),
                 conv2d_activation="relu", pool_sizes=(4,), dense_sizes=(64,),
                 dense_activation="relu", output_activation="softmax",
                 lr=0.001, optimizer="adam", adam_beta_1=0.9,
                 adam_beta_2=0.999, sgd_momentum=0.9, decay=0, loss="mse",
                 metrics = None, batch_size=32, epochs=2, verbose=0):
        self.filters = filters
        self.kernel_sizes = [tuple((v,v)) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.pool_sizes = [tuple((v,v)) for v in pool_sizes]
        self.dense_sizes = dense_sizes
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.sgd_momentum = sgd_momentum
        self.decay = decay
        self.loss = loss if loss not in custom_losses else custom_losses[loss]
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def build_neural_network(self, input_shape, output_shape):
        """Create Keras neural network model and compile it."""
        conv_input = Input(shape=(input_shape), name="input")
        nn_model = conv_input
        for h in range(len(self.filters)):
            nn_model = Conv2D(self.filters[h],
                              self.kernel_sizes[h],
                              padding="same",
                              activation=self.conv2d_activation,
                              name=f"conv2D_{h:02d}")(nn_model)
            nn_model = MaxPool2D(self.pool_sizes[h],
                                 name=f"maxpool2D_{h:02d}")(nn_model)
        nn_model = Flatten()(nn_model)
        for h in range(len(self.dense_sizes)):
            nn_model = Dense(self.dense_sizes[h],
                             activation=self.dense_activation,
                             name=f"dense_{h:02d}")(nn_model)
        nn_model = Dense(output_shape,
                         activation=self.output_activation,
                         name=f"dense_output")(nn_model)
        self.model = Model(conv_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1,
                                      beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum,
                                     decay=self.decay)
        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss, 
            metrics=self.metrics
        )
        self.model.summary()

    def fit(self, x, y, xv=None, yv=None, callbacks=None):
        if len(x.shape[1:])==2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]
        input_shape = x.shape[1:]
        self.build_neural_network(input_shape, output_shape)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=self.verbose, validation_data=(xv, yv), callbacks=callbacks)
        return self.model.history.history

    def predict(self, x):
        y_out = self.model.predict(np.expand_dims(x, axis=-1),
                                   batch_size=self.batch_size)
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob

    def load_weights(self, weights):
        try:
            self.model.load_weights(weights)
            self.model.compile(
                optimizer=self.optimizer, 
                loss=self.loss, 
                metrics=self.metrics
            )
        except:
            print("You must first call build_neural_network before loading weights. Exiting.")
            sys.exit(1)


class ParticleAttentionNet(Model):
    def __init__(self, attention_neurons=100, hidden_layers=1, hidden_neurons=100, activation="relu",
                 min_filters=4, filter_width=5, pooling="mean", pooling_width=4, filter_growth_rate=2):
        super(ParticleAttentionNet, self).__init__()
        self.attention_neurons = attention_neurons
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.min_filters = min_filters
        self.pooling = pooling
        self.pooling_width = pooling_width
        self.filter_growth_rate = filter_growth_rate
        self.filter_width = filter_width
        return

    def particle_encoder(self, input_dim=4):
        input_layer = Input(shape=(None, input_dim), name="particle_encoder_input")
        pe = input_layer
        for h in range(self.hidden_layers):
            pe = Dense(self.hidden_neurons, activation=self.activation, name=f"particle_encoder_hidden_{h:d}")(pe)
        pe = Dense(self.attention_neurons, activation=self.activation, name="particle_encoder_output")(pe)
        pe_model = Model(input_layer, pe)
        return pe_model

    def particle_decoder(self, output_dim=5):
        input_layer = Input(shape=(None, self.attention_neurons), name="particle_decoder_input")
        pd = input_layer
        for h in range(self.hidden_layers):
            pd = Dense(self.hidden_neurons, activation=self.activation, name=f"particle_decoder_hidden_{h:d}")(pd)
        pd = Dense(output_dim, activation=self.activation, name="particle_decoder_output")(pd)
        pd_model = Model(input_layer, pd)
        return pd_model

    def hologram_encoder(self, input_shape):
        input_layer = Input(shape=input_shape, name="hologram_input")
        num_conv_layers = int(np.round((np.log(input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = self.min_filters
        h_cnn = input_layer
        for c in range(num_conv_layers):
            h_cnn = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               padding="same", name="conv_{0:02d}".format(c))(h_cnn)
            h_cnn = Activation(self.activation, name="hidden_activation_{0:02d}".format(c))(h_cnn)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                h_cnn = MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                      data_format=self.data_format, name="pooling_{0:02d}".format(c))(h_cnn)
            else:
                h_cnn = AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                             data_format=self.data_format, name="pooling_{0:02d}".format(c))(h_cnn)
        h_cnn = Conv2D(self.attention_neurons, (self.filter_width, self.filter_width), padding="same")(h_cnn)
        h_cnn = Activation(self.activation, name="holo_attention_activation")(h_cnn)
        h_cnn = Reshape((h_cnn.shape[1] * h_cnn.shape[2], h_cnn.shape[3]))(h_cnn)
        h_model = Model(input_layer, h_cnn)
        return h_model

    def full_attention_model(self, particle_encoder, hologram_encoder):
        att = Attention()([particle_encoder, hologram_encoder])

    def build_network(self, particle_input_shape, hologram_input_shape, particle_output_shape):

        return


