import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD


class Conv2DNeuralNetwork(object):
    """
    A Conv2D Neural Network Model that can support arbitrary numbers of layers.

    Attributes:
        conv2D_layers: Number of Conv2D layers
        filters: List of number of filters in each Conv2D layer
        kernel_sizes: List of tuple kernel sizes in each Conv2D layer
        conv2d_activation: Type of activation function for conv2d layers
        pool_sizes: List of tuple Max Pool sizes
        debse_sizes: Sizes of dense layers
        dense_activation: Type of activation function for dense layers
        learning_rate: Optimizer learning rate
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        verbose: Level of detail to provide during training
        model: Keras Model object
    """
    def __init__(self, conv2D_layers=3, filters=8, kernel_sizes=[(5,5)], conv2d_activation="relu",
                 pool_sizes = [(4,4)], dense_sizes = [64,32,4], dense_activation="relu",
                 lr=0.001, optimizer="adam",  adam_beta_1=0.9, adam_beta_2=0.999,
                 sgd_momentum=0.9, decay=0, loss="mae", batch_size=32, epochs=2, verbose=0):
        self.conv2D_layers = conv2D_layers
        self.filters = filters
        self.kernel_sizes = [tuple(v) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.pool_sizes = [tuple(v) for v in pool_sizes]
        self.dense_sizes = dense_sizes
        self.dense_activation = dense_activation
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.sgd_momentum = sgd_momentum
        self.decay = decay
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def build_neural_network(self, inputs, outputs):
        """Create Keras neural network model and compile it."""
        conv_input = Input(shape=(600, 400, 1), name="input")
        nn_model = conv_input
        for h in range(self.conv2D_layers):
            nn_model = Conv2D(self.filters[h], self.kernel_sizes[h], padding="same",
                              activation=self.conv2d_activation, name=f"conv2D_{h:02d}")(nn_model)
            nn_model = MaxPool2D(self.pool_sizes[h], name=f"maxpool2D_{h:02d}")(nn_model)
        nn_model = Flatten()(nn_model)
        for h in range(len(self.dense_sizes[:-1])):
            nn_model = Dense(self.dense_sizes[h], activation=self.dense_activation, name=f"dense_{h:02d}")(nn_model)
        h += 1
        nn_model = Dense(self.dense_sizes[-1], name=f"dense_{h:02d}")(nn_model)
        self.model = Model(conv_input, nn_model)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.summary()

    def fit(self, x, y):
        inputs = x.shape[1]
        if len(y.shape) == 1:
            outputs = 1
        else:
            outputs = y.shape[1]
        self.build_neural_network(inputs, outputs)
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        return

    def predict(self, x):
        y_out = self.model.predict(np.expand_dims(x.values, axis=-1), batch_size=self.batch_size)
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob
