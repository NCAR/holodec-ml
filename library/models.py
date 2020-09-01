import sherpa

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD


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
                 batch_size=32, epochs=2, verbose=0, sherpa=False, study=None,
                 trial=False):
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
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.sherpa = sherpa
        self.study = study
        self.trial = trial
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
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=[TruePositives(), FalsePositives(),
                                    FalseNegatives(), TrueNegatives()])
        self.model.summary()

    def fit(self, x, y, xv=None, yv=None):
        if len(x.shape[1:])==2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]
        input_shape = x.shape[1:]
        self.build_neural_network(input_shape, output_shape)
        if self.sherpa:
            sherpa_cb = self.study.keras_callback(self.trial,
                                                  objective_name='val_loss')
            self.model.fit(x, y, batch_size=self.batch_size,
                           epochs=self.epochs, verbose=self.verbose,
                           validation_data=(xv, yv),
                           callbacks=[sherpa_cb])            
        else:
            self.model.fit(x, y, batch_size=self.batch_size,
                           epochs=self.epochs, verbose=self.verbose,
                           validation_data=(xv, yv))
        return self.model.history.history

    def predict(self, x):
        y_out = self.model.predict(x,
                                   batch_size=self.batch_size)
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob
    
    def saliency(self, x, layer_index=-3, ref_activation=10):
        """
        Output the gradient of input field with respect to each neuron in the specified layer.
        Args:
            x:
            layer_index:
            ref_activation: Reference activation value for loss function.
        Returns:
        """
        saliency_values = np.zeros((self.model.layers[layer_index].output.shape[-1],
                                    x.shape[0], x.shape[1],
                                    x.shape[2], x.shape[3]),
                                   dtype=np.float32)
        for s in trange(self.model.layers[layer_index].output.shape[-1], desc="neurons"):
            sub_model = Model(self.model.input, self.model.layers[layer_index].output[:, s])
            batch_indices = np.append(np.arange(0, x.shape[0], self.batch_size), x.shape[0])
            for b, batch_index in enumerate(tqdm(batch_indices[:-1], desc="batch examples", leave=False)):
                x_case = tf.Variable(x[batch_index:batch_indices[b + 1]])
                with tf.GradientTape() as tape:
                    tape.watch(x_case)
                    act_out = sub_model(x_case)
                    loss = (ref_activation - act_out) ** 2
                saliency_values[s, batch_index:batch_indices[b + 1]] = tape.gradient(loss, x_case)

        return saliency_values
    
    def output_hidden_layer(self, x, batch_size=1024, layer_index=-3):
        """
        Chop the end off the neural network and capture the output from the specified layer index
        Args:
            x: input data
            layer_index (int): list index of the layer being output.
        Returns:
            output: array containing output of that layer for each example.
        """
        sub_model = Model(self.model.input, self.model.layers[layer_index].output)
        output = sub_model.predict(x, batch_size=batch_size)
        return output

class TruePositives(tf.metrics.Metric):
    def __init__(self, name="true_pos", **kwargs):
        super(TruePositives, self).__init__(name=name, **kwargs)
        self.true_pos = self.add_weight(name="tp", dtype="float64",
                                        initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.true_pos.assign(cm[1,1] / tf.keras.backend.sum(cm))

    def result(self):
        return self.true_pos

    def reset_states(self):
        self.true_pos.assign(0.0)

class FalsePositives(tf.metrics.Metric):
    def __init__(self, name="false_pos", **kwargs):
        super(FalsePositives, self).__init__(name=name, **kwargs)
        self.false_pos = self.add_weight(name="fp", dtype="float64",
                                         initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.false_pos.assign(cm[0,1]/tf.keras.backend.sum(cm))

    def result(self):
        return self.false_pos

    def reset_states(self):
        self.false_pos.assign(0.0)

class FalseNegatives(tf.metrics.Metric):
    def __init__(self, name="false_neg", **kwargs):
        super(FalseNegatives, self).__init__(name=name, **kwargs)
        self.false_neg = self.add_weight(name="fn", dtype="float64",
                                         initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1/y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.false_neg.assign(cm[1,0]/tf.keras.backend.sum(cm))

    def result(self):
        return self.false_neg

    def reset_states(self):
        self.false_neg.assign(0.0)

class TrueNegatives(tf.metrics.Metric):
    def __init__(self, name="true_neg", **kwargs):
        super(TrueNegatives, self).__init__(name=name, **kwargs)
        self.true_neg = self.add_weight(name="tn", dtype="float64",
                                        initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        true = tf.keras.backend.flatten(tf.where(y_true > 0, 1, 0))
        pred = tf.keras.backend.flatten(tf.where(y_pred > 1 / y_pred.shape[1],
                                                 1, 0))
        cm = tf.math.confusion_matrix(true, pred)
        self.true_neg.assign(cm[0,0] / tf.keras.backend.sum(cm))

    def result(self):
        return self.true_neg

    def reset_states(self):
        self.true_neg.assign(0.0)
        