import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.ndimage import gaussian_filter

from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, MaxPool2D,
                                     AveragePooling2D, Activation,
                                     Reshape, Attention, Layer)
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy

from holodecml.losses import *
from holodecml.metrics import *

logger = logging.getLogger(__name__)

custom_losses = {
    "sce": SymmetricCrossEntropy(0.5, 0.5),
    "rmse": rmse,
    "weighted_mse": wmse,
    "r2": R2,
    "attn": attention_net_loss
}

custom_metrics = {
    "TP": TruePositives,
    "FP": FalsePositives,
    "TN": TrueNegatives,
    "FN": FalseNegatives
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
                 metrics=None, batch_size=32, epochs=2, verbose=0, **kwargs):
        self.filters = filters
        self.kernel_sizes = [tuple((v, v)) for v in kernel_sizes]
        self.conv2d_activation = conv2d_activation
        self.pool_sizes = [tuple((v, v)) for v in pool_sizes]
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
        self.metrics = [custom_metrics[m]() for m in metrics]
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
            if self.pool_sizes[h][0] > 0:
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
            optimizer=self.optimizer_obj,
            loss=self.loss,
            metrics=self.metrics
        )
        self.model.summary()

    def fit(self, x, y, xv=None, yv=None, callbacks=None):
        if len(x.shape[1:]) == 2:
            x = np.expand_dims(x, axis=-1)
        if len(y.shape) == 1:
            output_shape = 1
        else:
            output_shape = y.shape[1]
        input_shape = x.shape[1:]
        self.build_neural_network(input_shape, output_shape)
        self.model.fit(x, y, batch_size=self.batch_size,
                       epochs=self.epochs, verbose=self.verbose,
                       validation_data=(xv, yv), callbacks=callbacks)
        return self.model.history.history

    def predict(self, x):
        y_out = self.model.predict(x,
                                   batch_size=self.batch_size)
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob

    def load_weights(self, weights):
        try:
            self.model.load_weights(weights)
            self.model.compile(
                optimizer=self.optimizer_obj,
                loss=self.loss,
                metrics=self.metrics
            )
        except Exception as e:
            print("You must first call build_neural_network before loading weights. Exiting.")
            sys.exit(1)

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

class ParticleEncoder(Layer):
    def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", attention_neurons=100, **kwargs):
        super(ParticleEncoder, self).__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.attention_neurons = attention_neurons
        for i in range(self.hidden_layers):
            setattr(self, f"dense_{i:02d}", Dense(self.hidden_neurons, activation=activation))
        self.attention_dense = Dense(self.attention_neurons)

    def call(self, inputs, **kwargs):
        out = inputs
        for i in range(self.hidden_layers):
            out = getattr(self, f"dense_{i:02d}")(out)
        out = self.attention_dense(out)
        return out

    def get_config(self):
        config = super().get_config()
        config["hidden_layers"] = self.hidden_layers
        config["hidden_neurons"] = self.hidden_neurons
        config["activation"] = self.activation
        config["attention_neurons"] = self.attention_neurons
        return config


class ParticleDecoder(Layer):
    def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", output_num=2, **kwargs):
        super(ParticleDecoder, self).__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_num = output_num
        for i in range(self.hidden_layers):
            setattr(self, f"dense_{i:02d}", Dense(self.hidden_neurons, activation=activation))
        self.output_dense = Dense(self.output_num)

    def call(self, inputs, **kwargs):
        out = inputs
        for i in range(self.hidden_layers):
            out = getattr(self, f"dense_{i:02d}")(out)
        out = self.output_dense(out)
        return out

    def get_config(self):
        config = super().get_config()
        config["hidden_layers"] = self.hidden_layers
        config["hidden_neurons"] = self.hidden_neurons
        config["activation"] = self.activation
        config["output_num"] = self.output_num
        return config


class HologramEncoder(Layer):
    def __init__(self, min_filters=4, filter_width=5, pooling="mean", pooling_width=2, filter_growth_rate=2,
                 min_data_width=16, attention_neurons=100, activation="relu", **kwargs):
        super(HologramEncoder, self).__init__(**kwargs)
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.pooling = pooling
        self.pooling_width = pooling_width
        self.filter_growth_rate = filter_growth_rate
        self.activation = activation
        self.attention_neurons = attention_neurons
        self.min_data_width = min_data_width
        self.num_conv_layers = 0

    def build(self, input_shape):
        self.num_conv_layers = int(np.round((np.log(input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = self.min_filters
        data_shape = np.array(input_shape[1:-1])
        for c in range(self.num_conv_layers):
            setattr(self, f"enc_conv_{c:02d}", Conv2D(num_filters, (self.filter_width, self.filter_width),
                           padding="same", activation=self.activation, name="enc_conv_{0:02d}".format(c)))
            if self.pooling.lower() == "max":
                setattr(self, f"enc_pool_{c:02d}", MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                  name="enc_pool_{0:02d}".format(c)))
            else:
                setattr(self, f"enc_pool_{c:02d}", AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                                             name="enc_pool_{0:02d}".format(c)))
            num_filters = int(num_filters * self.filter_growth_rate)
            data_shape //= self.pooling_width
        self.attention_conv = Conv2D(self.attention_neurons, (self.filter_width, self.filter_width), padding="same",
                                     activation=self.activation, name="attention_conv")
        self.attention_reshape = Reshape((data_shape[0] * data_shape[1], self.attention_neurons))

    def call(self, inputs, **kwargs):
        enc = inputs
        for c in range(self.num_conv_layers):
            enc = getattr(self, f"enc_conv_{c:02d}")(enc)
            enc = getattr(self, f"enc_pool_{c:02d}")(enc)
        att_conv = self.attention_conv(enc)
        reshape_out = self.attention_reshape(att_conv)
        return reshape_out

    def get_config(self):
        config = super().get_config()
        config["min_filters"] = self.min_filters
        config["filter_width"] = self.filter_width
        config["pooling"] = self.pooling
        config["pooling_width"] = self.pooling_width
        config["filter_growth_rate"] = self.filter_growth_rate
        config["activation"] = self.activation
        config["attention_neurons"] = self.attention_neurons
        config["num_conv_layers"] = self.num_conv_layers
        config["min_data_width"] = self.min_data_width
        return config


class ParticleAttentionNet(Model):
    def __init__(self, attention_neurons=100, hidden_layers=1, hidden_neurons=100, activation="relu",
                 min_filters=4, filter_width=5, pooling="mean", pooling_width=2, filter_growth_rate=2,
                 output_num=2, **kwargs):
        super(ParticleAttentionNet, self).__init__(**kwargs)
        self.attention_neurons = attention_neurons
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.min_filters = min_filters
        self.pooling = pooling
        self.pooling_width = pooling_width
        self.filter_growth_rate = filter_growth_rate
        self.filter_width = filter_width
        self.output_num = output_num
        self.particle_encoder = ParticleEncoder(self.hidden_layers, self.hidden_neurons, self.activation,
                                               attention_neurons=self.attention_neurons, name="particle_encoder")
        self.particle_decoder = ParticleDecoder(self.hidden_layers, self.hidden_neurons, self.activation,
                                                output_num=self.output_num, name="particle_decoder")
        self.hologram_encoder = HologramEncoder(min_filters=self.min_filters, filter_width=self.filter_width,
                                                filter_growth_rate=filter_growth_rate, pooling_width=pooling_width,
                                                activation=activation, attention_neurons=attention_neurons,
                                                name="hologram_encoder")
        self.particle_attention = Attention(use_scale=True)
        return

    def call(self, inputs, **kwargs):
        particle_enc = self.particle_encoder(inputs[0])
        # particle_enc.shape = batch_size x proposed_num_particles x self.attention_neurons 
        holo_enc = self.hologram_encoder(inputs[1])
        # holo_enc.shape = batch_size x area_of_final_conv_layer x self.attention_neurons
        attention_out = self.particle_attention([particle_enc, holo_enc])
        # attention_out.shape = batch_size x proposed_num_particles x self.attention_neurons
        particle_dec = self.particle_decoder(attention_out)
        # particle_dec.shape = batch_size x proposed_num_particles x num_coordinates (x,y,z,d,p(particle_is_real)<optional)
        return particle_dec
    
def generate_gaussian_particles(num_images=1000, num_particles=5, image_size_pixels=100,
                                gaussian_sd=3, random_seed=124):
    np.random.seed(random_seed)
    particle_pos = np.random.random(size=(num_images, num_particles, 2)).astype(np.float32)
    holo = np.zeros((num_images, image_size_pixels, image_size_pixels, 1), dtype=np.float32)
    for i in range(num_images):
        for p in range(num_particles):
            holo[i, int(particle_pos[i, p, 0] * image_size_pixels),
                 int(particle_pos[i, p, 1] * image_size_pixels)] = 1
        holo[i] = gaussian_filter(holo[i], gaussian_sd)
    return particle_pos, holo


def run_particleattentionnet():
    net = ParticleAttentionNet()
    num_images = 1000
    num_particles = 5
    image_size_pixels = 100
    filter_size = 3
    noise_sd = 0.2
    particle_pos, holo = generate_gaussian_particles(num_images=num_images, num_particles=num_particles,
                                image_size_pixels=image_size_pixels, gaussian_sd=filter_size)
    particle_pos_noisy = particle_pos * (1 + np.random.normal(0, noise_sd, particle_pos.shape))
    net.compile(optimizer="adam", loss=custom_losses["attn"])
    net.fit([particle_pos_noisy, holo], particle_pos, epochs=15, batch_size=32, verbose=1)
    pred_particle_pos = net.predict([particle_pos_noisy, holo], batch_size=128)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.contourf(holo[0, :, :, 0], cmap="Blues")
    plt.scatter(pred_particle_pos[0, :, 1] * image_size_pixels,
                pred_particle_pos[0, :, 0] * image_size_pixels, 10, "r")
    plt.scatter(particle_pos_noisy[0, :, 1] * image_size_pixels,
                particle_pos_noisy[0, :, 0] * image_size_pixels, 10, "g")
    plt.savefig("pred_particle.png", dpi=150, bbox_inches="tight")
    return

if __name__ == "__main__":
    run_particleattentionnet()
