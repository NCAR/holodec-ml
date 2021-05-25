import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.ndimage import gaussian_filter

from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, MaxPool2D,
                                     AveragePooling2D, Activation,
                                     Reshape, Attention, Layer, BatchNormalization,
                                     Conv2DTranspose, MaxPooling2D, Dropout,
                                     SpatialDropout2D, UpSampling2D, concatenate,
                                     multiply, add, Activation)
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
    "noisy": noisy_true_particle_loss,
    "random": random_particle_distance_loss
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
            optimizer=self.optimizer,
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
                optimizer=self.optimizer,
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

class ParticleEncoder(Model):
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


class ParticleDecoder(Model):
    def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", output_num=2, **kwargs):
        super(ParticleDecoder, self).__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_num = output_num
        for i in range(self.hidden_layers):
            setattr(self, f"dense_{i:02d}", Dense(self.hidden_neurons, activation=activation))
        self.output_dense = Dense(self.output_num, activation=None)

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


class HologramEncoder(Model):
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
        self.particle_attention = Attention(use_scale=False)
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
    net.compile(optimizer="adam", loss=custom_losses["noisy"], metrics="noisy")
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

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. in their Attention U-net, see: https://arxiv.org/abs/1804.03999.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])

def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])

def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c

def custom_unet(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=3,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def custom_jnet(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=3,
    pool_down=[2,2,2],
    pool_up=[2,2,2],
    skip_pool=10,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple
    
    pool_down = [(p,p) for p in pool_down]
    pool_up = [(p,p) for p in pool_up]
    
    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for i,l in enumerate(range(len(pool_down))):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D(pool_down[i])(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    for i, down_layer in enumerate(down_layers):
        print(f"down_layer {i} shape: {down_layer.shape}")
    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for i, conv in enumerate(reversed(down_layers[:len(pool_up)])):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        print("x.shape before upsample", x.shape)
        x = upsample(filters, pool_up[i], strides=pool_up[i], padding="same")(x)
        if use_attention:
            print("conv.shape before MaxPool", conv.shape)
            conv = MaxPooling2D((skip_pool, skip_pool))(conv)
            print("conv.shape after MaxPool", conv.shape)
            print("x.shape after upsample", x.shape)
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def custom_jnet_full(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=3,
    pool_down=[2,2,2],
    pool_up=[2,2,2],
    skip_pool=10,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple
    
    pool_down = [(p,p) for p in pool_down]
    pool_up = [(p,p) for p in pool_up]
    
    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for i,l in enumerate(range(len(pool_down))):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D(pool_down[i])(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for i, conv in enumerate(down_layers):
        print("conv: ", i, conv.shape)

    down_layers[0] = MaxPooling2D((20, 20))(down_layers[0])
    down_layers[1] = MaxPooling2D((10, 10))(down_layers[1])
    down_layers[2] = MaxPooling2D((5, 5))(down_layers[2])
    down_layers[3] = MaxPooling2D((5, 5))(down_layers[3])
    down_layers[3] = upsample(down_layers[3].shape[3], (2,2), strides=(2,2), padding="same")(down_layers[3])
    
    for i, conv in enumerate(down_layers):
        print("conv: ", i, conv.shape)
        
    for i in range(len(pool_up)):
        print("I ------", i)
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        print("x before upsample", x.shape)
        print("pool_up[i]: ", pool_up[i])
        x = upsample(filters, pool_up[i], strides=pool_up[i], padding="same")(x)
        print("x after upsample", x.shape)
        if use_attention:
            for conv in down_layers:
                if i > 0:
                    print("conv before upsample", conv.shape)
                    up = (2*i, 2*i)
                    print("(2*i,2*i): ", up)
                    conv = upsample(conv.shape[3], (1,1), strides=(1,1), padding="same")(x)
                    print("conv after upsample", conv.shape)
                print(x.shape)
                x = attention_concat(conv_below=x, skip_connection=conv)
                print(x.shape)
        else:
            for conv in down_layers:
                x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def jnet_512(
    input_shape,
    num_classes=1,
    activation="relu",
    use_batch_norm=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    dropout=0.3,
    dropout_change_per_layer=0.0,
    dropout_type="spatial",
    use_dropout_on_upsampling=False,
    use_attention=False,
    filters=16,
    num_layers=3,
    pool_down=[2,2,2],
    pool_up=[2,2,2],
    skip_pool=10,
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    """
    Customisable UNet architecture (Ronneberger et al. 2015 [1]).
    Arguments:
    input_shape: 3D Tensor of shape (x, y, num_channels)
    num_classes (int): Unique classes in the output mask. Should be set to 1 for binary segmentation
    activation (str): A keras.activations.Activation to use. ReLu by default.
    use_batch_norm (bool): Whether to use Batch Normalisation across the channel axis between convolutional layers
    upsample_mode (one of "deconv" or "simple"): Whether to use transposed convolutions or simple upsampling in the decoder part
    dropout (float between 0. and 1.): Amount of dropout after the initial convolutional block. Set to 0. to turn Dropout off
    dropout_change_per_layer (float between 0. and 1.): Factor to add to the Dropout after each convolutional block
    dropout_type (one of "spatial" or "standard"): Type of Dropout to apply. Spatial is recommended for CNNs [2]
    use_dropout_on_upsampling (bool): Whether to use dropout in the decoder part of the network
    use_attention (bool): Whether to use an attention dynamic when concatenating with the skip-connection, implemented as proposed by Oktay et al. [3]
    filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block
    num_layers (int): Number of total layers in the encoder not including the bottleneck layer
    output_activation (str): A keras.activations.Activation to use. Sigmoid by default for binary segmentation
    Returns:
    model (keras.models.Model): The built U-Net
    Raises:
    ValueError: If dropout_type is not one of "spatial" or "standard"
    """

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple
    
    pool_down = [(p,p) for p in pool_down]
    pool_up = [(p,p) for p in pool_up]
    
    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for i,l in enumerate(range(len(pool_down))):
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )
        down_layers.append(x)
        x = MaxPooling2D(pool_down[i])(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(
        inputs=x,
        filters=filters,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        dropout_type=dropout_type,
        activation=activation,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0
        
    for i, conv in enumerate(down_layers):
        print(i, conv.shape)

    for i, conv in enumerate(reversed(down_layers[-len(pool_up):])):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        print("x before upsample shape", x.shape)
        x = upsample(filters, pool_up[i], strides=pool_up[i], padding="same")(x)
        print("x after upsample shape", x.shape)
        if use_attention:
            print("conv shape before max pool", conv.shape)
#             conv = MaxPooling2D((skip_pool, skip_pool))(conv)
            print("conv shape after max pool", conv.shape)
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            filters=filters,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            dropout_type=dropout_type,
            activation=activation,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
