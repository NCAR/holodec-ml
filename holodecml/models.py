import sys
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.ndimage import gaussian_filter

# from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, MaxPool2D,
#                                      AveragePooling2D, Activation,
#                                      Reshape, Attention, Layer)
# import tensorflow as tf
# from tensorflow.keras.models import Model, save_model
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy

from holodecml.losses import *
from holodecml.metrics import *

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


logger = logging.getLogger(__name__)



# custom_losses = {
#     "sce": SymmetricCrossEntropy(0.5, 0.5),
#     "rmse": rmse,
#     "r2": R2,
#     "noisy": noisy_true_particle_loss,
#     "random": random_particle_distance_loss
# }

# custom_metrics = {
#     "TP": TruePositives,
#     "FP": FalsePositives,
#     "TN": TrueNegatives,
#     "FN": FalseNegatives
# } 



class ResNet(nn.Module):
    def __init__(self, fcl_layers = [], dr = 0.0, color_dim = 2, output_size = 1, resnet_model = 18, pretrained = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model 
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(color_dim, 64, (7, 7), (2, 2), (3, 3), bias=False) # Manually change color dim to match our data
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fcn = self.make_fcn(self.resnet_output_dim, output_size, fcl_layers, dr)
        
        # All pretrained models utilize this transformation for 3 color channels, so just using average for 1 channel data here
        self.mean = np.mean([0.485, 0.456, 0.406])
        self.std = np.mean([0.229, 0.224, 0.225])
        
    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers)-1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i+1]),
                        nn.BatchNorm1d(fcl_layers[i+1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i+1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

    
    
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )



class ResNetUNet(nn.Module):
    def __init__(self, n_class, color_dim = 2):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = torch.nn.Conv2d(color_dim, 64, (7, 7), (2, 2), (3, 3), bias=False) 
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(color_dim , 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        out = torch.nn.Sigmoid()(out)

        return out
    

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    
class UNet(nn.Module):
    def __init__(self,
                 enc_chs=(2, 64, 128, 256, 512, 1024),
                 dec_chs=(1024, 512, 256, 128, 64),
                 num_class=1,
                 retain_dim=False, out_sz=(572,572)):
        
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz
        
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        out = torch.nn.Sigmoid()(out)
        return out


# class Conv2DNeuralNetwork(object):
#     """
#     A Conv2D Neural Network Model that can support an arbitrary numbers of
#     layers.

#     Attributes:
#         filters: List of number of filters in each Conv2D layer
#         kernel_sizes: List of kernel sizes in each Conv2D layer
#         conv2d_activation: Type of activation function for conv2d layers
#         pool_sizes: List of Max Pool sizes
#         dense_sizes: Sizes of dense layers
#         dense_activation: Type of activation function for dense layers
#         output_activation: Type of activation function for output layer
#         lr: Optimizer learning rate
#         optimizer: Name of optimizer or optimizer object.
#         adam_beta_1: Exponential decay rate for the first moment estimates
#         adam_beta_2: Exponential decay rate for the first moment estimates
#         sgd_momentum: Stochastic Gradient Descent momentum
#         decay: Optimizer decay
#         loss: Name of loss function or loss object
#         batch_size: Number of examples per batch
#         epochs: Number of epochs to train
#         verbose: Level of detail to provide during training
#         model: Keras Model object
#     """

#     def __init__(self, filters=(8,), kernel_sizes=(5,),
#                  conv2d_activation="relu", pool_sizes=(4,), dense_sizes=(64,),
#                  dense_activation="relu", output_activation="softmax",
#                  lr=0.001, optimizer="adam", adam_beta_1=0.9,
#                  adam_beta_2=0.999, sgd_momentum=0.9, decay=0, loss="mse",
#                  metrics=None, batch_size=32, epochs=2, verbose=0, **kwargs):
#         self.filters = filters
#         self.kernel_sizes = [tuple((v, v)) for v in kernel_sizes]
#         self.conv2d_activation = conv2d_activation
#         self.pool_sizes = [tuple((v, v)) for v in pool_sizes]
#         self.dense_sizes = dense_sizes
#         self.dense_activation = dense_activation
#         self.output_activation = output_activation
#         self.lr = lr
#         self.optimizer = optimizer
#         self.optimizer_obj = None
#         self.adam_beta_1 = adam_beta_1
#         self.adam_beta_2 = adam_beta_2
#         self.sgd_momentum = sgd_momentum
#         self.decay = decay
#         self.loss = loss if loss not in custom_losses else custom_losses[loss]
#         self.metrics = [custom_metrics[m]() for m in metrics]
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.verbose = verbose
#         self.model = None

#     def build_neural_network(self, input_shape, output_shape):
#         """Create Keras neural network model and compile it."""
#         conv_input = Input(shape=(input_shape), name="input")
#         nn_model = conv_input
#         for h in range(len(self.filters)):
#             nn_model = Conv2D(self.filters[h],
#                               self.kernel_sizes[h],
#                               padding="same",
#                               activation=self.conv2d_activation,
#                               name=f"conv2D_{h:02d}")(nn_model)
#             if self.pool_sizes[h][0] > 0:
#                 nn_model = MaxPool2D(self.pool_sizes[h],
#                                      name=f"maxpool2D_{h:02d}")(nn_model)
#         nn_model = Flatten()(nn_model)
#         for h in range(len(self.dense_sizes)):
#             nn_model = Dense(self.dense_sizes[h],
#                              activation=self.dense_activation,
#                              name=f"dense_{h:02d}")(nn_model)
#         nn_model = Dense(output_shape,
#                          activation=self.output_activation,
#                          name=f"dense_output")(nn_model)
#         self.model = Model(conv_input, nn_model)
#         if self.optimizer == "adam":
#             self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1,
#                                       beta_2=self.adam_beta_2, decay=self.decay)
#         elif self.optimizer == "sgd":
#             self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum,
#                                      decay=self.decay)
#         self.model.compile(
#             optimizer=self.optimizer_obj,
#             loss=self.loss,
#             metrics=self.metrics
#         )
#         self.model.summary()

#     def fit(self, x, y, xv=None, yv=None, callbacks=None):
#         if len(x.shape[1:]) == 2:
#             x = np.expand_dims(x, axis=-1)
#         if len(y.shape) == 1:
#             output_shape = 1
#         else:
#             output_shape = y.shape[1]
#         input_shape = x.shape[1:]
#         self.build_neural_network(input_shape, output_shape)
#         self.model.fit(x, y, batch_size=self.batch_size,
#                        epochs=self.epochs, verbose=self.verbose,
#                        validation_data=(xv, yv), callbacks=callbacks)
#         return self.model.history.history

#     def predict(self, x):
#         y_out = self.model.predict(x,
#                                    batch_size=self.batch_size)
#         return y_out

#     def predict_proba(self, x):
#         y_prob = self.model.predict(x, batch_size=self.batch_size)
#         return y_prob

#     def load_weights(self, weights):
#         try:
#             self.model.load_weights(weights)
#             self.model.compile(
#                 optimizer=self.optimizer_obj,
#                 loss=self.loss,
#                 metrics=self.metrics
#             )
#         except Exception as e:
#             print("You must first call build_neural_network before loading weights. Exiting.")
#             sys.exit(1)

#     def saliency(self, x, layer_index=-3, ref_activation=10):
#         """
#         Output the gradient of input field with respect to each neuron in the specified layer.
#         Args:
#             x:
#             layer_index:
#             ref_activation: Reference activation value for loss function.
#         Returns:
#         """
#         saliency_values = np.zeros((self.model.layers[layer_index].output.shape[-1],
#                                     x.shape[0], x.shape[1],
#                                     x.shape[2], x.shape[3]),
#                                    dtype=np.float32)
#         for s in trange(self.model.layers[layer_index].output.shape[-1], desc="neurons"):
#             sub_model = Model(self.model.input, self.model.layers[layer_index].output[:, s])
#             batch_indices = np.append(np.arange(0, x.shape[0], self.batch_size), x.shape[0])
#             for b, batch_index in enumerate(tqdm(batch_indices[:-1], desc="batch examples", leave=False)):
#                 x_case = tf.Variable(x[batch_index:batch_indices[b + 1]])
#                 with tf.GradientTape() as tape:
#                     tape.watch(x_case)
#                     act_out = sub_model(x_case)
#                     loss = (ref_activation - act_out) ** 2
#                 saliency_values[s, batch_index:batch_indices[b + 1]] = tape.gradient(loss, x_case)

#         return saliency_values
    
#     def output_hidden_layer(self, x, batch_size=1024, layer_index=-3):
#         """
#         Chop the end off the neural network and capture the output from the specified layer index
#         Args:
#             x: input data
#             layer_index (int): list index of the layer being output.
#         Returns:
#             output: array containing output of that layer for each example.
#         """
#         sub_model = Model(self.model.input, self.model.layers[layer_index].output)
#         output = sub_model.predict(x, batch_size=batch_size)
#         return output            

# class ParticleEncoder(Layer):
#     def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", attention_neurons=100, **kwargs):
#         super(ParticleEncoder, self).__init__(**kwargs)
#         self.hidden_layers = hidden_layers
#         self.hidden_neurons = hidden_neurons
#         self.activation = activation
#         self.attention_neurons = attention_neurons
#         for i in range(self.hidden_layers):
#             setattr(self, f"dense_{i:02d}", Dense(self.hidden_neurons, activation=activation))
#         self.attention_dense = Dense(self.attention_neurons)

#     def call(self, inputs, **kwargs):
#         out = inputs
#         for i in range(self.hidden_layers):
#             out = getattr(self, f"dense_{i:02d}")(out)
#         out = self.attention_dense(out)
#         return out

#     def get_config(self):
#         config = super().get_config()
#         config["hidden_layers"] = self.hidden_layers
#         config["hidden_neurons"] = self.hidden_neurons
#         config["activation"] = self.activation
#         config["attention_neurons"] = self.attention_neurons
#         return config


# class ParticleDecoder(Layer):
#     def __init__(self, hidden_layers=1, hidden_neurons=10, activation="relu", output_num=2, **kwargs):
#         super(ParticleDecoder, self).__init__(**kwargs)
#         self.hidden_layers = hidden_layers
#         self.hidden_neurons = hidden_neurons
#         self.activation = activation
#         self.output_num = output_num
#         for i in range(self.hidden_layers):
#             setattr(self, f"dense_{i:02d}", Dense(self.hidden_neurons, activation=activation))
#         self.output_dense = Dense(self.output_num, activation=None)

#     def call(self, inputs, **kwargs):
#         out = inputs
#         for i in range(self.hidden_layers):
#             out = getattr(self, f"dense_{i:02d}")(out)
#         out = self.output_dense(out)
#         return out

#     def get_config(self):
#         config = super().get_config()
#         config["hidden_layers"] = self.hidden_layers
#         config["hidden_neurons"] = self.hidden_neurons
#         config["activation"] = self.activation
#         config["output_num"] = self.output_num
#         return config


# class HologramEncoder(Layer):
#     def __init__(self, min_filters=4, filter_width=5, pooling="mean", pooling_width=2, filter_growth_rate=2,
#                  min_data_width=16, attention_neurons=100, activation="relu", **kwargs):
#         super(HologramEncoder, self).__init__(**kwargs)
#         self.min_filters = min_filters
#         self.filter_width = filter_width
#         self.pooling = pooling
#         self.pooling_width = pooling_width
#         self.filter_growth_rate = filter_growth_rate
#         self.activation = activation
#         self.attention_neurons = attention_neurons
#         self.min_data_width = min_data_width
#         self.num_conv_layers = 0

#     def build(self, input_shape):
#         self.num_conv_layers = int(np.round((np.log(input_shape[1]) - np.log(self.min_data_width))
#                                        / np.log(self.pooling_width)))
#         num_filters = self.min_filters
#         data_shape = np.array(input_shape[1:-1])
#         for c in range(self.num_conv_layers):
#             setattr(self, f"enc_conv_{c:02d}", Conv2D(num_filters, (self.filter_width, self.filter_width),
#                            padding="same", activation=self.activation, name="enc_conv_{0:02d}".format(c)))
#             if self.pooling.lower() == "max":
#                 setattr(self, f"enc_pool_{c:02d}", MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
#                                   name="enc_pool_{0:02d}".format(c)))
#             else:
#                 setattr(self, f"enc_pool_{c:02d}", AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
#                                                              name="enc_pool_{0:02d}".format(c)))
#             num_filters = int(num_filters * self.filter_growth_rate)
#             data_shape //= self.pooling_width
#         self.attention_conv = Conv2D(self.attention_neurons, (self.filter_width, self.filter_width), padding="same",
#                                      activation=self.activation, name="attention_conv")
#         self.attention_reshape = Reshape((data_shape[0] * data_shape[1], self.attention_neurons))

#     def call(self, inputs, **kwargs):
#         enc = inputs
#         for c in range(self.num_conv_layers):
#             enc = getattr(self, f"enc_conv_{c:02d}")(enc)
#             enc = getattr(self, f"enc_pool_{c:02d}")(enc)
#         att_conv = self.attention_conv(enc)
#         reshape_out = self.attention_reshape(att_conv)
#         return reshape_out

#     def get_config(self):
#         config = super().get_config()
#         config["min_filters"] = self.min_filters
#         config["filter_width"] = self.filter_width
#         config["pooling"] = self.pooling
#         config["pooling_width"] = self.pooling_width
#         config["filter_growth_rate"] = self.filter_growth_rate
#         config["activation"] = self.activation
#         config["attention_neurons"] = self.attention_neurons
#         config["num_conv_layers"] = self.num_conv_layers
#         config["min_data_width"] = self.min_data_width
#         return config


# class ParticleAttentionNet(Model):
#     def __init__(self, attention_neurons=100, hidden_layers=1, hidden_neurons=100, activation="relu",
#                  min_filters=4, filter_width=5, pooling="mean", pooling_width=2, filter_growth_rate=2,
#                  output_num=2, **kwargs):
#         super(ParticleAttentionNet, self).__init__(**kwargs)
#         self.attention_neurons = attention_neurons
#         self.hidden_layers = hidden_layers
#         self.hidden_neurons = hidden_neurons
#         self.activation = activation
#         self.min_filters = min_filters
#         self.pooling = pooling
#         self.pooling_width = pooling_width
#         self.filter_growth_rate = filter_growth_rate
#         self.filter_width = filter_width
#         self.output_num = output_num
#         self.particle_encoder = ParticleEncoder(self.hidden_layers, self.hidden_neurons, self.activation,
#                                                attention_neurons=self.attention_neurons, name="particle_encoder")
#         self.particle_decoder = ParticleDecoder(self.hidden_layers, self.hidden_neurons, self.activation,
#                                                 output_num=self.output_num, name="particle_decoder")
#         self.hologram_encoder = HologramEncoder(min_filters=self.min_filters, filter_width=self.filter_width,
#                                                 filter_growth_rate=filter_growth_rate, pooling_width=pooling_width,
#                                                 activation=activation, attention_neurons=attention_neurons,
#                                                 name="hologram_encoder")
#         self.particle_attention = Attention(use_scale=True)
#         return

#     def call(self, inputs, **kwargs):
#         particle_enc = self.particle_encoder(inputs[0])
#         # particle_enc.shape = batch_size x proposed_num_particles x self.attention_neurons 
#         holo_enc = self.hologram_encoder(inputs[1])
#         # holo_enc.shape = batch_size x area_of_final_conv_layer x self.attention_neurons
#         attention_out = self.particle_attention([particle_enc, holo_enc])
#         # attention_out.shape = batch_size x proposed_num_particles x self.attention_neurons
#         particle_dec = self.particle_decoder(attention_out)
#         # particle_dec.shape = batch_size x proposed_num_particles x num_coordinates (x,y,z,d,p(particle_is_real)<optional)
#         return particle_dec
    
# def generate_gaussian_particles(num_images=1000, num_particles=5, image_size_pixels=100,
#                                 gaussian_sd=3, random_seed=124):
#     np.random.seed(random_seed)
#     particle_pos = np.random.random(size=(num_images, num_particles, 2)).astype(np.float32)
#     holo = np.zeros((num_images, image_size_pixels, image_size_pixels, 1), dtype=np.float32)
#     for i in range(num_images):
#         for p in range(num_particles):
#             holo[i, int(particle_pos[i, p, 0] * image_size_pixels),
#                  int(particle_pos[i, p, 1] * image_size_pixels)] = 1
#         holo[i] = gaussian_filter(holo[i], gaussian_sd)
#     return particle_pos, holo


# def run_particleattentionnet():
#     net = ParticleAttentionNet()
#     num_images = 1000
#     num_particles = 5
#     image_size_pixels = 100
#     filter_size = 3
#     noise_sd = 0.2
#     particle_pos, holo = generate_gaussian_particles(num_images=num_images, num_particles=num_particles,
#                                 image_size_pixels=image_size_pixels, gaussian_sd=filter_size)
#     particle_pos_noisy = particle_pos * (1 + np.random.normal(0, noise_sd, particle_pos.shape))
#     net.compile(optimizer="adam", loss=custom_losses["noisy"], metrics="noisy")
#     net.fit([particle_pos_noisy, holo], particle_pos, epochs=15, batch_size=32, verbose=1)
#     pred_particle_pos = net.predict([particle_pos_noisy, holo], batch_size=128)
#     import matplotlib.pyplot as plt
#     plt.ion()
#     plt.contourf(holo[0, :, :, 0], cmap="Blues")
#     plt.scatter(pred_particle_pos[0, :, 1] * image_size_pixels,
#                 pred_particle_pos[0, :, 0] * image_size_pixels, 10, "r")
#     plt.scatter(particle_pos_noisy[0, :, 1] * image_size_pixels,
#                 particle_pos_noisy[0, :, 0] * image_size_pixels, 10, "g")
#     plt.savefig("pred_particle.png", dpi=150, bbox_inches="tight")
#     return



# if __name__ == "__main__":
#     run_particleattentionnet()
