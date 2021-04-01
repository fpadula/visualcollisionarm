import torch.nn as nn
import numpy as np

def layer_init(layer, w):
    # Initializes layer using a uniform distribution
    nn.init.uniform_(layer.weight, -w, w)
    nn.init.uniform_(layer.bias, -w, w)

def hidden_layer_init(layer):
    # Initializes hidden layer using a uniform distribution based on
    # the input size of the layer
    fan_in = layer.in_features
    w = 1. / np.sqrt(fan_in)
    layer_init(layer, w)