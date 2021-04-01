import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn.functional import relu
import matplotlib.pyplot as plt

class BaseActor(nn.Module):
    # def __init__(self, state_dims, action_dims, output_scale=1.0):
    #     # super(BaseActor, self).__init__()
    #     self.state_dims = state_dims
    #     self.action_dims = action_dims        
    #     self.output_scale = output_scale        

    # def init(self, bclass, instance):
    #     super(bclass, instance).__init__()

    def forward(self, x):
        pass

    def visualize_feature_maps(self, x, model_name, image_no):
        pass

    def clone_parameters(self, model):
        raise Exception

    def update_weights_from_model(self, model, tau=1.0):
        for parameter, tgt_parameter in zip(self.parameters(), model.parameters()):
            parameter.data.copy_(tgt_parameter.data * tau +  parameter.data* (1.0 - tau))

    def augmentation_capable(self):
        return False

    def memory_capable(self):
        return False

class BaseCritic(nn.Module):
    # def __init__(self, state_dims, action_dims):
    #     # super(BaseCritic, self).__init__()        
    #     self.state_dims = state_dims
    #     self.action_dims = action_dims                        

    # def init(self, bclass, instance):
    #     super(bclass, instance).__init__()

    def forward(self, states, actions):
        pass

    def Q1(self, states, actions):
        pass

    def clone_parameters(self, model):
        raise Exception

    def update_weights_from_model(self, model, tau=1.0):
        for parameter, tgt_parameter in zip(self.parameters(), model.parameters()):
            parameter.data.copy_(tgt_parameter.data * tau +  parameter.data* (1.0 - tau))

    def augmentation_capable(self):
        return False

    def memory_capable(self):
        return False