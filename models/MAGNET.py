import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.graph_cheb import MagNet_layer, complex_relu_layer
from readouts.basic_readout import readout_function



class MagNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(MagNet, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        self.CReLU = complex_relu_layer()
        
        # Graph convolution layer
        self.layers = nn.ModuleList([
                MagNet_layer(n_feat, agg_hidden, device) if i == 0 else 
                MagNet_layer(agg_hidden, agg_hidden, device) for i in range(n_layer)
            ])
        
        # Fully-connected layer
        self.fc1 = nn.Linear(2*agg_hidden, agg_hidden)
        self.fc2 = nn.Linear(agg_hidden, n_class)

  
    def forward(self, data):

        x_real,x_imag, L_real, L_imag = data[:4]
        sizes = x.size()
        x_real = x 
        x_imag = data[-1]

        for i in range(self.n_layer):
            # Graph convolution layer
            x_real, x_imag = self.CReLU((self.layers[i](x_real, x_imag, L_real, L_imag)))       

        # Readout
        x = readout_function([x_real, x_imag], self.readout, complex = True)
        
        # Fully- layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)

        return x
        
    def __repr__(self):
        layers = f'\033[1mMAGNET\033[0m\n'
        
        for i in range(self.n_layer):
            layers += str(self.layers[i])
            layers += '\n'
        layers += str(self.fc1) + '\n'
        return layers
       

