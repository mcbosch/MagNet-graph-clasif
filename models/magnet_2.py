import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.magnetic_chebs import MagNet_layer, complex_relu_layer
from readouts.basic_readout import readout_function

class MagNet2(nn.Module):
    def __init__(self, 
                 n_feat, 
                 n_class, 
                 n_layer, 
                 agg_hidden,  
                 dropout, 
                 readout, 
                 device, 
                 order = 1, 
                 freq = [0.25],
                 simetric = True):
        super(MagNet2, self).__init__()

        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        self.order = order
        self.simetric = simetric
        self.CReLU = complex_relu_layer()

        if len(freq) != self.n_layer:
            self.freqs = [[] for _ in range(self.n_layer)]
        else: self.freqs = freq
        # Graph convolution layer
        self.layers = nn.ModuleList([
                MagNet_layer(n_feat, agg_hidden, device, K=order, simetric = simetric, frequencies=self.freqs[0]) if i == 0 else 
                MagNet_layer(agg_hidden, agg_hidden, device, K=order, simetric = simetric, frequencies=self.freqs[i]) for i in range(n_layer)
            ])
        
        # Fully-connected layer
        self.fc1 = nn.Linear(2*n_feat*agg_hidden*agg_hidden, 128)
        self.fc2 = nn.Linear(128, n_class)

  
    def forward(self, data):
        x_real, _, adj = data[:3]


        for i in range(self.n_layer):
            # Graph convolution layer
            x_real, x_imag = self.CReLU((self.layers[i](x_real, x_imag, adj)))       

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
       

