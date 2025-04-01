import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution_layer import GraphConvolutionLayer
from readouts.basic_readout import readout_function

"""
  Base paper: https://arxiv.org/abs/1609.02907
"""

class GCN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(GCN, self).__init__()
        """
        La GCN és un model espacial que servirà per grafs dirigits i no dirigits.
        Arguments:
          > n_feat: dimensió dels atributs dels nodes (int)
          > n_class: nombre de classes sobre les que classificarem (int)
          > n_layer: nombre de capes convolucionals (int)
          > agg_hidden: 
          > fc_hidden:
          > dropout:
          > readout:
          > device: ()

        We define the model for an imput X (node features matrix) and a matrix A.
        If we don't modify the matrix A we are using the adjacency matrix. Other 
        options are the Laplacian or the Normalitzed Laplacian. The extension for this 
        is not defined yet.

        Note that the dimension of A is n x n where n is the number of nodes and X is a matrix 
        of dimension n x n_feat. 
        """
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout  
        
        # Graph convolution layer
        self.graph_convolution_layers = []
        for i in range(n_layer):
           if i == 0:
             exec(f"self.graph_convolution_layer_{i} = GraphConvolutionLayer(n_feat, agg_hidden, device)")
             exec(f"self.graph_convolution_layers.append(self.graph_convolution_layer_{i})")
             
           else:
             exec(f"self.graph_convolution_layer_{i} = GraphConvolutionLayer(agg_hidden, agg_hidden, device)")
             exec(f"self.graph_convolution_layers.append(self.graph_convolution_layer_{i})")
        
        # Fully-connected layer
        if n_layer > 0:
          self.fc1 = nn.Linear(agg_hidden, n_class)
        else: self.fc1 = nn.Linear(agg_hidden, n_class)
        
    
    def forward(self, data):

        x, _, adj = data[:3]
        for i in range(self.n_layer):
           # Graph convolution layer
           x = F.relu(self.graph_convolution_layers[i](x, adj))
                      
           # Dropout ULTIMA CAPA
           if i != self.n_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Readout
        x = readout_function(x, self.readout)
        
        # Fully- layer
        x = F.softmax(self.fc1(x))
        

        return x
        
    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_convolution_layers[i])
            layers += '\n'
        layers += str(self.fc1) + '\n'
        return layers
    
    def help(self):
       print('The function help for an explanation is not defined\n if u want an explanation please go to the source code')
       
            
            