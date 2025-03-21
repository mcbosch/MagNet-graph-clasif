import math

import torch
import torch.nn as nn

"""
References: https://github.com/tkipf/pygcn
"""

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=False):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight, learnable
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Escala els paràmetres, ja que al crearlos aleatoris, no volem 
        valors massa disparats.
        '''

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        
        '''
        X represent the nodes attributs, Nevertherles is a tensor of 3 sizes; 
        Thus if we want to represent it as a matrix for the convolution we have 
        to reshape
        '''
        x = x.reshape(x.size()[0] * x.size()[1], x.size()[2])
        x = torch.mm(x, self.weight)
        x = x.reshape(adj.size()[0], adj.size()[1], self.weight.size()[-1])

        '''
        torch.bmm és una multiplicació de tensors 3-D --> Adj és un tensor 3-D?
        Podem superposar potèncias de la matriu per tenir els diferents nivells.
        '''
        output = torch.bmm(adj, x)

        '''
        Why we work with this format of x?
        '''
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

