r"""
This layerc consist of a polinomial
of order K to aproximate a wavelet filter using the Magnetic Laplacian.
"""

import torch, math
import numpy as np
import torch.nn as nn

class MagNet_layer(nn.Module):

    def __init__(self, 
                 in_features, 
                 out_features, 
                 device, 
                 bias = True, 
                 K=1, 
                 simetric = True, 
                 Matrix = 'Laplacian_N', 
                 q = 0.25):

        r"""
        To define this layer we use 
            Parameteers:
        
        :param: in_features -> The 3-dimension of the tensor *node features*
        :param: out_features-> The 3-dimension of the output tensor *node features* 
        :param: device      -> Where we compute the calculations
        :param: bias        -> We add a bias factor
        :param: K           -> The order of the polinomial
        :param: simetric    -> If K = 1 and c0 = -c1
        :param: Matrix      -> The GSO we use to define
        """
        super(MagNet_layer, self).__init__()
        self.order = K
        self.q = q
        self.in_features = in_features # Number of signals we want to process
        self.out_features = out_features # Number of kernels
    
        self.weight = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.order+1)).to(device) for _ in range(out_features)] )
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
        for param in self.weight:
            param.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, X_real, X_imag, L_real, L_imag):

        sizes = L_imag.size()
        I = torch.stack([torch.eye(sizes[1]) for _ in range(sizes[0])]).to(device=self.device)

        X_real = X_real.reshape(X_real.size()[0]*X_real.size()[1], X_real.size()[2])
        X_imag = X_imag.reshape(X_imag.size()[0]*X_imag.size()[1], X_imag.size()[2])
        X_real = torch.mm(X_real, self.weight) 
        X_imag = torch.mm(X_imag, self.weight)

        X_real = X_real.reshape(sizes[0], sizes[1], self.weight.size()[-1])
        X_imag = X_imag.reshape(sizes[0], sizes[1], self.weight.size()[-1])
        H_real = I - L_real
        H_imag = -L_imag

        for i in range(self.order+1):
            