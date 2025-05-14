import torch, math
import torch.nn as nn
import numpy as np

r"""
    We define a layer taking care that we recieve as input the adjacency matrix and 
    the node atributes as a 3 dimensional tensor (the data batched). Moreover, this 
    layer is defined for complex attributes. 
"""


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
        self.simetric = simetric
        self.in_features = in_features
        self.out_features = out_features
        self.q = q
        self.device = device

        if K == 1 and simetric:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        else:
            self.weight = nn.Parameter(torch.FloatTensor(K+1, in_features, out_features)).to(device)

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

    def forward(self, X_real, X_imag, L_real, L_imag):
        '''
        Define the matrix Asked

        Now we use by defect l_max ~ 2
        '''
        # Define the Magnetic Laplacian Normalized
        sizes = L_real.size()
        # Fer breakpoint -> tenim un  tensor 3-dimensional?
        
        I = torch.stack([torch.eye(sizes[1]) for _ in range(sizes[0])]).to(device=self.device)
        # Scale the magnetic laplacian with l_max ~ 2
        if self.order == 1 and self.simetric:

            X_real = X_real.reshape(X_real.size()[0]*X_real.size()[1], X_real.size()[2])
            X_imag = X_imag.reshape(X_imag.size()[0]*X_imag.size()[1], X_imag.size()[2])
            X_real = torch.mm(X_real, self.weight) 
            X_imag = torch.mm(X_imag, self.weight)

            X_real = X_real.reshape(sizes[0], sizes[1], self.weight.size()[-1])
            X_imag = X_imag.reshape(sizes[0], sizes[1], self.weight.size()[-1])

            H_real = I - L_real
            H_imag = -L_imag
            output = [torch.bmm(H_real, X_real) - torch.bmm(H_imag,X_imag),torch.bmm(H_imag,X_real)+torch.bmm(H_real,X_imag)]
            
            if self.bias is not None:
               output[0], output[1] = output[0] + self.bias, output[1] + self.bias

            return output[0], output[1]
            
            

    def __repr__(self):
        return self.__class__.__name__ + '(' \
                    + str(self.in_features) +'->' \
                    + str(self.out_features) + ')'


class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img
