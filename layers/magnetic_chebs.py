import torch, math
import torch.nn as nn
import numpy as np
import sys

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
                 frequencies = []):
    
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
        self.device = device

        if len(frequencies) != self.out_features: 
            if frequencies == []: self.frequencies = [4]*self.out_features
            else: 
                print(f'Please enter a valid freqüencies\n It must be a list of length {self.out_features}.\n')
                sys.exit()
        else: self.frequencies = frequencies

     
        self.weight = nn.ParameterList([nn.Parameter(torch.FloatTensor(K+1, in_features)).to(device) for _ in range(out_features)]).to(device)
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

        for pa in self.weight: 
                stdv = 1. / math.sqrt(pa.size(1))
                pa.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    @staticmethod
    def ad2MagL(Adj, q, normalized=True):

        As = 0.5*(Adj + Adj.T)
        D = [np.sum(As[i]) for i in range(len(As))]
        D_norm = np.diag([np.power(D[i], -0.5) if D[i] != 0 else 0 for i in range(len(D))])
        T = 2*math.pi*q*(Adj - Adj.T) 
        T = np.cos(T) + np.sin(T)*1.0j
        I = np.eye(len(Adj)) 

        if normalized:
            L_n = I - (np.matmul(np.matmul(D_norm, As),D_norm))*T
            L_n_real, L_n_imag = L_n.real, L_n.imag 
            return L_n_real, L_n_imag
        else:
            L =  np.diag(D) -  As*T 
            return L.real, L.imag
    
    def compute_mag(self, adj, idx):
        r"""
        This function recieves a 3-d tensor consisting of
        adjacency matrixs representing diferent graphs. It computes 
        the magnetic Laplacian of frequenci 1/self.q
        """
        sizes = adj.size()
        real, imag = [], []
        q = 1/self.frequencies[idx]
        for i in range(sizes[0]):
            A = adj[i].copy()
            L_real, L_imag = self.ad2MagL(A, q)
            real.append(L_real)
            imag.append(L_imag)
        L_real = torch.stack(real).to(device=self.device)
        L_imag = torch.stack(imag).to(device=self.device)

    def forward(self, X_real, X_imag, adj):
        '''
        Define the matrix Asked

        Now we use by defect l_max ~ 2
        '''

       
        for idx in range(self.out_features):
            L_real, L_imag = self.compute_mag(adj, idx)
            # Define the Magnetic Laplacian Normalized
            sizes = L_real.size()
            # Fer breakpoint -> tenim un  tensor 3-dimensional?
            
            I = torch.stack([torch.eye(sizes[1]) for _ in range(sizes[0])]).to(device=self.device)
            # Scale the magnetic laplacian with l_max ~ 2
            
            Y_real = torch.zeros(sizes[0],sizes[1],sizes[2]).to(self.device)
            Y_imag =  torch.zeros(sizes[0],sizes[1],sizes[2]).to(self.device)
            H_real = I - L_real
            H_imag = -L_imag
            cheb_pols_real = [I, H_real]
            cheb_pols_imag = [torch.zeros(sizes[0],sizes[1],sizes[2]).to(self.device), H_imag]
            for i in range(self.order+1):
                if i>1:
                    Y_real = 2*(torch.bmm(L_real,cheb_pols_real[i-1])-torch.bmm(L_imag,cheb_pols_imag[i-1]))+cheb_pols_real[i-2]
                    Y_imag = 2*(torch.bmm(L_real,cheb_pols_imag[i-1])+torch.bmm(L_imag,cheb_pols_real[i-1]))+cheb_pols_imag[i-2]
                    cheb_pols_imag.append(Y_imag)
                    cheb_pols_real.append(Y_real)
            
            for inf in range(self.in_features):
                signal_real = X_real[:,:,inf].clone().to(self.device)
                signal_imag = X_imag[:,:,inf].clone().to(self.device)
                batch = signal_imag.size()[0]
                nodes = signal_imag.size()[1]
                output_real = torch.zeros(batch,nodes).to(self.device)
                output_imag = torch.zeros(batch,nodes).to(self.device)

                for i in range(self.order +1):
                    output_real = output_real + self.weight[idx][i][inf]*(torch.bmm(cheb_pols_real[i], signal_real) - torch.bmm(cheb_pols_imag[i],signal_imag))
                    output_imag = output_imag + self.weight[idx][i][inf]*(torch.bmm(cheb_pols_imag[i], signal_real) - torch.bmm(cheb_pols_real[i],signal_imag))

                if self.bias is not None:
                    output_real= output_real + self.bias

                if idx == 0:
                    Output_real = output_real.clone()
                    Output_imag = output_imag.clone()
                else:
                    Output_real = torch.stack([Output_real,output_real], dim=2)
                    Output_imag = torch.stack([Output_imag,output_imag], dim=2)

        return Output_real, Output_imag
                
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
