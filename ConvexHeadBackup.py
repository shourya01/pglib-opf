import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

# THE NON MONOTONIC PART USES MNN

class ConvexNet(nn.Module):
    
    def __init__(self, layer_sizes: Tuple, activation = nn.Softplus, dimV: int = 10, monotonicity = 1, dtype = torch.float32):
        
        super(ConvexNet,self).__init__()
        
        # Monotone network
        layer_sizes_grad = tuple([itm if idx!=len(layer_sizes)-1 else layer_sizes[0] for idx,itm in enumerate(layer_sizes)])
        
        self.gradWeights = nn.ParameterList()
        self.gradBiases = nn.ParameterList()
        
        for idx in range(len(layer_sizes)-1):
            self.gradWeights.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],layer_sizes_grad[idx],dtype=dtype)))
            self.gradBiases.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],dtype=dtype)))
            
        self.activation2 = activation()
        
        self.monotonicity = monotonicity
        
        # monotone gradient network
        self.V = nn.Parameter(torch.zeros(dimV,layer_sizes_grad[0]))
            
    def forward(self, x, mode: str = 'monotonic'):
        
        # in this setup, the forward() returns the gradients
        if mode == 'monotonic':
        
            for w,b in zip(self.gradWeights,self.gradBiases):
                x = F.linear(x,weight=self.monotonicity*torch.square(w),bias=b) # monotone increasing gradient implies convex function
                x = self.monotonicity * self.activation2(x) # monotone increasing (please ensure that activation is an increasing function)
                    
            return x
        
        else:
            
            return torch.matmul(x,self.V.t() @ self.V)       
    
    def forwardCost(self, x):
        
        # triangular one-step integration to calculate cost
        
        raise NotImplementedError
        
        
    
    