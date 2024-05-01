import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

class ConvexNet(nn.Module):
    
    def __init__(self, layer_sizes: Tuple, activation = nn.GELU, monotonicity = 1, dtype = torch.float32):
        
        super(ConvexNet,self).__init__()
        
        layer_sizes_grad = tuple([itm if idx!=len(layer_sizes)-1 else layer_sizes[0] for idx,itm in enumerate(layer_sizes)])
        
        self.gradWeights = nn.ParameterList()
        self.gradBiases = nn.ParameterList()
        
        for idx in range(len(layer_sizes)-1):
            self.gradWeights.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],layer_sizes_grad[idx],dtype=dtype)))
            self.gradBiases.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],dtype=dtype)))
            
        self.activation = activation()
        
        self.monotonicity = monotonicity
            
    def forward(self, x):
        
        # in this setup, the forward() returns the gradients
        
        for w,b in zip(self.gradWeights,self.gradBiases):
            x = F.linear(x,weight=self.monotonicity*torch.square(w),bias=b) # monotone increasing gradient implies convex function
            x = self.monotonicity * self.activation(x) # monotone increasing (please ensure that activation is an increasing function)
                
        return x
    
    def forwardCost(self, x):
        
        # triangular one-step integration to calculate cost
        
        raise NotImplementedError
        
        
    
    