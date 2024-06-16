import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

# THE NON MONOTONIC PART USES MNN

class ConvexNet(nn.Module):
    
    def __init__(self, layer_sizes: Tuple, activation = nn.Softplus, dimV: int = 1, monotonicity = 1, dtype = torch.float32):
        
        super(ConvexNet,self).__init__()
        
        # mixture of experts head
        
        self.Router = nn.Sequential(
            nn.Linear(layer_sizes[0],100),
            nn.Softplus(),
            nn.Linear(100,100),
            nn.Softplus(),
            nn.Linear(100,layer_sizes[0]),
        )
        
        # Monotone network
        layer_sizes_grad = tuple([itm if idx!=len(layer_sizes)-1 else layer_sizes[0] for idx,itm in enumerate(layer_sizes)])
        
        self.gradWeights = nn.ParameterList()
        self.gradBiases = nn.ParameterList()
        
        for idx in range(len(layer_sizes)-1):
            self.gradWeights.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],layer_sizes_grad[idx],dtype=dtype)))
            self.gradBiases.append(nn.Parameter(torch.zeros(layer_sizes_grad[idx+1],dtype=dtype)))
            
        self.activation = activation()
        
        self.monotonicity = monotonicity
        
        # monotone gradient network
        self.V = nn.Parameter(torch.zeros(dimV,layer_sizes_grad[0]))
            
    def forward(self, x, mode: str = 'inference'):
        
        # in this setup, the forward() returns the gradients
        # options for mode:
        # train_monotonic
        # train_monotone
        # train_router
        # inference
        
        # monotonic expert
        if mode == 'train_monotonic':
            for w,b in zip(self.gradWeights,self.gradBiases):
                x = F.linear(x,weight=self.monotonicity*torch.square(w),bias=b) # monotone increasing gradient implies convex function
                x = self.monotonicity * self.activation(x) # monotone increasing (please ensure that activation is an increasing function)
            return x
        
        # monotone expert
        if mode == 'train_monotone':    
            return torch.matmul(x,self.V.t() @ self.V)    
        
        # router  
        if mode == 'train_router':      
            with torch.no_grad():
                x_monotonic_expert = self.forward(x,mode='train_monotonic')
                x_monotone_expert = self.forward(x,mode='train_monotone')
            x = torch.stack((x_monotonic_expert,x_monotone_expert),dim=1)
            expert_weights = F.softmax(self.Router(x),dim=1)
            return x_monotonic_expert*expert_weights[:,1,:] + x_monotone_expert*expert_weights[:,0,:]
        
        # inference
        if mode == 'inference':
            with torch.no_grad():
                x = self.forward(x,mode='train_router')
            return x
        
    def forwardCost(self, x):
        
        # triangular one-step integration to calculate cost
        
        raise NotImplementedError
        
        
    
    