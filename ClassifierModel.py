import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

class ClassifierNet(nn.Module):
    
    def __init__(self, layer_sizes: Tuple, activation = nn.GELU):
        
        super(ClassifierNet,self).__init__()
        
        layer_sizes_grad = tuple([itm if idx!=len(layer_sizes)-1 else layer_sizes[0] for idx,itm in enumerate(layer_sizes)])
        
        self.Weights = nn.ParameterList()
        self.Biases = nn.ParameterList()
        
        for idx in range(len(layer_sizes)-1):
            self.Weights.append(nn.Parameter(torch.ones(layer_sizes_grad[idx+1],layer_sizes_grad[idx])))
            self.Biases.append(nn.Parameter(torch.ones(layer_sizes_grad[idx+1])))
            
        self.activation = activation()
            
    def forward(self, x):
        
        # in this setup, the forward() returns the unnormalized logits for classification
        
        for idx, (w,b) in enumerate(zip(self.Weights,self.Biases)):
            x = F.linear(x,weight=w,bias=b) 
            if idx == len(self.Weights)-1:
                pass
            else:
                x = self.activation(x)
                
            x = 1 - torch.sigmoid(x) 
                
        return x
        
        
    
    