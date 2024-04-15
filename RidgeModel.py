import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

class RidgeNet(nn.Module):
    
    def __init__(self, input_size = int):
        
        super(RidgeNet,self).__init__()
        
        self.linear = nn.Linear(in_features = input_size, out_features = input_size, bias = False)
            
    def forward(self, x): 
                
        return self.linear(x)
    
    def l2_param(self):
        
        l2_reg = None
        
        for p in self.parameters():
            
            if l2_reg is None:
                l2_reg = 0.5*torch.sum(  torch.pow(p,2)  )
            else:
                l2_reg = l2_reg + 0.5*torch.sum(  torch.pow(p,2)  )
                
        return l2_reg