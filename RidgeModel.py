import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd.functional import jacobian
import torch.nn.functional as F

class RidgeNet(nn.Module):
    
    def __init__(self, input_size = int, output_size = int):
        
        super(RidgeNet,self).__init__()
        
        self.linear = nn.Linear(in_features = input_size, out_features = output_size, bias = False)
            
    def forward(self, x): 
                
        return self.linear(x)