import torch
import torch.nn as nn
from CGN1 import ConvexGradientNet1
from typing import List, Tuple, Union

class ConvexGradientNetFull(nn.Module):
    
    def __init__(self, dim_in:int, dim_V: int, activations: Union[List,Tuple]):
        
        super(ConvexGradientNetFull,self).__init__()
        
        self.dim_in, self.dim_V = dim_in, dim_V
        self.CGNunits = nn.ModuleList()
        
        for act in activations:
            self.CGNunits.append(ConvexGradientNet1(in_dim=self.dim_in,activation=act))
            
        self.lowRankV = nn.Parameter(torch.randn(self.dim_in,self.dim_V))
        self.bias = nn.Parameter(torch.randn(self.dim_in))
        
    def get_lr_mat(self):
        
        return self.lowRankV @ self.lowRankV.t()
    
    def forward(self, x):
        
        outs = [mod(x) for mod in self.CGNunits]
        units = sum(outs)
        vtvx = torch.matmul(x,self.get_lr_mat())
        return (vtvx + units) if outs != [] else (self.bias + vtvx)
    
    