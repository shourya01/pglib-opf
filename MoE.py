import torch
import torch.nn as nn
from ICGN import ICGN
from ICNN import ICNN
from typing import List, Tuple, Union
from SplitLinear import SplitLinear
from CGN import ConvexGradientNetFull

EPS = 1e-5

class MoE(nn.Module):
    
    def __init__(self,in_dim:int,V_dim:int,activations: Union[List,Tuple],hidden_dim:int,layers:int=1,pos:bool=True):
        
        super(MoE,self).__init__()
        
        self.ICGN = ConvexGradientNetFull(dim_in=in_dim,dim_V=200,activations=[]).to('cuda')
        # self.ICGN = ICGN(in_dim = in_dim, hidden = hidden_dim,layers=0)
        self.ICNN = ICNN(in_dim = in_dim, hidden = hidden_dim,layers=layers,pos=pos)
        
        self.gating = nn.Sequential(
            SplitLinear(in_dim,hidden_dim),
            nn.PReLU(num_parameters=hidden_dim),
            SplitLinear(hidden_dim,hidden_dim),
            nn.PReLU(num_parameters=hidden_dim),
            SplitLinear(hidden_dim,in_dim),
            nn.Sigmoid()
        )
        
        # self.norm2 = lambda x,y: ((x**2 + EPS) / (x**2 + y**2 + EPS), (y**2 + EPS) / (x**2 + y**2 + EPS))
        
    def forwardICGN(self, x, N=20):
        
        # return self.ICGN(x, N=N)
        return self.ICGN(x)
    
    def forwardICNN(self, x):
        
        return self.ICNN(x)
    
    def forwardGating(self, x):
        
        return self.gating(x)
    
    def forward(self, x, N=20):
        
        # xICGN = self.ICGN(x, N=N)
        xICGN = self.ICGN(x)
        xICNN = self.ICNN(x)
        
        g = self.gating(x)

        return g*xICNN + (1-g)*xICGN
        # return xICNN
        
    
    