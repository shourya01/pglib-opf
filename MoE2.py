import torch
import torch.nn as nn
from CGN import ConvexGradientNetFull
from ICNN import ICNN
from typing import List, Tuple, Union

EPS = 1e-5
DUAL_THRES = 1e-4

class MoE(nn.Module):
    
    def __init__(self,in_dim:int,V_dim:int,activations: Union[List,Tuple],hidden_dim:int,layers:int=1,pos:bool=True):
        
        super(MoE,self).__init__()
        
        self.CGN = ConvexGradientNetFull(dim_in = in_dim, dim_V = V_dim, activations = activations)
        self.ICNN = ICNN(in_dim = in_dim, hidden=hidden_dim,layers=layers,pos=pos)
        
        self.expertICNN = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            activations[0](),
            nn.Linear(hidden_dim,in_dim)
        )
        self.expertGCN = nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            activations[0](),
            nn.Linear(hidden_dim,in_dim)
        )
        
        # self.norm2 = lambda x,y: ((x**2 + EPS) / (x**2 + y**2 + EPS), (y**2 + EPS) / (x**2 + y**2 + EPS))
        
    def forwardCGN(self, x):
        
        return self.CGN(x)
    
    def forwardICNN(self, x):
        
        return self.ICNN(x)
    
    def forward(self, x):
        
        xCGN = self.CGN(x).detach()
        xICNN = self.ICNN(x).detach()
        
        expICNN = self.expertICNN(x)**2
        expCGN = self.expertICNN(x)**2
        
        return expICNN*xICNN+expCGN*xCGN
        
    
    