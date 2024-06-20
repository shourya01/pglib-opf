import torch
import torch.nn as nn
from SplitLinear import SplitLinear

class ConvexGradientNet1(nn.Module):
    
    def __init__(self, in_dim: int, activation: nn.Module):
        
        super(ConvexGradientNet1,self).__init__()
        
        self.activationFn = activation()
        self.lin = SplitLinear(in_dim,in_dim,num_cuda=torch.cuda.device_count()) 
        
    def act_(self, x):
        
        return self.activationFn(x)
    
    def d_act_(self, x):
        
        y = self.act_(x)
        dx = torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y),create_graph=True)[0]
        return dx
    
    def forward(self, x):
        
        z = self.lin(x)
        
        x = self.act_(z)
        y = self.d_act_(z)
        W = self.lin.get_weight().t()
        
        return x * torch.matmul(y,W)
        
        
    
    
    
    