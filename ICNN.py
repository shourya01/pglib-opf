import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

#this is an alternative parameterization (see [Huang,21] -- ref in paper)
class posLinear(nn.Module):

    def __init__(self,in_dim,out_dim,bias):
        super(posLinear,self).__init__()

        self.lin = nn.Linear(in_dim,out_dim)
    
    def forward(self,x):
        return F.linear(x, torch.square(self.lin.weight.data),bias=self.lin.bias)

# this is a typical scalar input convex neural network (ICNN)
class ICNN(nn.Module):
    def __init__(self, in_dim, hidden,layers=1,pos=False):
        super(ICNN,self).__init__()

        self.hidden = hidden
        self.pos = pos
        if pos:
            self.lin = nn.ModuleList([
                    posLinear(in_dim,hidden,bias=False),
                    *[posLinear(hidden,hidden,bias=False) for _ in range(layers)],
                    posLinear(hidden,1,bias=False)
          ])
        else:
          self.lin = nn.ModuleList([
                  nn.Linear(in_dim,hidden,bias=False),
                  *[nn.Linear(hidden,hidden,bias=False) for _ in range(layers)],
                  nn.Linear(hidden,1,bias=False)
          ])


        self.res = nn.ModuleList([
                *[nn.Linear(in_dim,hidden) for _ in range(layers)],
                nn.Linear(in_dim,1)
        ])
        self.act = nn.Softplus()
    
    def scalar(self, x):
        y = x.clone()
        y = self.act(self.lin[0](y))
        for (core,res) in zip(self.lin[1:-1],self.res[:-1]):
            y = self.act(core(y) + res(x))
        
        y = self.lin[-1](y) + self.res[-1](x)
        return y
    
    def init_weights(self,mean,std):
        with torch.no_grad():
            for core in self.lin:
                core.weight.data.normal_(mean,std).exp_()

    
    #this clips the weights to be non-negative to preserve convexity 
    def wclip(self):
        if self.pos: return
        with torch.no_grad():
            for core in self.lin:
                core.weight.data.clamp_(0)
    
    def forward(self,x):
        with torch.enable_grad():
            x_ = x.clone().detach()
            x_.requires_grad = True
            grad = torch.autograd.grad(self.scalar(x_).sum(), x_,retain_graph=True, create_graph=True)[0]

        return grad