import torch
import torch.nn as nn

class LinearModelWithActivation(nn.Module):
    
    def __init__(self, out_size: int, activation: nn.Module, dtype = torch.float32):
        
        super(LinearModelWithActivation,self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_size,dtype=dtype))
        self.activation = activation()
        
    def forward(self,x,wt_mat):
        
        return self.activation(torch.matmul(x,wt_mat))+self.bias
    
class splitNetwork(nn.Module):
    
    def __init__(self, in_size: int, num_layers: int, activation: nn.Module, num_cuda: int, dtype = torch.float32, V_dim: int = 50):
        
        super(splitNetwork,self).__init__()
        
        assert in_size % num_cuda == 0, "Input size is not a multiple of the number of CUDA devices available."
        
        self.in_size, self.num_layers, self.num_cuda = in_size, num_layers, num_cuda
        self.V_dim = V_dim
        
        self.moduleList = nn.ModuleList()
        self.partialWtList = nn.ParameterList()
        
        for deviceID in range(self.num_cuda):
            self.partialWtList.append(nn.Parameter(torch.zeros(in_size,in_size//num_cuda,dtype=dtype).to(f"cuda:{deviceID}")))
        
        for layerID in range(self.num_layers):
            
            for deviceID in range(self.num_cuda):
                self.moduleList.append(LinearModelWithActivation(in_size//num_cuda,activation,dtype).to(f"cuda:{deviceID}"))
            
        #low-rank V matrix
        self.U = nn.Parameter(torch.randn(self.in_size,self.V_dim,dtype=dtype).to('cuda:0'))
        self.D = nn.Parameter(torch.randn(self.V_dim,dtype=dtype).to('cuda:0'))
        
    def forward(self,x):
        
        x = x.to('cuda:0')
        
        # create V in advance
        V = self.U @ torch.diag(self.D) @ self.U.t()
        xV = torch.matmul(x,V)
        
        for tidx in range(self.num_layers):
            for cidx in range(self.num_cuda):
                x_splits = [self.moduleList[tidx*self.num_cuda+cidx](x.clone().to(f"cuda:{cidx}"),self.partialWtList[cidx]) for cidx in range(self.num_cuda)]
                if tidx == 0:
                    x_splits = [x.to('cuda:0') for x in x_splits]
                    x = torch.cat(x_splits,dim=-1)
                else:
                    x_splits = [x.to('cuda:0') for x in x_splits]
                    x = x + torch.cat(x_splits,dim=-1)
        
        # chunk x into num_cuda pieces
        x_splits = torch.split(x,x.shape[-1]//self.num_cuda,dim=-1)
        x_splits = [x.to(f"cuda:{i}") for i,x in enumerate(x_splits)]
        
        # multiply by wt transpose
        x_splits = [torch.matmul(x,self.partialWtList[i].t()) for i,x in enumerate(x_splits)]
        x_splits = [x.to('cuda:0') for x in x_splits]
        xfinal = sum(x_splits) + xV
        
        return xfinal
        
        
            
        