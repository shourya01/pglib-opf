import torch
import torch.nn as nn

class SplitLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None, num_cuda: int = 0):
        
        super(SplitLinear,self).__init__()
        
        self.num_cuda = num_cuda
        
        if num_cuda == 0:
            self.lin = nn.Linear(in_features=in_features,out_features=out_features,bias=bias,dtype=dtype)
        else:
            self.lin = nn.ModuleList()
            out_szs = [out_features//num_cuda for _ in range(num_cuda)]
            out_szs[-1] += (out_features-sum(out_szs))
            for cuda_idx, sz in enumerate(out_szs):
                self.lin.append(nn.Linear(in_features=in_features,out_features=sz,bias=bias,dtype=dtype).to(f"cuda:{cuda_idx}"))
                
            self.move_op_done = False
                
    def forward(self, x):
        
        # device check
        if self.num_cuda != 0 and self.move_op_done == False:
            for midx, mod in enumerate(self.lin):
                if self.get_device_(mod) != f"cuda:{midx}":
                    mod.to(f"cuda:{midx}")
            self.move_op_done = True
        
        if self.num_cuda == 0:
            return self.lin(x)
        else:
            x_outs = [self.lin[cuda_idx](x.clone().to(f"cuda:{cuda_idx}")).to(f"cuda:0") for cuda_idx in range(self.num_cuda)]
            return torch.cat(x_outs,dim=-1)
        
    def get_device_(self,module):
        
        return str(next(module.parameters()).device)
    
    def to(self, device):
        
        super(SplitLinear,self).to(device)
        self.move_op_done = False
        
    def get_weight(self):
        
        return torch.vstack([mod.weight.clone().to(self.get_device_(self.lin[0])) for mod in self.lin]) if self.num_cuda != 0 else self.lin.weight
            
