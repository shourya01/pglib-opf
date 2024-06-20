import torch
import torch.nn as nn

class SplitLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None, num_cuda: int = 0):
        
        super(SplitLinear,self).__init__()
        
        self.lin1 = nn.Linear(in_features,in_features // 10, bias=False, dtype = dtype)
        self.lin2 = nn.Linear(in_features // 10, out_features, bias=bias, dtype = dtype)
                
    def forward(self, x):
        
        return self.lin2(self.lin1(x))
        
    def get_weight(self):
        
        return self.lin2.weight @ self.lin1.weight
            

