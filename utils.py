import torch
import torch.nn as nn

DUAL_THRES = 1e-4

def make_data_parallel(model:nn.Module, sz: int):
    
    if torch.cuda.is_available():
        # empty cache
        torch.cuda.empty_cache()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and sz > 5000:
        return nn.DataParallel(model)
    else:
        return model 
    
def create_gating_network_targets(gt, model1out, model2out):
    
    # here assuming that model1 is the 'preferred' model
    # and you should only revery to model2 when model1 is wrong
    
    out = torch.ones_like(gt)
    
    out = torch.where((model1out != gt) & (model2out == gt),
                    torch.zeros_like(gt), torch.ones_like(gt))
    out.to(model1out.device)
    
    return out

def thres_tensor(ten):
    
    return torch.where(torch.abs(ten)<DUAL_THRES,torch.zeros_like(ten).to(ten.device),torch.ones_like(ten).to(ten.device))
    