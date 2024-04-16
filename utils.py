import torch
import torch.nn as nn

def make_data_parallel(model:nn.Module, sz: int):
    
    if torch.cuda.is_available():
        # empty cache
        torch.cuda.empty_cache()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and sz > 30000:
        return nn.DataParallel(model)
    else:
        return model